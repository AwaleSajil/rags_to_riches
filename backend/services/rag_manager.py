"""Per-user MoneyRAG instances, held only as long as they are worth holding.

An instance is expensive to keep: it owns a temp directory on disk, a merchant
cache, a LangGraph checkpointer that grows with every chat thread, and live LLM
and embedding clients. It is also expensive to rebuild, which is why they are
cached at all.

The cache used to be a plain dict that only ever grew — nothing evicted it
except a config change or shutdown. On a single-instance deploy (which the
in-memory state here requires) that is a memory and disk leak proportional to
every user who has ever sent a message. This adds the two bounds that stop it:
an idle timeout, and a hard ceiling on how many are resident at once.

An instance that is CURRENTLY streaming a reply is never evicted. Its temp
directory holds the chart and image handoff files the running chat is about to
read, so removing it mid-answer would delete the output from under the request.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

# Add project root to path so we can import money_rag.py directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from money_rag import MoneyRAG

logger = logging.getLogger("moneyrag.services.rag_manager")

# How long an untouched instance survives. Long enough that a user reading an
# answer and then asking a follow-up does not pay the rebuild, short enough that
# a day's worth of one-question visitors does not stay resident.
IDLE_TTL_SECONDS = 30 * 60

# A ceiling regardless of the TTL, because a burst of new users inside one TTL
# window would otherwise still exhaust the box. Least-recently-used goes first.
MAX_INSTANCES = 50

# One in-flight generation per user. Every chat spawns an MCP subprocess that
# imports pandas, plotly and langchain, so concurrency here is measured in
# hundreds of megabytes, not in threads. A client stuck in a retry loop is the
# realistic way this gets hit, and the cost of that is the whole process.
MAX_CONCURRENT_CHATS_PER_USER = 1


class ChatBusy(Exception):
    """Raised when a user already has a generation in flight."""


class RAGManager:
    """Manages per-user MoneyRAG instances, replacing Streamlit session_state."""

    def __init__(self):
        self._instances: dict[str, MoneyRAG] = {}
        # user_id -> monotonic timestamp of last use, for TTL and LRU.
        self._last_used: dict[str, float] = {}
        # user_id -> number of generations currently streaming.
        self._active: dict[str, int] = {}
        logger.debug("RAGManager initialized — empty instance cache")

    async def get_or_create(self, user: dict, config: dict) -> MoneyRAG:
        user_id = user["id"]
        if user_id not in self._instances:
            logger.info(
                "Creating new MoneyRAG instance for user_id=%s — provider=%s, model=%s, embedding=%s",
                user_id,
                config["llm_provider"],
                config.get("decode_model", "gemini-3-flash-preview"),
                config.get("embedding_model", "gemini-embedding-001"),
            )
            self._instances[user_id] = MoneyRAG(
                llm_provider=config["llm_provider"],
                model_name=config.get("decode_model", "gemini-3-flash-preview"),
                embedding_model_name=config.get("embedding_model", "gemini-embedding-001"),
                api_key=config["api_key"],
                user_id=user_id,
                access_token=user.get("access_token"),
            )
            logger.debug("MoneyRAG instance created for user_id=%s", user_id)
        else:
            logger.debug("Reusing cached MoneyRAG instance for user_id=%s", user_id)

        self._last_used[user_id] = time.monotonic()
        # Enforced on the way in, so the ceiling holds even if the periodic
        # sweep is not running (tests, or a worker whose lifespan never started).
        await self._enforce_ceiling()
        logger.debug("Active RAG instances: %d", len(self._instances))
        return self._instances[user_id]

    def active_chats(self, user_id: str) -> int:
        """How many generations this user has in flight right now."""
        return self._active.get(user_id, 0)

    @asynccontextmanager
    async def chatting(self, user_id: str):
        """Mark a generation in flight, and refuse a second one.

        Also pins the instance against eviction for the duration — its temp
        directory is where the running chat writes its chart and image output.
        """
        if self._active.get(user_id, 0) >= MAX_CONCURRENT_CHATS_PER_USER:
            raise ChatBusy(user_id)
        self._active[user_id] = self._active.get(user_id, 0) + 1
        try:
            yield
        finally:
            remaining = self._active.get(user_id, 1) - 1
            if remaining > 0:
                self._active[user_id] = remaining
            else:
                self._active.pop(user_id, None)
            # A generation that ran for a while should not be evicted the
            # instant it finishes just because it started long ago.
            self._last_used[user_id] = time.monotonic()

    async def sweep(self) -> int:
        """Evict instances nobody has used lately. Returns how many went."""
        cutoff = time.monotonic() - IDLE_TTL_SECONDS
        stale = [
            uid for uid, last in self._last_used.items()
            if last < cutoff and not self._active.get(uid)
        ]
        for uid in stale:
            await self.invalidate(uid)
        if stale:
            logger.info(
                "Swept %d idle RAG instance(s) — %d remain", len(stale), len(self._instances)
            )
        return len(stale)

    async def _enforce_ceiling(self) -> None:
        """Drop least-recently-used instances until back under MAX_INSTANCES."""
        while len(self._instances) > MAX_INSTANCES:
            evictable = [uid for uid in self._instances if not self._active.get(uid)]
            if not evictable:
                # Everything resident is mid-generation. Refusing to evict is
                # correct — deleting a busy instance's temp dir breaks a live
                # request — so log it rather than looping forever.
                logger.warning(
                    "RAG instance ceiling (%d) exceeded but all are busy", MAX_INSTANCES
                )
                return
            oldest = min(evictable, key=lambda uid: self._last_used.get(uid, 0.0))
            logger.info("Evicting LRU RAG instance for user_id=%s (over ceiling)", oldest)
            await self.invalidate(oldest)

    async def invalidate(self, user_id: str):
        if user_id in self._instances:
            logger.info("Invalidating RAG instance for user_id=%s", user_id)
            try:
                await self._instances[user_id].cleanup()
                logger.debug("Cleanup succeeded for user_id=%s", user_id)
            except Exception as e:
                logger.warning("Cleanup failed for user_id=%s: %s", user_id, e, exc_info=True)
            del self._instances[user_id]
            self._last_used.pop(user_id, None)
            logger.debug("RAG instance removed — %d active instances remain", len(self._instances))
        else:
            logger.debug("No RAG instance to invalidate for user_id=%s", user_id)

    async def cleanup_all(self):
        logger.info("Cleaning up all RAG instances — %d active", len(self._instances))
        for uid in list(self._instances):
            await self.invalidate(uid)
        logger.info("All RAG instances cleaned up")


rag_manager = RAGManager()


async def run_janitor(interval_seconds: int = 300) -> None:
    """Periodically release what nobody is using any more.

    Sweeping only when a request arrives leaves an idle box holding every
    instance and temp directory from its last busy period — which is exactly
    when a small container gets restarted for memory. This runs on a timer so
    quiet periods actually free memory.
    """
    from backend.services import capture_service

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await rag_manager.sweep()
            # Abandoned photo captures hold bytes on disk on the same box, and
            # were only ever swept when someone uploaded another one.
            await asyncio.to_thread(capture_service.sweep_pending)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            # A janitor that dies takes the memory bound with it, silently.
            logger.error("Janitor pass failed: %s", e, exc_info=True)

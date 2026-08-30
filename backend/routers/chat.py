import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.dependencies import get_current_user
from backend.schemas.chat import ChatRequest
from backend.services import config_service, conversation_service
from backend.services.rag_manager import (
    MAX_CONCURRENT_CHATS_PER_USER,
    ChatBusy,
    rag_manager,
)
from backend.services import stream_gate as markers
from backend.services.stream_gate import MarkerGate, extract_blocks

logger = logging.getLogger("moneyrag.routers.chat")

router = APIRouter()

# Preview LLM models intermittently return transient server errors (5xx /
# UNAVAILABLE / overloaded). Auto-retry the generation a couple of times before
# surfacing an error, but only while nothing has streamed to the client yet.
_MAX_CHAT_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.5
# Tokens are held back for this long at the start of a run. Streaming and the
# retry above are in tension: once a token reaches the client the request cannot
# be restarted without duplicating text, so retries are off from that moment.
# Nearly every transient provider failure happens before the first token, and
# three quarters of a second is imperceptible, so this buys the retry back for
# almost nothing.
_RETRY_GRACE_SECONDS = 0.75
# Wall-clock ceiling on one question. The agent already has a 40-step recursion
# limit, but steps are not time: a provider that accepts the connection and then
# never answers leaves the run hanging with an MCP subprocess attached. A long
# comparison across several places legitimately takes a minute or two, so this
# is set well above normal and is a backstop, not a budget.
_CHAT_TIMEOUT_SECONDS = 300
_TRANSIENT_ERROR_KEYS = (
    "500", "502", "503", "internal", "unavailable", "overloaded",
    "temporarily", "deadline", "timeout", "timed out", "try again",
)


def _parse_blocks(blocks: list[str], label: str) -> list:
    """Parse marker payloads as JSON, dropping any the model mangled.

    One unparseable block must not cost the user the whole answer, so a bad one
    is logged and skipped rather than raised.
    """
    parsed = []
    for block in blocks:
        try:
            parsed.append(json.loads(block))
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s JSON", label)
    return parsed


def _is_transient(exc: Exception) -> bool:
    """True for provider errors that a retry is likely to clear (not quota/auth)."""
    low = str(exc).lower()
    if any(k in low for k in ("quota", "resource_exhausted", "rate limit", "429",
                              "api key", "permission_denied", "unauthenticated", "401", "403")):
        return False
    return any(k in low for k in _TRANSIENT_ERROR_KEYS)


def _friendly_error(exc: Exception) -> str:
    """Turn a raw provider exception into a short, human message for the chat UI."""
    low = str(exc).lower()
    # The agent ran out of steps. Nothing is wrong with the service, and telling
    # someone to "try again in a moment" sends them to repeat a question that
    # will fail the same way — the useful advice is to narrow it.
    if "recursion limit" in low or "graphrecursion" in type(exc).__name__.lower():
        return (
            "That question took more steps than I'm allowed to spend, so I "
            "stopped before finishing. Try narrowing it — one place or one "
            "time period at a time usually gets there."
        )
    if any(k in low for k in ("resource_exhausted", "429", "quota", "too many requests", "rate limit")):
        return (
            "You've hit your AI provider's rate limit (the Gemini free tier allows "
            "~20 requests/day). Please wait a moment and try again, or enable billing "
            "on your Google Cloud project."
        )
    if "no longer available" in low or "not_found" in low or ("404" in low and "model" in low):
        return (
            "The selected AI model isn't available for your API key. Open Settings and "
            "switch the model to “gemini-3-flash-preview”."
        )
    if any(k in low for k in ("api key not valid", "api_key_invalid", "permission_denied", "unauthenticated", "401", "403")):
        return "Your AI provider API key looks invalid or unauthorized — please re-check it in Settings."
    if any(k in low for k in ("deadline", "timeout", "timed out")):
        return "The AI provider took too long to respond. Please try again."
    return "Something went wrong while generating a response. Please try again in a moment."


@router.post("")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    logger.debug("Chat request from user_id=%s | message=%s", user["id"], body.message[:100])

    config = await config_service.get_config(user)
    if not config:
        logger.warning("No config found for user_id=%s — returning 400", user["id"])
        raise HTTPException(status_code=400, detail="Account config required. Please configure your API key first.")

    # Rejected before any work is done — notably before a conversation row is
    # created, so a client retry-storm cannot fill the table with empty threads.
    if rag_manager.active_chats(user["id"]) >= MAX_CONCURRENT_CHATS_PER_USER:
        logger.warning("Rejecting concurrent chat for user_id=%s", user["id"])
        raise HTTPException(
            status_code=429,
            detail="You already have a question in progress. Wait for it to finish, then try again.",
        )

    rag = await rag_manager.get_or_create(user, config)

    # --- Conversation setup: resume an existing one or start a new one ---
    if body.conversation_id:
        conversation_id = body.conversation_id
    else:
        conv = await conversation_service.create_conversation(user)
        conversation_id = conv["id"]
    logger.debug("Chat in conversation_id=%s for user_id=%s", conversation_id, user["id"])

    # Load prior turns for agent context, then persist this user message.
    prior = await conversation_service.get_messages(user, conversation_id)
    history = conversation_service.to_agent_history(prior)
    await conversation_service.add_message(
        user, conversation_id, "user", body.message,
        bill_file_ids=body.bill_file_ids or None,
    )
    await conversation_service.set_title_from_first_message(user, conversation_id, body.message)

    async def _stream_once():
        """One full generation attempt, yielding SSE strings. Raises on failure."""
        event_count = 0
        start = time.perf_counter()
        # Never lets a partial ===MARKER=== or its contents reach the screen.
        gate = MarkerGate()
        # Tokens are withheld for the first moment of the run so a provider that
        # fails early can still be retried silently — see _RETRY_GRACE_SECONDS.
        withheld: list[str] = []
        released = False

        async for event in rag.chat(body.message, history=history):
            event_count += 1

            if event["type"] == "token":
                safe = gate.feed(event.get("text", ""))
                if not safe:
                    continue
                if not released:
                    withheld.append(safe)
                    if time.perf_counter() - start < _RETRY_GRACE_SECONDS:
                        continue
                    safe = "".join(withheld)
                    withheld.clear()
                    released = True
                yield f"event: token\ndata: {json.dumps({'text': safe})}\n\n"
                continue

            if event["type"] == "final":
                # Whatever the gate is still holding is text that never became a
                # marker. Release it so a short answer is not swallowed whole.
                tail = "".join(withheld) + gate.flush()
                if tail:
                    yield f"event: token\ndata: {json.dumps({'text': tail})}\n\n"
                    withheld.clear()

                content = event.get("content", "")

                # A chart is forwarded as the raw Plotly JSON the tool wrote —
                # this layer never needs to look inside it.
                content, charts = extract_blocks(content, markers.CHART)

                # Each images block is a JSON ARRAY of URLs, so several blocks
                # flatten into one list.
                content, blocks = extract_blocks(content, markers.IMAGES)
                images = [url for group in _parse_blocks(blocks, "images") for url in group]

                content, blocks = extract_blocks(content, markers.CONFIRM_TX)
                pending_transactions = _parse_blocks(blocks, "pending transaction")

                content, blocks = extract_blocks(content, markers.CONFIRM_FIX)
                pending_corrections = _parse_blocks(blocks, "pending correction")

                final_content = content.strip()

                # Persist the assistant turn so it survives reloads/restarts.
                try:
                    await conversation_service.add_message(
                        user, conversation_id, "assistant", final_content,
                        charts=charts or None,
                        images=images or None,
                        pending_transactions=pending_transactions or None,
                    )
                except Exception as e:
                    logger.error("Failed to persist assistant message: %s", e, exc_info=True)

                yield (
                    "event: final\ndata: "
                    + json.dumps({
                        "content": final_content,
                        "charts": charts,
                        "images": images,
                        "pendingTransactions": pending_transactions,
                        "pendingCorrections": pending_corrections,
                    })
                    + "\n\n"
                )
            else:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "SSE stream complete for user_id=%s conv=%s — %d events in %.1fms",
            user["id"], conversation_id, event_count, elapsed_ms,
        )
        yield "event: done\ndata: {}\n\n"

    async def _attempts():
        """Retry a transient provider failure a couple of times, but only while
        nothing has streamed yet — once tokens/tool events reach the client we
        can't cleanly restart, so we surface the error instead of duplicating."""
        for attempt in range(_MAX_CHAT_ATTEMPTS):
            streamed_any = False
            try:
                async for sse in _stream_once():
                    streamed_any = True
                    yield sse
                return
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                transient = _is_transient(e)
                can_retry = transient and not streamed_any and attempt < _MAX_CHAT_ATTEMPTS - 1
                logger.error(
                    "SSE stream error for user_id=%s (attempt %d/%d, transient=%s, retrying=%s): %s",
                    user["id"], attempt + 1, _MAX_CHAT_ATTEMPTS, transient, can_retry, e,
                    exc_info=True,
                )
                if can_retry:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                yield f"event: error\ndata: {json.dumps({'error': _friendly_error(e)})}\n\n"
                return

    async def event_generator():
        # Let the client know which conversation this maps to (esp. brand-new ones).
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"

        try:
            # Holds the per-user slot for the whole generation, and pins the RAG
            # instance so the janitor cannot delete the temp directory this run
            # is still writing its chart and image output into.
            async with rag_manager.chatting(user["id"]):
                deadline = time.monotonic() + _CHAT_TIMEOUT_SECONDS
                agen = _attempts().__aiter__()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        # Bound each step, not just the run: an agent wedged on a
                        # provider that never responds would otherwise hold a
                        # subprocess and a connection open indefinitely.
                        sse = await asyncio.wait_for(agen.__anext__(), timeout=remaining)
                    except StopAsyncIteration:
                        break
                    yield sse
        except ChatBusy:
            # Lost a race with another request between the check above and here.
            logger.warning("Concurrent chat rejected mid-setup for user_id=%s", user["id"])
            yield (
                "event: error\ndata: "
                + json.dumps({"error": "You already have a question in progress."})
                + "\n\n"
            )
        except asyncio.TimeoutError:
            logger.error(
                "Chat timed out after %ds for user_id=%s conv=%s",
                _CHAT_TIMEOUT_SECONDS, user["id"], conversation_id,
            )
            yield (
                "event: error\ndata: "
                + json.dumps({
                    "error": (
                        "That question took too long and I had to stop. Try narrowing "
                        "it — one place or one time period at a time usually gets there."
                    )
                })
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

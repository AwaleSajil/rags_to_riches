import sys
import os

# Add project root to path so we can import money_rag.py directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from money_rag import MoneyRAG


class RAGManager:
    """Manages per-user MoneyRAG instances, replacing Streamlit session_state."""

    def __init__(self):
        self._instances: dict[str, MoneyRAG] = {}

    async def get_or_create(self, user: dict, config: dict) -> MoneyRAG:
        user_id = user["id"]
        if user_id not in self._instances:
            self._instances[user_id] = MoneyRAG(
                llm_provider=config["llm_provider"],
                model_name=config.get("decode_model", "gemini-3-flash-preview"),
                embedding_model_name=config.get("embedding_model", "gemini-embedding-001"),
                api_key=config["api_key"],
                user_id=user_id,
                access_token=user.get("access_token"),
            )
        return self._instances[user_id]

    async def invalidate(self, user_id: str):
        if user_id in self._instances:
            try:
                await self._instances[user_id].cleanup()
            except Exception as e:
                print(f"Warning: cleanup failed for user {user_id}: {e}")
            del self._instances[user_id]

    async def cleanup_all(self):
        for uid in list(self._instances):
            await self.invalidate(uid)


rag_manager = RAGManager()

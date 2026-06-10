from langchain_core.tools import tool

from app.core.logging import get_logger
from app.db.database import AsyncSessionLocal
from app.memory.sqlite_backend import SQLAlchemyMemoryStore

logger = get_logger(__name__)


@tool
async def get_user_memory(user_id: str) -> str:
    """
    Retrieves stored long-term cross-session facts and summarized context about a specific user.
    Use this tool if the user refers to past conversations, asks what you discussed before,
    or uses vague pronouns like "that" or "it" referring to a past topic.

    Args:
        user_id: The unique identification string of the customer.
    """
    logger.info("tool_get_user_memory_invoked", user_id=user_id)

    try:
        async with AsyncSessionLocal() as session:
            memory_store = SQLAlchemyMemoryStore(session)
            summary = await memory_store.get_user_summary(user_id)

        if summary:
            logger.info("tool_get_user_memory_success", user_id=user_id, found=True)
            return f"Past context and facts for user {user_id}: {summary}"

        logger.info("tool_get_user_memory_success", user_id=user_id, found=False)
        return f"No past context found for user {user_id}. Treat this as a brand new conversation."

    except Exception as e:
        logger.error("tool_get_user_memory_failed", error=str(e), user_id=user_id)
        return "System Error: Unable to retrieve user memory at this time. Proceed without past context."
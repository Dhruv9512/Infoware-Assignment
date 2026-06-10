from sqlalchemy import update

from app.core.logging import get_logger
from app.db.database import AsyncSessionLocal
from app.db.models import MessageTable
from langchain_core.tools import tool

logger = get_logger(__name__)


@tool
async def flag_for_human(user_id: str, session_id: str, reason: str) -> dict:
    """
    Escalates the current conversation session for manual human intervention.

    Args:
        user_id: The unique identifier of the customer.
        session_id: The active conversation tracking UUID.
        reason: The operational rationale provided by the agent for the handoff.
    """
    logger.warning(
        "conversation_escalated_to_human",
        user_id=user_id,
        session_id=session_id,
        escalation_reason=reason,
    )

    try:
        async with AsyncSessionLocal() as db_session:
            stmt = (
                update(MessageTable)
                .where(MessageTable.session_id == session_id)
                .values(flagged=True, eval_reasoning=f"Human Handoff Triggered: {reason}")
            )
            await db_session.execute(stmt)
            await db_session.commit()

        return {
            "status": "escalated",
            "message": "Human support notification dispatched successfully.",
            "reason_logged": reason,
            "session_id": session_id,
        }

    except Exception as e:
        logger.error("human_escalation_persistence_failed", error=str(e), session_id=session_id)
        return {
            "status": "error",
            "message": f"Escalation failed to persist, but system was alerted: {str(e)}",
        }
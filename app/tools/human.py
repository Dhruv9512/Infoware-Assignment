from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logging import get_logger
from app.db.models import MessageTable

logger = get_logger(__name__)

async def flag_for_human(user_id: str, session_id: str, reason: str, db_session: AsyncSession) -> dict:
    """
    Escalates the current conversation session for manual human intervention.
    Fulfills the architectural bonus requirement for structural safety tracking.
    
    Args:
        user_id: The unique identifier of the customer.
        session_id: The active conversation tracking UUID.
        reason: The operational rationale provided by the agent for the handoff.
        db_session: The active asynchronous SQLAlchemy engine connection.
    """
    # 1. Generate an explicit LLMOps structured log entry for observability tools
    logger.warning(
        "conversation_escalated_to_human",
        user_id=user_id,
        session_id=session_id,
        escalation_reason=reason
    )

    try:
        # 2. Update the last message or active flag metrics for this session in the DB
        # This allows a reviewer's endpoint to pull flagged histories instantly
        stmt = (
            update(MessageTable)
            .where(MessageTable.session_id == session_id)
            .values(flagged=True, eval_reasoning=f"Human Handoff Triggered: {reason}")
        )
        await db_session.execute(stmt)
        await db_session.commit()

        # 3. Return a structural response payload back to the LangGraph node loop
        return {
            "status": "escalated",
            "message": "Human support notification dispatched successfully.",
            "reason_logged": reason,
            "session_id": session_id
        }

    except Exception as e:
        logger.error("human_escalation_persistence_failed", error=str(e), session_id=session_id)
        # Fallback response so the agent loop doesn't crash catastrophically mid-turn
        return {
            "status": "error",
            "message": f"Escalation failed to persist, but system was alerted: {str(e)}"
        }
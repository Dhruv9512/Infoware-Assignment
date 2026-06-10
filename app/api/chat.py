from fastapi import APIRouter, Depends, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.schemas import (
    ChatRequest, 
    ChatResponse, 
    UserHistoryResponse, 
    PerformanceMetricsResponse
)
from app.services.chat_service import build_chat_service
from app.memory.sqlite_backend import SQLAlchemyMemoryStore
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/{user_id}", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    user_id: str = Path(..., description="The unique identifier of the user"),
    db: AsyncSession = Depends(get_db)
):
    """Core endpoint: Sends a message to the agent and returns the text and eval block."""
    return await build_chat_service(db_session= db, user_id=user_id, request=request)


@router.get("/{user_id}/history", response_model=UserHistoryResponse)
async def get_user_history(
    user_id: str = Path(..., description="The unique identifier of the user"),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves the full conversational history and evaluation audits for a user."""
    memory_store = SQLAlchemyMemoryStore(db)
    history = await memory_store.get_conversation_history(user_id, limit=100)
    return UserHistoryResponse(user_id=user_id, history=history)


@router.delete("/{user_id}/memory")
async def clear_user_memory(
    user_id: str = Path(..., description="The unique identifier of the user"),
    db: AsyncSession = Depends(get_db)
):
    """GDPR-style wipe: Deletes all memory summaries and chat histories for a user."""
    memory_store = SQLAlchemyMemoryStore(db)
    await memory_store.clear_user_memory(user_id)
    logger.info("user_memory_wiped", user_id=user_id)
    return {"status": "success", "message": f"Memory completely wiped for user {user_id}."}


@router.get("/{user_id}/evals", response_model=PerformanceMetricsResponse)
async def get_user_evals(
    user_id: str = Path(..., description="The unique identifier of the user"),
    db: AsyncSession = Depends(get_db)
):
    """Bonus Endpoint: Aggregates evaluation metrics across all sessions for a user."""
    memory_store = SQLAlchemyMemoryStore(db)
    history = await memory_store.get_conversation_history(user_id, limit=1000)

    # Filter for AI responses that include evaluation scores
    eval_msgs = [msg for msg in history if msg["role"] == "assistant" and msg.get("groundedness") is not None]

    if not eval_msgs:
        return PerformanceMetricsResponse(
            user_id=user_id, total_responses_evaluated=0,
            average_groundedness=0.0, average_relevance=0.0,
            average_confidence=0.0, total_flagged_escalations=0
        )

    total = len(eval_msgs)
    avg_g = sum(m["groundedness"] for m in eval_msgs) / total
    avg_r = sum(m["relevance"] for m in eval_msgs) / total
    avg_c = sum(m["confidence"] for m in eval_msgs) / total
    flagged = sum(1 for m in eval_msgs if m["flagged"])

    return PerformanceMetricsResponse(
        user_id=user_id,
        total_responses_evaluated=total,
        average_groundedness=round(avg_g, 2),
        average_relevance=round(avg_r, 2),
        average_confidence=round(avg_c, 2),
        total_flagged_escalations=flagged
    )
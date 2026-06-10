import json
from typing import Any, Dict, List, Optional
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MessageTable, UserMemorySummaryTable
from app.memory.base import BaseMemoryStore


class SQLAlchemyMemoryStore(BaseMemoryStore):
    """
    Concrete implementation of the memory store interface utilizing 
    SQLAlchemy for relational database persistence (SQLite/PostgreSQL).
    """

    def __init__(self, db_session: AsyncSession):
        self.session = db_session

    async def get_conversation_history(
        self, 
        user_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(MessageTable)
            .where(MessageTable.user_id == user_id)
            .order_by(MessageTable.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages = result.scalars().all()

        return [
            {
                "id": msg.id,
                "session_id": msg.session_id,
                "role": msg.role,
                "content": msg.content,
                "tools_called": json.loads(msg.tools_called),
                "created_at": msg.created_at,
                "groundedness": msg.groundedness,
                "relevance": msg.relevance,
                "confidence": msg.confidence,
                "flagged": msg.flagged,
                "eval_reasoning": msg.eval_reasoning,
            }
            for msg in messages
        ]

    async def get_user_summary(self, user_id: str) -> Optional[str]:
        stmt = select(UserMemorySummaryTable.summary).where(UserMemorySummaryTable.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def save_message_turn(
        self, 
        user_id: str, 
        session_id: str, 
        role: str, 
        content: str,
        tools_called: List[str],
        catalog_context: str,
        evaluation: Dict[str, Any]
    ) -> None:
        # First ensure the user profile record exists (Upsert logic skeleton)
        summary_exists = await self.get_user_summary(user_id)
        if summary_exists is None:
            new_user = UserMemorySummaryTable(user_id=user_id, summary="")
            self.session.add(new_user)
            await self.session.flush()

        # Insert the message turn
        new_message = MessageTable(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            tools_called=json.dumps(tools_called),
            catalog_context=catalog_context,
            groundedness=evaluation.get("groundedness"),
            relevance=evaluation.get("relevance"),
            confidence=evaluation.get("confidence"),
            flagged=evaluation.get("flagged", False),
            eval_reasoning=evaluation.get("reasoning", "")
        )
        self.session.add(new_message)
        await self.session.commit()

    async def update_user_summary(self, user_id: str, new_summary: str) -> None:
        stmt = (
            update(UserMemorySummaryTable)
            .where(UserMemorySummaryTable.user_id == user_id)
            .values(summary=new_summary)
        )
        await self.session.execute(stmt)
        await self.session.commit()

    async def clear_user_memory(self, user_id: str) -> None:
        # Due to ForeignKey cascade delete configuration on the model,
        # deleting the user summary record wipes all historical messages automatically.
        stmt = delete(UserMemorySummaryTable).where(UserMemorySummaryTable.user_id == user_id)
        await self.session.execute(stmt)
        await self.session.commit()
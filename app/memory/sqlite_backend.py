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

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    async def get_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters = [MessageTable.user_id == user_id]
        if session_id is not None:
            filters.append(MessageTable.session_id == session_id)

        stmt = (
            select(MessageTable)
            .where(*filters)
            .order_by(MessageTable.created_at.asc())
        )

        result = await self.session.execute(stmt)
        messages = result.scalars().all()

        return [
            {
                "id": msg.id,
                "session_id": msg.session_id,
                "user_message": msg.user_message,     
                "agent_message": msg.agent_message,  
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
        stmt = (
            select(UserMemorySummaryTable.summary)
            .where(UserMemorySummaryTable.user_id == user_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    async def save_message_turn(
       self, user_id: str, session_id: str, user_message: str,
        agent_message: str, tools_called: List[str], 
        catalog_context: str, evaluation: Dict[str, Any],
    ) -> None:
      
        # Ensure a user profile record exists before inserting the message
        if await self.get_user_summary(user_id) is None:
            self.session.add(UserMemorySummaryTable(user_id=user_id, summary=""))
            await self.session.flush()

        self.session.add(
            MessageTable(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                agent_message=agent_message,
                tools_called=json.dumps(tools_called),
                catalog_context=catalog_context,
                groundedness=evaluation.get("groundedness"),
                relevance=evaluation.get("relevance"),
                confidence=evaluation.get("confidence"),
                flagged=evaluation.get("flagged", False),
                eval_reasoning=evaluation.get("reasoning", ""),
            )
        )
        await self.session.commit()

    async def update_user_summary(self, user_id: str, new_summary: str) -> None:
        stmt = (
            update(UserMemorySummaryTable)
            .where(UserMemorySummaryTable.user_id == user_id)
            .values(summary=new_summary)
        )
        await self.session.execute(stmt)
        await self.session.commit()

    # -------------------------------------------------------------------------
    # Delete
    # -------------------------------------------------------------------------

    async def clear_user_memory(self, user_id: str) -> None:
        await self.session.execute(
            delete(MessageTable).where(MessageTable.user_id == user_id)
        )
        await self.session.execute(
            delete(UserMemorySummaryTable).where(UserMemorySummaryTable.user_id == user_id)
        )
        await self.session.commit()
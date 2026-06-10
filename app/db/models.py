from datetime import datetime, timezone
import uuid
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

def get_utc_now_naive():
    """Strips the timezone info to match PostgreSQL TIMESTAMP WITHOUT TIME ZONE"""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class UserMemorySummaryTable(Base):
    __tablename__ = "user_memory_summaries"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=get_utc_now_naive, 
        onupdate=get_utc_now_naive, 
        nullable=False
    )

    messages = relationship("MessageTable", back_populates="user", cascade="all, delete-orphan")

class MessageTable(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("user_memory_summaries.user_id"), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    tools_called: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    catalog_context: Mapped[str] = mapped_column(Text, default="", nullable=False)

    groundedness: Mapped[float] = mapped_column(Float, nullable=True)
    relevance: Mapped[float] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    eval_reasoning: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, # Removed timezone=True
        default=get_utc_now_naive, # Uses the new naive function
        nullable=False
    )

    user = relationship("UserMemorySummaryTable", back_populates="messages")
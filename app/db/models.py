from datetime import datetime, timezone
import uuid
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class UserMemorySummaryTable(Base):
    """
    Stores long-term aggregated facts and summaries about a specific user.
    This acts as the persistent cross-session profile storage.
    """
    __tablename__ = "user_memory_summaries"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationship back to individual message tracks
    messages = relationship("MessageTable", back_populates="user", cascade="all, delete-orphan")


class MessageTable(Base):
    """
    Stores the absolute granular, turn-by-turn conversation logs.
    Tracks sessions, actual text, tools invoked, and structural self-evaluation scores.
    """
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("user_memory_summaries.user_id"), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)    # The message text
    
    # Metadata for verification audits & evaluation logging
    tools_called: Mapped[str] = mapped_column(Text, default="[]", nullable=False)  # JSON serialized array of strings
    catalog_context: Mapped[str] = mapped_column(Text, default="", nullable=False)  # Relevant snapshot of catalog data used
    
    # Structured self-evaluation scoring data
    groundedness: Mapped[float] = mapped_column(Float, nullable=True)
    relevance: Mapped[float] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    eval_reasoning: Mapped[str] = mapped_column(Text, default="", nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=lambda: datetime.now(timezone.utc), 
        nullable=False
    )

    # Inverse relationship linking messages to their parent user profile
    user = relationship("UserMemorySummaryTable", back_populates="messages")
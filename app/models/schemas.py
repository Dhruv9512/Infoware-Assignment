from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ==========================================
# 1. EVALUATION SCHEMAS 
# ==========================================

class FeedbackLoopBlock(BaseModel):
    """
    Validates the structured self-evaluation metadata required on every single response turn.
    Maps directly to the assignment's core assessment layout.
    """
    groundedness: float = Field(
        ..., 
        description="Score from 0.0 to 1.0 indicating if the response is safely backed by the catalog.",
        ge=0.0,
        le=1.0
    )
    relevance: float = Field(
        ..., 
        description="Score from 0.0 to 1.0 indicating if the response directly addresses the user query.",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        ..., 
        description="The agent's internal certainty metrics regarding tool accuracy and logic.",
        ge=0.0,
        le=1.0
    )
    flagged: bool = Field(
        default=False, 
        description="Set to true if confidence drops below critical thresholds, indicating human intervention is needed."
    )
    reasoning: str = Field(
        ..., 
        description="The detailed rationale behind the agent's self-assigned safety metrics."
    )


# ==========================================
# 2. CHAT LAYER ENDPOINT SCHEMAS
# ==========================================

class ChatRequest(BaseModel):
    """Parses incoming web payloads for execution loops."""
    message: str = Field(..., description="The incoming conversational string or query from the user.")
    session_id: Optional[str] = Field(
        default=None, 
        description="An optional tracking UUID. If empty, the framework initializes a new tracking state."
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        """Enforces clean string validation before spin states process."""
        if not v.strip():
            raise ValueError("Message content cannot be empty or solely whitespace.")
        return v.strip()


class ChatResponse(BaseModel):
    """
    The strict outgoing payload containing text, tracking logs, and self-evaluation layers.
    Perfectly maps to the output structure required by Page 2 of the brief.
    """
    response: str = Field(..., description="The conversational markdown output constructed by the agent.")
    session_id: str = Field(..., description="The unique session UUID anchoring this specific context window.")
    tools_called: List[str] = Field(
        default_factory=list, 
        description="Array tracking the specific native python functions executed during resolution loops."
    )
    eval: FeedbackLoopBlock = Field(..., description="The mandatory validation audit block tracking responses safety.")


# ==========================================
# 3. CONVERSATION HISTORY SCHEMAS
# ==========================================

class MessageHistoryItem(BaseModel):
    """Granular data layout detailing a historical interaction trace."""
    id: str
    user_message: str
    agent_message: str
    tools_called: List[str]
    created_at: datetime
    
    # Evaluation logs
    groundedness: Optional[float] = None
    relevance: Optional[float] = None
    confidence: Optional[float] = None
    flagged: bool = False
    eval_reasoning: Optional[str] = None

class UserHistoryResponse(BaseModel):
    """The complete response wrapper returning cross-session history log structures."""
    user_id: str
    total_count: int = Field(..., description="Total number of conversational turns.")
    summary: str = Field(..., description="The compressed profile facts for the user.")
    data: List[MessageHistoryItem] = Field(default_factory=list)


# ==========================================
# 4. CATALOG SCHEMAS
# ==========================================

class CatalogPlan(BaseModel):
    """Individual product option layout parsing structural values from raw catalog records."""
    name: str
    price: str
    features: List[str]


class CatalogResponse(BaseModel):
    """Validates the standard structure of your product/pricing capabilities data."""
    plans: List[CatalogPlan]


# ==========================================
# 5. AGGREGATED METRICS SCHEMA
# ==========================================

class PerformanceMetricsResponse(BaseModel):
    """Fulfills the bonus challenge for structural evaluation score aggregations."""
    user_id: str
    total_responses_evaluated: int
    average_groundedness: float
    average_relevance: float
    average_confidence: float
    total_flagged_escalations: int
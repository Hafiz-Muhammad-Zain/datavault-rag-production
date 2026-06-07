from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


# ============================================================
# INGESTION MODELS
# ============================================================

class IngestResponse(BaseModel):
    """
    Returned after a document is successfully ingested.
    The frontend shows this to confirm the upload worked.
    """
    document_id: str
    filename: str
    total_chunks: int
    status: str                  # "complete" or "failed"
    message: str


# ============================================================
# QUERY MODELS
# ============================================================

class ChatMessage(BaseModel):
    """
    A single message in the conversation history.
    Role is either "user" or "assistant".

    Why we need this: passing conversation history to the LLM
    allows it to answer follow-up questions in context.
    Example: user asks "What is Article 33?" then "How long do I have?"
    The second question only makes sense with the first in context.
    """
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    """
    The request body for the /query endpoint.
    Contains the current question and the full conversation history.
    """
    question: str = Field(..., min_length=1, max_length=2000)

    # Last N messages of chat history sent with each query
    # Empty list on first message — grows as conversation continues
    chat_history: list[ChatMessage] = Field(default_factory=list)


class Citation(BaseModel):
    """
    A single source citation returned with every answer.
    Links the answer back to the exact chunk it came from.
    This is what makes the system legally defensible —
    every claim is traceable to a specific page of a specific document.
    """
    chunk_id: str
    source_file: str
    source_url: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    # The actual text of the chunk — shown in the UI as "source excerpt"
    excerpt: str


class QueryResponse(BaseModel):
    """
    The full response returned to the user after a query.

    Two possible states:
    1. answered=True  — system found relevant context, LLM generated answer
    2. answered=False — top chunk score < confidence threshold, system refuses
                        answer_text will be None, citations will be empty
    """
    answered: bool
    answer_text: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: Optional[float] = None

    # Scores for transparency — shown in the dashboard
    top_rrf_score: Optional[float] = None
    latency_total_ms: Optional[int] = None

    # Message shown when the system refuses to answer
    refusal_reason: Optional[str] = None


# ============================================================
# LOG MODELS (for the observability dashboard)
# ============================================================

class QueryLogEntry(BaseModel):
    """
    A single row in the live query log table on the dashboard.
    Fetched from the query_logs PostgreSQL table.
    """
    id: str
    queried_at: datetime
    query_text: str
    answered: bool
    answer_text: Optional[str] = None
    top_rrf_score: Optional[float] = None
    confidence_score: Optional[float] = None
    latency_total_ms: Optional[int] = None
    citations: Optional[list] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class QueryLogsResponse(BaseModel):
    """
    The full response from the /logs endpoint.
    Contains recent query logs + system health summary.
    """
    logs: list[QueryLogEntry]
    total_count: int


class SystemHealth(BaseModel):
    """
    System health summary shown at the top of the dashboard.
    Gives a quick view of how the system is performing.
    """
    total_queries: int
    total_answered: int
    total_refused: int
    answer_rate_pct: float
    avg_latency_ms: Optional[float] = None
    avg_confidence: Optional[float] = None


# ============================================================
# HEALTH CHECK MODEL
# ============================================================

class EvalHealth(BaseModel):
    """RAGAS evaluation health summary for the last 24 hours."""
    total_evaluated: int
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None


class HealthResponse(BaseModel):
    """
    Response from the /health endpoint.
    Used by Docker, Coolify, and monitoring tools to verify the app is alive.
    """
    status: str          # "ok" or "degraded"
    database: bool       # is PostgreSQL reachable?
    version: str = "1.0.0"

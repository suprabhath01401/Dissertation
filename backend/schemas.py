"""Pydantic request/response schemas."""
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming user chat message, optionally attached to an existing session."""
    session_id: uuid.UUID | None = None
    user_id: str = "default"
    message: str
    title: str | None = None
    use_mixtral: bool = False


class SourceInfo(BaseModel):
    """A single retrieved chunk cited as a source for an assistant response."""
    source_type: str        # "legal" | "app_docs"
    filename: str
    page: int | str
    chunk_index: int
    sac_summary: str | None = None
    score: float = 0.0


class MessageResponse(BaseModel):
    """API representation of a stored chat message, including routing and source metadata."""
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    model_used: str | None
    route_path: str | None
    sources: list[dict] | None
    temporal_facts: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    """Request payload to create a new chat session."""
    user_id: str = "default"
    title: str = "New Chat"


class SessionResponse(BaseModel):
    """API representation of a chat session."""
    id: uuid.UUID
    user_id: str
    title: str | None
    summary: str | None
    created_at: datetime
    updated_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class ConstraintResponse(BaseModel):
    """API representation of a stored session constraint."""
    id: uuid.UUID
    session_id: uuid.UUID
    constraint_type: str
    value: str
    is_active: bool
    is_permanent: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ConstraintDelete(BaseModel):
    """Request payload identifying a constraint to delete."""
    constraint_id: uuid.UUID


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    """API representation of an ingested document's metadata."""
    id: uuid.UUID
    filename: str
    source_type: str
    file_path: str
    chunk_count: int
    ingested_at: datetime
    qdrant_ids: list[str] | None
    # Count of distinct chat sessions whose messages cite this document (same
    # JSONB containment check the delete endpoint uses to report
    # `sessions_deleted`, but computed read-only here so the UI can warn
    # about it *before* the destructive delete happens). Not a model column —
    # populated by the /documents endpoint handler.
    sessions_using: int = 0

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvalResultResponse(BaseModel):
    """API representation of a single per-sample evaluation score."""
    id: uuid.UUID
    dataset: str
    config: str
    metric: str
    score: float
    run_at: datetime
    notes: str | None

    model_config = {"from_attributes": True}


class EvalRunResponse(BaseModel):
    """API representation of an aggregated evaluation run result."""
    id: uuid.UUID
    run_id: uuid.UUID
    dataset: str
    config: str
    metric: str
    score: float
    n_samples: int
    run_at: datetime
    notes: str | None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthStatus(BaseModel):
    """Health check status of each backend dependency."""
    qdrant: str
    postgres: str
    ollama: str
    cluster: str

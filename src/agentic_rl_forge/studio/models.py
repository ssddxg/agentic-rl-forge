from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from agentic_rl_forge.contracts import ContractModel

SourceStatus = Literal["pending", "processing", "ready", "failed"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
JobKind = Literal["ingest", "reindex"]
SearchMode = Literal["bm25", "hybrid_character"]


class KnowledgeBase(ContractModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    revision: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime


class Source(ContractModel):
    id: str = Field(min_length=1, max_length=80)
    knowledge_base_id: str = Field(min_length=1, max_length=80)
    original_name: str = Field(min_length=1, max_length=255)
    stored_name: str = Field(min_length=1, max_length=255)
    media_type: str = Field(default="application/octet-stream", min_length=1, max_length=255)
    size_bytes: int = Field(ge=1)
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: SourceStatus = "pending"
    error: str | None = Field(default=None, max_length=2000)
    document_count: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime


class IngestionJob(ContractModel):
    id: str = Field(min_length=1, max_length=80)
    knowledge_base_id: str = Field(min_length=1, max_length=80)
    source_id: str | None = Field(default=None, max_length=80)
    kind: JobKind
    status: JobStatus = "queued"
    error: str | None = Field(default=None, max_length=2000)
    created_at: datetime
    updated_at: datetime


class ModelSettings(ContractModel):
    """Safe-to-return model configuration; it deliberately never contains the API key."""

    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=255)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    api_key_configured: bool = False
    updated_at: datetime


class IndexStats(ContractModel):
    knowledge_base_id: str = Field(min_length=1, max_length=80)
    source_revision: int = Field(ge=0)
    generation: int = Field(ge=1)
    document_count: int = Field(ge=0)
    corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    built_at: datetime


class SearchHit(ContractModel):
    document_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    contents: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    bm25_score: float = Field(ge=0.0)
    character_score: float = Field(ge=0.0, le=1.0)
    scoring_method: SearchMode
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(ContractModel):
    index: int = Field(ge=1)
    document_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class ChatAnswer(ContractModel):
    content: str = Field(min_length=1)
    citations: tuple[Citation, ...]
    model: str = Field(min_length=1)

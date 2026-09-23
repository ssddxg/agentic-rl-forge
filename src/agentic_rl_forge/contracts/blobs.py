from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject


class BlobInfo(ContractModel):
    key: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    etag: str | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    last_modified: datetime | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_last_modified(self) -> BlobInfo:
        if self.last_modified is not None and (
            self.last_modified.tzinfo is None or self.last_modified.utcoffset() is None
        ):
            raise ValueError("blob last-modified time must be timezone-aware")
        return self


class BlobPutResult(ContractModel):
    created: bool
    blob: BlobInfo

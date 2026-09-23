from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import orjson
from pydantic import BaseModel, ConfigDict


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    def canonical_bytes(self, *, exclude: set[str] | None = None) -> bytes:
        payload = self.model_dump(mode="json", exclude=exclude or set(), exclude_none=True)
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)

    def digest(self, *, exclude: set[str] | None = None) -> str:
        return hashlib.sha256(self.canonical_bytes(exclude=exclude)).hexdigest()


JsonObject = dict[str, Any]

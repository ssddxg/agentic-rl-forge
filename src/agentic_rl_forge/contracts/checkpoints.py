from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject, new_id, utc_now


class ArtifactLocation(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class CheckpointArtifact(ContractModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    uri: str = Field(min_length=1)
    location: ArtifactLocation
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    metadata: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_file(cls, name: str, path: Path) -> CheckpointArtifact:
        resolved = path.resolve(strict=True)
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(
            name=name,
            uri=str(resolved),
            location=ArtifactLocation.LOCAL,
            sha256=digest.hexdigest(),
            size_bytes=resolved.stat().st_size,
        )


class CheckpointManifest(ContractModel):
    checkpoint_id: str = Field(
        default_factory=lambda: new_id("checkpoint"),
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
    )
    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    policy_version: str = Field(min_length=1)
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[CheckpointArtifact, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    parent_checkpoint_id: str | None = None
    dataset_manifest_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rng_state_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float] = Field(default_factory=dict)
    framework_versions: dict[str, str] = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_artifacts(self) -> CheckpointManifest:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("checkpoint artifact names must be unique")
        if self.parent_checkpoint_id == self.checkpoint_id:
            raise ValueError("checkpoint cannot be its own parent")
        return self


class CheckpointVerification(ContractModel):
    checkpoint_id: str
    valid: bool
    fully_verified: bool
    verified_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    mismatched_artifacts: tuple[str, ...] = ()
    unverified_artifacts: tuple[str, ...] = ()

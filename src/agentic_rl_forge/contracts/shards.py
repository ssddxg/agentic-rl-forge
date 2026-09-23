from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject


class TrajectoryShardRef(ContractModel):
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_path(self) -> TrajectoryShardRef:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("trajectory shard path must stay within the run directory")
        return self


class TrajectoryShardManifest(ContractModel):
    manifest_id: str = Field(pattern=r"^manifest_[0-9a-f]{24}$")
    run_id: str = Field(min_length=1)
    created_at: datetime
    policy_versions: tuple[str, ...] = Field(min_length=1)
    environment_versions: tuple[str, ...] = Field(min_length=1)
    trajectory_count: int = Field(ge=1)
    task_count: int = Field(ge=1)
    group_count: int = Field(ge=1)
    shards: tuple[TrajectoryShardRef, ...] = Field(min_length=1)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contents(self) -> TrajectoryShardManifest:
        if self.trajectory_count != len(self.shards):
            raise ValueError("trajectory_count must match shard count")
        trajectory_ids = [shard.trajectory_id for shard in self.shards]
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("manifest trajectory IDs must be unique")
        paths = [shard.relative_path for shard in self.shards]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest shard paths must be unique")
        return self


class ShardVerification(ContractModel):
    manifest_id: str
    valid: bool
    complete: bool
    verified_count: int = Field(ge=0)
    missing_trajectory_ids: tuple[str, ...] = ()
    mismatched_trajectory_ids: tuple[str, ...] = ()
    unexpected_paths: tuple[str, ...] = ()

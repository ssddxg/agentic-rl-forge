from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject
from agentic_rl_forge.contracts.provenance import DataOrigin


class TrainerTrajectoryRef(ContractModel):
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: DataOrigin
    reward: float


class TrainerBatchManifest(ContractModel):
    batch_id: str = Field(pattern=r"^batch_[0-9a-f]{24}$")
    created_at: datetime
    policy_version: str = Field(min_length=1)
    environment_versions: tuple[str, ...] = Field(min_length=1)
    source_run_id: str | None = None
    group_size: int = Field(ge=2)
    group_count: int = Field(ge=1)
    trajectory_count: int = Field(ge=2)
    format: str = "verl_trajectory_jsonl_v1"
    payload_key: str = Field(min_length=1)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_size_bytes: int = Field(ge=1)
    trajectories: tuple[TrainerTrajectoryRef, ...] = Field(min_length=2)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_groups(self) -> TrainerBatchManifest:
        if self.trajectory_count != len(self.trajectories):
            raise ValueError("trajectory_count must match trajectory references")
        if self.trajectory_count != self.group_count * self.group_size:
            raise ValueError("trajectory_count must equal group_count times group_size")
        ids = [item.trajectory_id for item in self.trajectories]
        if len(ids) != len(set(ids)):
            raise ValueError("trainer batch trajectory IDs must be unique")
        groups: dict[str, list[TrainerTrajectoryRef]] = {}
        for trajectory in self.trajectories:
            groups.setdefault(trajectory.group_id, []).append(trajectory)
        if len(groups) != self.group_count:
            raise ValueError("group_count must match trajectory groups")
        for group_id, trajectories in groups.items():
            if len(trajectories) != self.group_size:
                raise ValueError(f"group {group_id!r} has an unexpected size")
            if len({item.task_id for item in trajectories}) != 1:
                raise ValueError(f"group {group_id!r} mixes multiple tasks")
            if any(item.origin is not DataOrigin.ON_POLICY for item in trajectories):
                raise ValueError(f"group {group_id!r} contains derived data")
        return self


class TrainerBatchVerification(ContractModel):
    batch_id: str
    valid: bool
    payload_exists: bool
    payload_digest_matches: bool
    payload_size_matches: bool
    record_count_matches: bool
    trajectory_ids_match: bool
    trajectory_contents_match: bool

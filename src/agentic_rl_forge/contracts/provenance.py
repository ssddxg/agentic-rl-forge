from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject, utc_now


class DataOrigin(str, Enum):
    ON_POLICY = "on_policy"
    MCTS = "mcts"
    REJECTION_SAMPLING = "rejection_sampling"
    HINDSIGHT_RELABELED = "hindsight_relabeled"
    PRM_DATASET = "prm_dataset"
    REPLAY = "replay"
    DEMONSTRATION = "demonstration"
    IMPORTED = "imported"


class Provenance(ContractModel):
    origin: DataOrigin
    producer: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    parent_ids: tuple[str, ...] = ()
    transform: str | None = None
    manifest_digest: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lineage(self) -> Provenance:
        derived = {
            DataOrigin.MCTS,
            DataOrigin.REJECTION_SAMPLING,
            DataOrigin.HINDSIGHT_RELABELED,
            DataOrigin.PRM_DATASET,
            DataOrigin.REPLAY,
        }
        if self.origin in derived and not self.parent_ids:
            raise ValueError(f"{self.origin.value} data requires at least one parent_id")
        if self.parent_ids and not self.transform:
            raise ValueError("derived data requires a transform name")
        return self

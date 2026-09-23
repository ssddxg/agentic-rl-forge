from __future__ import annotations

import hashlib
from datetime import datetime

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts.artifacts import (
    RunArtifactGcPreview,
    RunArtifactGcRecord,
    RunArtifactReleaseState,
    RunArtifactTransportStatus,
)
from agentic_rl_forge.contracts.base import ContractModel


class RunArtifactStoreInventory(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    observed_at: datetime
    releases: tuple[RunArtifactTransportStatus, ...] = ()
    invalid_keys: tuple[str, ...] = ()
    release_count: int = Field(ge=0)
    object_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    state_counts: dict[str, int]
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> RunArtifactStoreInventory:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("release inventory observation time must be timezone-aware")
        archive_ids = tuple(item.archive_id for item in self.releases)
        if archive_ids != tuple(sorted(archive_ids)) or len(archive_ids) != len(set(archive_ids)):
            raise ValueError("release inventory archive IDs must be sorted and unique")
        if any(item.observed_at != self.observed_at for item in self.releases):
            raise ValueError("release inventory statuses must share one observation time")
        if self.invalid_keys != tuple(sorted(set(self.invalid_keys))):
            raise ValueError("release inventory invalid keys must be sorted and unique")
        if self.release_count != len(self.releases):
            raise ValueError("release inventory count does not match releases")
        if self.object_count != sum(item.object_count for item in self.releases):
            raise ValueError("release inventory object count does not match releases")
        if self.total_bytes != sum(item.total_bytes for item in self.releases):
            raise ValueError("release inventory byte count does not match releases")
        expected_counts = {
            state.value: sum(item.state is state for item in self.releases)
            for state in RunArtifactReleaseState
        }
        if self.state_counts != expected_counts:
            raise ValueError("release inventory state counts do not match releases")
        expected_digest = self.expected_state_digest(
            releases=self.releases,
            invalid_keys=self.invalid_keys,
        )
        if self.state_digest != expected_digest:
            raise ValueError("release inventory state digest does not match contents")
        return self

    @staticmethod
    def expected_state_digest(
        *,
        releases: tuple[RunArtifactTransportStatus, ...],
        invalid_keys: tuple[str, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "releases": [
                    item.model_dump(
                        mode="json",
                        exclude={"observed_at", "detail"},
                        exclude_none=True,
                    )
                    for item in releases
                ],
                "invalid_keys": invalid_keys,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(payload).hexdigest()


class RunArtifactGcPlan(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    plan_id: str = Field(pattern=r"^run_gc_plan_[0-9a-f]{24}$")
    inventory: RunArtifactStoreInventory
    min_age_seconds: float = Field(ge=1)
    candidates: tuple[RunArtifactGcPreview, ...] = ()
    candidate_count: int = Field(ge=0)
    reclaimable_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_plan(self) -> RunArtifactGcPlan:
        if self.inventory.invalid_keys:
            raise ValueError("GC plan cannot include an inventory with unclassified keys")
        if any(item.state is RunArtifactReleaseState.INVALID for item in self.inventory.releases):
            raise ValueError("GC plan cannot include an invalid release status")
        archive_ids = tuple(item.archive_id for item in self.candidates)
        if archive_ids != tuple(sorted(archive_ids)) or len(archive_ids) != len(set(archive_ids)):
            raise ValueError("GC plan candidates must be sorted and unique")
        staged_ids = {
            item.archive_id
            for item in self.inventory.releases
            if item.state is RunArtifactReleaseState.STAGED
        }
        if any(
            not item.eligible
            or item.min_age_seconds != self.min_age_seconds
            or item.archive_id not in staged_ids
            or item.observed_at != self.inventory.observed_at
            for item in self.candidates
        ):
            raise ValueError("GC plan contains an ineligible or inconsistent candidate")
        if self.candidate_count != len(self.candidates):
            raise ValueError("GC plan candidate count does not match candidates")
        if self.reclaimable_bytes != sum(item.total_bytes for item in self.candidates):
            raise ValueError("GC plan reclaimable bytes do not match candidates")
        expected_id = self.expected_plan_id(
            inventory_state_digest=self.inventory.state_digest,
            min_age_seconds=self.min_age_seconds,
            candidates=self.candidates,
        )
        if self.plan_id != expected_id:
            raise ValueError("GC plan ID does not match its contents")
        return self

    @staticmethod
    def expected_plan_id(
        *,
        inventory_state_digest: str,
        min_age_seconds: float,
        candidates: tuple[RunArtifactGcPreview, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "inventory_state_digest": inventory_state_digest,
                "min_age_seconds": min_age_seconds,
                "candidates": [
                    {
                        "archive_id": item.archive_id,
                        "state_digest": item.state_digest,
                    }
                    for item in candidates
                ],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_gc_plan_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactGcBatchIntent(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    batch_id: str = Field(pattern=r"^run_gc_batch_[0-9a-f]{24}$")
    plan: RunArtifactGcPlan
    operator: str = Field(min_length=1)
    reason: str = Field(min_length=8)
    created_at: datetime

    @model_validator(mode="after")
    def validate_intent(self) -> RunArtifactGcBatchIntent:
        if not self.plan.candidates:
            raise ValueError("GC batch intent requires at least one candidate")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("GC batch intent time must be timezone-aware")
        expected_id = self.expected_batch_id(
            plan_id=self.plan.plan_id,
            operator=self.operator,
            reason=self.reason,
        )
        if self.batch_id != expected_id:
            raise ValueError("GC batch ID does not match its intent")
        return self

    @staticmethod
    def expected_batch_id(
        *,
        plan_id: str,
        operator: str,
        reason: str,
    ) -> str:
        payload = orjson.dumps(
            {
                "plan_id": plan_id,
                "operator": operator,
                "reason": reason,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_gc_batch_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactGcBatchRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    intent: RunArtifactGcBatchIntent
    completed_at: datetime
    records: tuple[RunArtifactGcRecord, ...]
    candidate_count: int = Field(ge=1)
    deleted_object_count: int = Field(ge=0)
    already_missing_object_count: int = Field(ge=0)
    target_bytes: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_record(self) -> RunArtifactGcBatchRecord:
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("GC batch completion time must be timezone-aware")
        if self.completed_at < self.intent.created_at:
            raise ValueError("GC batch completion precedes its intent")
        if self.candidate_count != len(self.intent.plan.candidates):
            raise ValueError("GC batch candidate count does not match its plan")
        if len(self.records) != self.candidate_count:
            raise ValueError("GC batch records do not cover every candidate")
        expected_ids = tuple(item.archive_id for item in self.intent.plan.candidates)
        if tuple(item.archive_id for item in self.records) != expected_ids:
            raise ValueError("GC batch record order does not match its plan")
        for preview, record in zip(self.intent.plan.candidates, self.records, strict=True):
            if (
                record.state_digest != preview.state_digest
                or record.operator != self.intent.operator
                or record.reason != self.intent.reason
                or record.started_at < self.intent.created_at
            ):
                raise ValueError("GC batch member evidence does not match its intent")
        if self.deleted_object_count != sum(item.deleted_object_count for item in self.records):
            raise ValueError("GC batch deleted count does not match member records")
        if self.already_missing_object_count != sum(
            item.already_missing_object_count for item in self.records
        ):
            raise ValueError("GC batch missing count does not match member records")
        if self.target_bytes != sum(item.target_bytes for item in self.records):
            raise ValueError("GC batch byte count does not match member records")
        return self

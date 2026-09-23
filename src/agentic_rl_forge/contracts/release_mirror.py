from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Literal

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts.artifacts import (
    RunArtifactArchiveAttestationVerification,
    RunArtifactReleaseState,
    RunArtifactTransportCommit,
    RunArtifactTransportManifest,
    RunArtifactTransportObjectRef,
)
from agentic_rl_forge.contracts.base import ContractModel
from agentic_rl_forge.contracts.blobs import BlobInfo
from agentic_rl_forge.contracts.release_lifecycle import RunArtifactStoreInventory


class RunArtifactMirrorAction(str, Enum):
    COPY = "copy"
    REUSE = "reuse"


class RunArtifactMirrorObjectPlan(ContractModel):
    role: Literal["chunk", "checksum", "attestation", "manifest", "commit"]
    reference: RunArtifactTransportObjectRef
    source: BlobInfo
    destination: BlobInfo | None = None
    action: RunArtifactMirrorAction

    @model_validator(mode="after")
    def validate_object(self) -> RunArtifactMirrorObjectPlan:
        if self.source.key != self.reference.key:
            raise ValueError("mirror source identity does not match its reference key")
        if (
            self.source.size_bytes != self.reference.size_bytes
            or self.source.content_sha256 != self.reference.content_digest
            or self.source.etag is None
            or self.source.last_modified is None
        ):
            raise ValueError("mirror source identity is incomplete or inconsistent")
        if self.destination is None:
            if self.action is not RunArtifactMirrorAction.COPY:
                raise ValueError("missing mirror destination object must be copied")
        else:
            if (
                self.destination.key != self.reference.key
                or self.destination.size_bytes != self.reference.size_bytes
                or self.destination.content_sha256 != self.reference.content_digest
                or self.destination.etag is None
                or self.destination.last_modified is None
            ):
                raise ValueError("mirror destination identity is incomplete or inconsistent")
            if self.action is not RunArtifactMirrorAction.REUSE:
                raise ValueError("existing mirror destination object must be reused")
        return self


class RunArtifactMirrorPlan(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    plan_id: str = Field(pattern=r"^run_mirror_plan_[0-9a-f]{24}$")
    observed_at: datetime
    archive_id: str = Field(pattern=r"^run_archive_[0-9a-f]{24}$")
    commit: RunArtifactTransportCommit
    manifest: RunArtifactTransportManifest
    require_attestation: bool
    attestation_verification: RunArtifactArchiveAttestationVerification | None = None
    objects: tuple[RunArtifactMirrorObjectPlan, ...] = Field(min_length=4)
    unexpected_destination_objects: tuple[BlobInfo, ...] = ()
    copy_object_count: int = Field(ge=0)
    reuse_object_count: int = Field(ge=0)
    copy_bytes: int = Field(ge=0)
    reuse_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_plan(self) -> RunArtifactMirrorPlan:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("mirror plan observation time must be timezone-aware")
        if (
            self.commit.archive_id != self.archive_id
            or self.manifest.receipt.archive_id != self.archive_id
            or self.commit.transport_id != self.manifest.transport_id
            or self.commit.manifest.key
            != f"{RunArtifactTransportManifest.ROOT}/{self.archive_id}/manifests/"
            f"{self.manifest.transport_id}.json"
        ):
            raise ValueError("mirror plan release identities are inconsistent")
        if self.require_attestation and self.attestation_verification is None:
            raise ValueError("authenticated mirror plan requires attestation verification")
        if self.attestation_verification is not None and not self.attestation_verification.valid:
            raise ValueError("mirror plan cannot contain failed attestation verification")
        expected_roles_and_refs: list[tuple[str, RunArtifactTransportObjectRef]] = [
            ("chunk", item) for item in self.manifest.chunks
        ]
        expected_roles_and_refs.append(("checksum", self.manifest.checksum))
        if self.manifest.attestation is not None:
            expected_roles_and_refs.append(("attestation", self.manifest.attestation))
        expected_roles_and_refs.append(("manifest", self.commit.manifest))
        commit_item = self.objects[-1]
        expected_commit_key = f"{RunArtifactTransportManifest.ROOT}/{self.archive_id}/commit.json"
        commit_payload = self.commit.canonical_bytes() + b"\n"
        if (
            commit_item.role != "commit"
            or commit_item.reference.key != expected_commit_key
            or commit_item.reference.media_type != "application/json"
            or commit_item.reference.size_bytes != len(commit_payload)
            or commit_item.reference.content_digest != hashlib.sha256(commit_payload).hexdigest()
        ):
            raise ValueError("mirror commit object is not canonical or final")
        expected_roles_and_refs.append(("commit", commit_item.reference))
        if len(self.objects) != len(expected_roles_and_refs) or any(
            item.role != role or not self._same_reference(item.reference, reference)
            for item, (role, reference) in zip(
                self.objects,
                expected_roles_and_refs,
                strict=True,
            )
        ):
            raise ValueError("mirror object graph does not match the release manifest")
        object_keys = tuple(item.reference.key for item in self.objects)
        if len(object_keys) != len(set(object_keys)):
            raise ValueError("mirror object keys must be unique")
        unexpected_keys = tuple(item.key for item in self.unexpected_destination_objects)
        if unexpected_keys != tuple(sorted(set(unexpected_keys))):
            raise ValueError("unexpected mirror destination keys must be sorted and unique")
        prefix = f"{RunArtifactTransportManifest.ROOT}/{self.archive_id}/"
        if any(
            not item.key.startswith(prefix)
            or item.key in object_keys
            or item.content_sha256 is None
            or item.etag is None
            or item.last_modified is None
            for item in self.unexpected_destination_objects
        ):
            raise ValueError("unexpected mirror destination identity is invalid")
        copied = tuple(item for item in self.objects if item.action is RunArtifactMirrorAction.COPY)
        reused = tuple(
            item for item in self.objects if item.action is RunArtifactMirrorAction.REUSE
        )
        if commit_item.action is RunArtifactMirrorAction.REUSE and any(
            item.action is RunArtifactMirrorAction.COPY for item in self.objects[:-1]
        ):
            raise ValueError("existing mirror commit requires the complete destination graph")
        if self.copy_object_count != len(copied) or self.reuse_object_count != len(reused):
            raise ValueError("mirror plan object counts do not match actions")
        if self.copy_bytes != sum(item.reference.size_bytes for item in copied):
            raise ValueError("mirror plan copy bytes do not match actions")
        if self.reuse_bytes != sum(item.reference.size_bytes for item in reused):
            raise ValueError("mirror plan reuse bytes do not match actions")
        expected_id = self.expected_plan_id(
            archive_id=self.archive_id,
            commit=self.commit,
            manifest=self.manifest,
            require_attestation=self.require_attestation,
            attestation_verification=self.attestation_verification,
            objects=self.objects,
            unexpected_destination_objects=self.unexpected_destination_objects,
        )
        if self.plan_id != expected_id:
            raise ValueError("mirror plan ID does not match its contents")
        return self

    @staticmethod
    def _same_reference(
        left: RunArtifactTransportObjectRef,
        right: RunArtifactTransportObjectRef,
    ) -> bool:
        return (
            left.key == right.key
            and left.media_type == right.media_type
            and left.size_bytes == right.size_bytes
            and left.content_digest == right.content_digest
        )

    @staticmethod
    def expected_plan_id(
        *,
        archive_id: str,
        commit: RunArtifactTransportCommit,
        manifest: RunArtifactTransportManifest,
        require_attestation: bool,
        attestation_verification: RunArtifactArchiveAttestationVerification | None,
        objects: tuple[RunArtifactMirrorObjectPlan, ...],
        unexpected_destination_objects: tuple[BlobInfo, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "archive_id": archive_id,
                "commit": commit.model_dump(mode="json"),
                "manifest": manifest.model_dump(mode="json"),
                "require_attestation": require_attestation,
                "attestation_verification": (
                    attestation_verification.model_dump(mode="json")
                    if attestation_verification is not None
                    else None
                ),
                "objects": [item.model_dump(mode="json") for item in objects],
                "unexpected_destination_objects": [
                    item.model_dump(mode="json") for item in unexpected_destination_objects
                ],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_mirror_plan_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactMirrorRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    mirror_id: str = Field(pattern=r"^run_mirror_[0-9a-f]{24}$")
    plan: RunArtifactMirrorPlan
    operator: str = Field(min_length=1)
    reason: str = Field(min_length=8)
    started_at: datetime
    completed_at: datetime
    created_keys: tuple[str, ...]
    reused_keys: tuple[str, ...]
    created_object_count: int = Field(ge=0)
    reused_object_count: int = Field(ge=0)
    created_bytes: int = Field(ge=0)
    reused_bytes: int = Field(ge=0)
    commit_created: bool

    @model_validator(mode="after")
    def validate_record(self) -> RunArtifactMirrorRecord:
        for value in (self.started_at, self.completed_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("mirror record times must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("mirror completion precedes start")
        plan_keys = tuple(item.reference.key for item in self.plan.objects)
        created_set = set(self.created_keys)
        reused_set = set(self.reused_keys)
        expected_created = tuple(key for key in plan_keys if key in created_set)
        expected_reused = tuple(key for key in plan_keys if key in reused_set)
        if self.created_keys != expected_created or self.reused_keys != expected_reused:
            raise ValueError("mirror record keys must follow plan order")
        if created_set & reused_set or created_set | reused_set != set(plan_keys):
            raise ValueError("mirror record keys do not partition its plan")
        if self.created_object_count != len(self.created_keys) or self.reused_object_count != len(
            self.reused_keys
        ):
            raise ValueError("mirror record object counts do not match its keys")
        sizes = {item.reference.key: item.reference.size_bytes for item in self.plan.objects}
        if self.created_bytes != sum(sizes[key] for key in self.created_keys):
            raise ValueError("mirror record created bytes do not match its keys")
        if self.reused_bytes != sum(sizes[key] for key in self.reused_keys):
            raise ValueError("mirror record reused bytes do not match its keys")
        if self.commit_created != (plan_keys[-1] in created_set):
            raise ValueError("mirror record commit result does not match its keys")
        expected_id = self.expected_mirror_id(
            plan_id=self.plan.plan_id,
            operator=self.operator,
            reason=self.reason,
        )
        if self.mirror_id != expected_id:
            raise ValueError("mirror record ID does not match its contents")
        return self

    @staticmethod
    def expected_mirror_id(*, plan_id: str, operator: str, reason: str) -> str:
        payload = orjson.dumps(
            {
                "plan_id": plan_id,
                "operator": operator,
                "reason": reason,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_mirror_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactMirrorSelection(str, Enum):
    EXPLICIT = "explicit"
    ALL_COMMITTED = "all_committed"


class RunArtifactMirrorBatchPolicy(ContractModel):
    max_release_count: int = Field(default=256, ge=1, le=100_000)
    max_copy_bytes: int = Field(default=1_099_511_627_776, ge=0)


class RunArtifactMirrorBatchPlan(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    plan_id: str = Field(pattern=r"^run_mirror_batch_plan_[0-9a-f]{24}$")
    source_inventory: RunArtifactStoreInventory
    destination_inventory: RunArtifactStoreInventory
    selection: RunArtifactMirrorSelection
    policy: RunArtifactMirrorBatchPolicy = Field(default_factory=RunArtifactMirrorBatchPolicy)
    require_attestation: bool
    selected_archive_ids: tuple[str, ...] = Field(min_length=1)
    releases: tuple[RunArtifactMirrorPlan, ...] = Field(min_length=1)
    release_count: int = Field(ge=1)
    object_count: int = Field(ge=1)
    copy_object_count: int = Field(ge=0)
    reuse_object_count: int = Field(ge=0)
    copy_bytes: int = Field(ge=0)
    reuse_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_plan(self) -> RunArtifactMirrorBatchPlan:
        observed_at = self.source_inventory.observed_at
        if self.destination_inventory.observed_at != observed_at or any(
            item.observed_at != observed_at for item in self.releases
        ):
            raise ValueError("mirror batch plan inventories must share one observation time")
        archive_ids = self.selected_archive_ids
        if archive_ids != tuple(sorted(set(archive_ids))):
            raise ValueError("mirror batch selected archive IDs must be sorted and unique")
        if tuple(item.archive_id for item in self.releases) != archive_ids:
            raise ValueError("mirror batch release order does not match its selection")
        source_by_id = {item.archive_id: item for item in self.source_inventory.releases}
        for release in self.releases:
            source_status = source_by_id.get(release.archive_id)
            if (
                source_status is None
                or source_status.state is not RunArtifactReleaseState.COMMITTED
                or source_status.commit_id != release.commit.commit_id
            ):
                raise ValueError("mirror batch release is not committed in its source inventory")
            if release.require_attestation != self.require_attestation:
                raise ValueError("mirror batch release trust policy is inconsistent")
        if self.selection is RunArtifactMirrorSelection.ALL_COMMITTED:
            committed_ids = tuple(
                item.archive_id
                for item in self.source_inventory.releases
                if item.state is RunArtifactReleaseState.COMMITTED
            )
            if archive_ids != committed_ids:
                raise ValueError("all-committed mirror selection is incomplete")
        if self.release_count != len(self.releases):
            raise ValueError("mirror batch release count does not match its plans")
        if self.release_count > self.policy.max_release_count:
            raise ValueError("mirror batch release count exceeds its safety policy")
        if self.object_count != sum(len(item.objects) for item in self.releases):
            raise ValueError("mirror batch object count does not match its plans")
        if self.copy_object_count != sum(item.copy_object_count for item in self.releases):
            raise ValueError("mirror batch copy count does not match its plans")
        if self.reuse_object_count != sum(item.reuse_object_count for item in self.releases):
            raise ValueError("mirror batch reuse count does not match its plans")
        if self.copy_bytes != sum(item.copy_bytes for item in self.releases):
            raise ValueError("mirror batch copy bytes do not match its plans")
        if self.copy_bytes > self.policy.max_copy_bytes:
            raise ValueError("mirror batch copy bytes exceed its safety policy")
        if self.reuse_bytes != sum(item.reuse_bytes for item in self.releases):
            raise ValueError("mirror batch reuse bytes do not match its plans")
        expected_id = self.expected_plan_id(
            source_state_digest=self.source_inventory.state_digest,
            destination_state_digest=self.destination_inventory.state_digest,
            selection=self.selection,
            policy=self.policy,
            require_attestation=self.require_attestation,
            selected_archive_ids=self.selected_archive_ids,
            releases=self.releases,
        )
        if self.plan_id != expected_id:
            raise ValueError("mirror batch plan ID does not match its contents")
        return self

    @staticmethod
    def expected_plan_id(
        *,
        source_state_digest: str,
        destination_state_digest: str,
        selection: RunArtifactMirrorSelection,
        policy: RunArtifactMirrorBatchPolicy,
        require_attestation: bool,
        selected_archive_ids: tuple[str, ...],
        releases: tuple[RunArtifactMirrorPlan, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "source_state_digest": source_state_digest,
                "destination_state_digest": destination_state_digest,
                "selection": selection.value,
                "policy": policy.model_dump(mode="json"),
                "require_attestation": require_attestation,
                "selected_archive_ids": selected_archive_ids,
                "release_plan_ids": [item.plan_id for item in releases],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_mirror_batch_plan_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactMirrorBatchIntent(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    batch_id: str = Field(pattern=r"^run_mirror_batch_[0-9a-f]{24}$")
    plan: RunArtifactMirrorBatchPlan
    operator: str = Field(min_length=1)
    reason: str = Field(min_length=8)
    created_at: datetime

    @model_validator(mode="after")
    def validate_intent(self) -> RunArtifactMirrorBatchIntent:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("mirror batch intent time must be timezone-aware")
        expected_id = self.expected_batch_id(
            plan_id=self.plan.plan_id,
            operator=self.operator,
            reason=self.reason,
        )
        if self.batch_id != expected_id:
            raise ValueError("mirror batch ID does not match its intent")
        return self

    @staticmethod
    def expected_batch_id(*, plan_id: str, operator: str, reason: str) -> str:
        payload = orjson.dumps(
            {
                "plan_id": plan_id,
                "operator": operator,
                "reason": reason,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_mirror_batch_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactMirrorBatchRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    intent: RunArtifactMirrorBatchIntent
    completed_at: datetime
    records: tuple[RunArtifactMirrorRecord, ...]
    release_count: int = Field(ge=1)
    created_object_count: int = Field(ge=0)
    reused_object_count: int = Field(ge=0)
    created_bytes: int = Field(ge=0)
    reused_bytes: int = Field(ge=0)
    commit_created_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_record(self) -> RunArtifactMirrorBatchRecord:
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("mirror batch completion time must be timezone-aware")
        if self.completed_at < self.intent.created_at:
            raise ValueError("mirror batch completion precedes its intent")
        plans = self.intent.plan.releases
        if self.release_count != len(plans) or len(self.records) != len(plans):
            raise ValueError("mirror batch records do not cover every release")
        for plan, record in zip(plans, self.records, strict=True):
            if (
                record.plan.canonical_bytes() != plan.canonical_bytes()
                or record.operator != self.intent.operator
                or record.reason != self.intent.reason
                or record.started_at < self.intent.created_at
                or record.completed_at > self.completed_at
            ):
                raise ValueError("mirror batch member evidence does not match its intent")
        if self.created_object_count != sum(item.created_object_count for item in self.records):
            raise ValueError("mirror batch created count does not match member evidence")
        if self.reused_object_count != sum(item.reused_object_count for item in self.records):
            raise ValueError("mirror batch reused count does not match member evidence")
        if self.created_bytes != sum(item.created_bytes for item in self.records):
            raise ValueError("mirror batch created bytes do not match member evidence")
        if self.reused_bytes != sum(item.reused_bytes for item in self.records):
            raise ValueError("mirror batch reused bytes do not match member evidence")
        if self.commit_created_count != sum(item.commit_created for item in self.records):
            raise ValueError("mirror batch commit count does not match member evidence")
        return self


class RunArtifactMirrorBatchMemberState(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    DESTINATION_COMPLETE = "destination_complete"
    COMPLETED = "completed"
    INVALID = "invalid"


class RunArtifactMirrorBatchMemberStatus(ContractModel):
    archive_id: str = Field(pattern=r"^run_archive_[0-9a-f]{24}$")
    mirror_id: str = Field(pattern=r"^run_mirror_[0-9a-f]{24}$")
    state: RunArtifactMirrorBatchMemberState
    planned_copy_object_count: int = Field(ge=0)
    planned_copy_bytes: int = Field(ge=0)
    present_copy_object_count: int | None = Field(default=None, ge=0)
    present_copy_bytes: int | None = Field(default=None, ge=0)
    remaining_copy_object_count: int | None = Field(default=None, ge=0)
    remaining_copy_bytes: int | None = Field(default=None, ge=0)
    record: RunArtifactMirrorRecord | None = None
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status(self) -> RunArtifactMirrorBatchMemberStatus:
        progress = (
            self.present_copy_object_count,
            self.present_copy_bytes,
            self.remaining_copy_object_count,
            self.remaining_copy_bytes,
        )
        if self.state is RunArtifactMirrorBatchMemberState.INVALID:
            if any(value is not None for value in progress) or self.record is not None:
                raise ValueError("invalid mirror member status cannot claim verified progress")
            return self
        if any(value is None for value in progress):
            raise ValueError("valid mirror member status requires complete progress counts")
        present_count = self.present_copy_object_count or 0
        present_bytes = self.present_copy_bytes or 0
        remaining_count = self.remaining_copy_object_count or 0
        remaining_bytes = self.remaining_copy_bytes or 0
        if present_count + remaining_count != self.planned_copy_object_count:
            raise ValueError("mirror member progress count does not match its plan")
        if present_bytes + remaining_bytes != self.planned_copy_bytes:
            raise ValueError("mirror member progress bytes do not match its plan")
        if self.state is RunArtifactMirrorBatchMemberState.PENDING and present_count != 0:
            raise ValueError("pending mirror member cannot contain copied progress")
        if self.state is RunArtifactMirrorBatchMemberState.PARTIAL and not (
            0 < present_count < self.planned_copy_object_count
        ):
            raise ValueError("partial mirror member requires incomplete copied progress")
        if (
            self.state
            in {
                RunArtifactMirrorBatchMemberState.DESTINATION_COMPLETE,
                RunArtifactMirrorBatchMemberState.COMPLETED,
            }
            and remaining_count != 0
        ):
            raise ValueError("complete mirror member cannot contain remaining objects")
        if self.state is RunArtifactMirrorBatchMemberState.COMPLETED:
            if self.record is None or self.record.mirror_id != self.mirror_id:
                raise ValueError("completed mirror member requires matching evidence")
        elif self.record is not None:
            raise ValueError("incomplete mirror member cannot contain completion evidence")
        return self


class RunArtifactMirrorBatchStatus(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    observed_at: datetime
    intent: RunArtifactMirrorBatchIntent
    members: tuple[RunArtifactMirrorBatchMemberStatus, ...]
    state_counts: dict[str, int]
    release_count: int = Field(ge=1)
    planned_copy_object_count: int = Field(ge=0)
    planned_copy_bytes: int = Field(ge=0)
    present_copy_object_count: int | None = Field(default=None, ge=0)
    present_copy_bytes: int | None = Field(default=None, ge=0)
    remaining_copy_object_count: int | None = Field(default=None, ge=0)
    remaining_copy_bytes: int | None = Field(default=None, ge=0)
    record: RunArtifactMirrorBatchRecord | None = None
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_status(self) -> RunArtifactMirrorBatchStatus:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("mirror batch status time must be timezone-aware")
        plans = self.intent.plan.releases
        if self.release_count != len(plans) or len(self.members) != len(plans):
            raise ValueError("mirror batch status does not cover every release")
        if tuple(item.archive_id for item in self.members) != tuple(
            item.archive_id for item in plans
        ):
            raise ValueError("mirror batch status order does not match its plan")
        for plan, member in zip(plans, self.members, strict=True):
            expected_mirror_id = RunArtifactMirrorRecord.expected_mirror_id(
                plan_id=plan.plan_id,
                operator=self.intent.operator,
                reason=self.intent.reason,
            )
            if (
                member.mirror_id != expected_mirror_id
                or member.planned_copy_object_count != plan.copy_object_count
                or member.planned_copy_bytes != plan.copy_bytes
            ):
                raise ValueError("mirror batch member status does not match its plan")
        expected_counts = {
            state.value: sum(item.state is state for item in self.members)
            for state in RunArtifactMirrorBatchMemberState
        }
        if self.state_counts != expected_counts:
            raise ValueError("mirror batch status counts do not match its members")
        if self.planned_copy_object_count != self.intent.plan.copy_object_count:
            raise ValueError("mirror batch planned copy count does not match its plan")
        if self.planned_copy_bytes != self.intent.plan.copy_bytes:
            raise ValueError("mirror batch planned bytes do not match its plan")
        invalid = any(
            item.state is RunArtifactMirrorBatchMemberState.INVALID for item in self.members
        )
        aggregate = (
            self.present_copy_object_count,
            self.present_copy_bytes,
            self.remaining_copy_object_count,
            self.remaining_copy_bytes,
        )
        if invalid:
            if any(value is not None for value in aggregate):
                raise ValueError("invalid mirror batch status cannot claim aggregate progress")
        else:
            if any(value is None for value in aggregate):
                raise ValueError("valid mirror batch status requires aggregate progress")
            if self.present_copy_object_count != sum(
                item.present_copy_object_count or 0 for item in self.members
            ):
                raise ValueError("mirror batch present count does not match its members")
            if self.present_copy_bytes != sum(
                item.present_copy_bytes or 0 for item in self.members
            ):
                raise ValueError("mirror batch present bytes do not match its members")
            if self.remaining_copy_object_count != sum(
                item.remaining_copy_object_count or 0 for item in self.members
            ):
                raise ValueError("mirror batch remaining count does not match its members")
            if self.remaining_copy_bytes != sum(
                item.remaining_copy_bytes or 0 for item in self.members
            ):
                raise ValueError("mirror batch remaining bytes do not match its members")
        if (
            self.record is not None
            and self.record.intent.canonical_bytes() != self.intent.canonical_bytes()
        ):
            raise ValueError("mirror batch completion record does not match its status")
        expected_digest = self.expected_state_digest(
            intent=self.intent,
            members=self.members,
            record=self.record,
        )
        if self.state_digest != expected_digest:
            raise ValueError("mirror batch status digest does not match its contents")
        return self

    @staticmethod
    def expected_state_digest(
        *,
        intent: RunArtifactMirrorBatchIntent,
        members: tuple[RunArtifactMirrorBatchMemberStatus, ...],
        record: RunArtifactMirrorBatchRecord | None,
    ) -> str:
        payload = orjson.dumps(
            {
                "batch_id": intent.batch_id,
                "members": [
                    {
                        **item.model_dump(mode="json", exclude={"detail", "record"}),
                        "record_digest": (
                            hashlib.sha256(item.record.canonical_bytes()).hexdigest()
                            if item.record is not None
                            else None
                        ),
                    }
                    for item in members
                ],
                "batch_record_digest": (
                    hashlib.sha256(record.canonical_bytes()).hexdigest()
                    if record is not None
                    else None
                ),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(payload).hexdigest()


class RunArtifactMirrorBatchResolutionKind(str, Enum):
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class RunArtifactMirrorBatchResolutionBasis(str, Enum):
    NOT_STARTED = "not_started"
    INVALID = "invalid"


class RunArtifactMirrorBatchResolution(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    resolution_id: str = Field(pattern=r"^run_mirror_batch_resolution_[0-9a-f]{24}$")
    batch_id: str = Field(pattern=r"^run_mirror_batch_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^run_mirror_batch_plan_[0-9a-f]{24}$")
    kind: RunArtifactMirrorBatchResolutionKind
    basis: RunArtifactMirrorBatchResolutionBasis
    confirmed_status_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmed_state_counts: dict[str, int]
    invalid_members: dict[str, str] = Field(default_factory=dict)
    release_count: int = Field(ge=1)
    resolver: str = Field(min_length=1)
    reason: str = Field(min_length=8)
    replacement_plan_id: str | None = Field(
        default=None,
        pattern=r"^run_mirror_batch_plan_[0-9a-f]{24}$",
    )
    created_at: datetime

    @model_validator(mode="after")
    def validate_resolution(self) -> RunArtifactMirrorBatchResolution:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("mirror batch resolution time must be timezone-aware")
        expected_counts = {
            state.value: self.confirmed_state_counts.get(state.value, 0)
            for state in RunArtifactMirrorBatchMemberState
        }
        if self.confirmed_state_counts != expected_counts or any(
            value < 0 for value in expected_counts.values()
        ):
            raise ValueError("mirror batch resolution state counts are invalid")
        if sum(expected_counts.values()) != self.release_count:
            raise ValueError("mirror batch resolution counts do not cover every release")
        invalid_count = expected_counts[RunArtifactMirrorBatchMemberState.INVALID.value]
        pending_count = expected_counts[RunArtifactMirrorBatchMemberState.PENDING.value]
        expected_basis = (
            RunArtifactMirrorBatchResolutionBasis.INVALID
            if invalid_count
            else RunArtifactMirrorBatchResolutionBasis.NOT_STARTED
        )
        if self.basis is not expected_basis or (
            not invalid_count and pending_count != self.release_count
        ):
            raise ValueError("mirror batch resolution is not based on an eligible status")
        if len(self.invalid_members) != invalid_count or any(
            not archive_id.startswith("run_archive_")
            or len(archive_id) != 36
            or any(character not in "0123456789abcdef" for character in archive_id[12:])
            or not detail.strip()
            for archive_id, detail in self.invalid_members.items()
        ):
            raise ValueError("mirror batch resolution invalid-member evidence is inconsistent")
        if self.kind is RunArtifactMirrorBatchResolutionKind.SUPERSEDED:
            if self.replacement_plan_id is None or self.replacement_plan_id == self.plan_id:
                raise ValueError("superseded mirror batch requires a different replacement plan")
        elif self.replacement_plan_id is not None:
            raise ValueError("cancelled mirror batch cannot name a replacement plan")
        expected_id = self.expected_resolution_id(
            batch_id=self.batch_id,
            plan_id=self.plan_id,
            kind=self.kind,
            basis=self.basis,
            confirmed_status_digest=self.confirmed_status_digest,
            invalid_members=self.invalid_members,
            resolver=self.resolver,
            reason=self.reason,
            replacement_plan_id=self.replacement_plan_id,
        )
        if self.resolution_id != expected_id:
            raise ValueError("mirror batch resolution ID does not match its contents")
        return self

    @staticmethod
    def expected_resolution_id(
        *,
        batch_id: str,
        plan_id: str,
        kind: RunArtifactMirrorBatchResolutionKind,
        basis: RunArtifactMirrorBatchResolutionBasis,
        confirmed_status_digest: str,
        invalid_members: dict[str, str],
        resolver: str,
        reason: str,
        replacement_plan_id: str | None,
    ) -> str:
        payload = orjson.dumps(
            {
                "batch_id": batch_id,
                "plan_id": plan_id,
                "kind": kind.value,
                "basis": basis.value,
                "confirmed_status_digest": confirmed_status_digest,
                "invalid_members": invalid_members,
                "resolver": resolver,
                "reason": reason,
                "replacement_plan_id": replacement_plan_id,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_mirror_batch_resolution_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactMirrorBatchDecisionKind(str, Enum):
    COMPLETED = "completed"
    RESOLVED = "resolved"


class RunArtifactMirrorBatchDecision(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    decision_id: str = Field(pattern=r"^run_mirror_batch_decision_[0-9a-f]{24}$")
    batch_id: str = Field(pattern=r"^run_mirror_batch_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^run_mirror_batch_plan_[0-9a-f]{24}$")
    kind: RunArtifactMirrorBatchDecisionKind
    completed: RunArtifactMirrorBatchRecord | None = None
    resolved: RunArtifactMirrorBatchResolution | None = None
    decided_at: datetime

    @model_validator(mode="after")
    def validate_decision(self) -> RunArtifactMirrorBatchDecision:
        if self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None:
            raise ValueError("mirror batch decision time must be timezone-aware")
        if self.kind is RunArtifactMirrorBatchDecisionKind.COMPLETED:
            if self.completed is None or self.resolved is not None:
                raise ValueError("completed mirror decision requires only batch evidence")
            if (
                self.completed.intent.batch_id != self.batch_id
                or self.completed.intent.plan.plan_id != self.plan_id
                or self.completed.completed_at != self.decided_at
            ):
                raise ValueError("completed mirror decision evidence is inconsistent")
            outcome = self.completed.canonical_bytes()
        else:
            if self.resolved is None or self.completed is not None:
                raise ValueError("resolved mirror decision requires only resolution evidence")
            if (
                self.resolved.batch_id != self.batch_id
                or self.resolved.plan_id != self.plan_id
                or self.resolved.created_at != self.decided_at
            ):
                raise ValueError("resolved mirror decision evidence is inconsistent")
            outcome = self.resolved.canonical_bytes()
        expected_id = self.expected_decision_id(
            batch_id=self.batch_id,
            plan_id=self.plan_id,
            kind=self.kind,
            outcome=outcome,
        )
        if self.decision_id != expected_id:
            raise ValueError("mirror batch decision ID does not match its contents")
        return self

    @staticmethod
    def expected_decision_id(
        *,
        batch_id: str,
        plan_id: str,
        kind: RunArtifactMirrorBatchDecisionKind,
        outcome: bytes,
    ) -> str:
        payload = orjson.dumps(
            {
                "batch_id": batch_id,
                "plan_id": plan_id,
                "kind": kind.value,
                "outcome_digest": hashlib.sha256(outcome).hexdigest(),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_mirror_batch_decision_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactMirrorBatchHealth(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    DEGRADED = "degraded"
    RESOLVED = "resolved"


class RunArtifactMirrorBatchInspection(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    status: RunArtifactMirrorBatchStatus
    resolution: RunArtifactMirrorBatchResolution | None = None
    health: RunArtifactMirrorBatchHealth
    resolution_allowed: bool
    resolution_basis: RunArtifactMirrorBatchResolutionBasis | None = None
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inspection(self) -> RunArtifactMirrorBatchInspection:
        if self.resolution is not None and (
            self.resolution.batch_id != self.status.intent.batch_id
            or self.resolution.plan_id != self.status.intent.plan.plan_id
        ):
            raise ValueError("mirror batch resolution belongs to another inspection")
        counts = self.status.state_counts
        invalid = counts[RunArtifactMirrorBatchMemberState.INVALID.value] > 0
        pending = counts[RunArtifactMirrorBatchMemberState.PENDING.value]
        active = any(
            counts[state.value] > 0
            for state in (
                RunArtifactMirrorBatchMemberState.PARTIAL,
                RunArtifactMirrorBatchMemberState.DESTINATION_COMPLETE,
                RunArtifactMirrorBatchMemberState.COMPLETED,
            )
        )
        if self.resolution is not None:
            expected_health = RunArtifactMirrorBatchHealth.RESOLVED
            allowed = False
            basis = None
        elif self.status.record is not None:
            expected_health = (
                RunArtifactMirrorBatchHealth.DEGRADED
                if invalid
                else RunArtifactMirrorBatchHealth.COMPLETE
            )
            allowed = False
            basis = None
        elif invalid:
            expected_health = RunArtifactMirrorBatchHealth.BLOCKED
            allowed = True
            basis = RunArtifactMirrorBatchResolutionBasis.INVALID
        elif active:
            expected_health = RunArtifactMirrorBatchHealth.ACTIVE
            allowed = False
            basis = None
        else:
            expected_health = RunArtifactMirrorBatchHealth.PENDING
            allowed = pending == self.status.release_count
            basis = RunArtifactMirrorBatchResolutionBasis.NOT_STARTED if allowed else None
        if (
            self.health is not expected_health
            or self.resolution_allowed != allowed
            or self.resolution_basis is not basis
        ):
            raise ValueError("mirror batch inspection health is inconsistent")
        expected_digest = self.expected_state_digest(
            status_digest=self.status.state_digest,
            resolution=self.resolution,
        )
        if self.state_digest != expected_digest:
            raise ValueError("mirror batch inspection digest does not match its contents")
        return self

    @staticmethod
    def expected_state_digest(
        *,
        status_digest: str,
        resolution: RunArtifactMirrorBatchResolution | None,
    ) -> str:
        payload = orjson.dumps(
            {
                "status_digest": status_digest,
                "resolution_digest": (
                    hashlib.sha256(resolution.canonical_bytes()).hexdigest()
                    if resolution is not None
                    else None
                ),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(payload).hexdigest()


class RunArtifactMirrorBatchEvidenceState(str, Enum):
    INTENT_ONLY = "intent_only"
    MEMBER_EVIDENCE = "member_evidence"
    COMPLETE = "complete"
    RESOLVED = "resolved"


class RunArtifactMirrorBatchLedgerEntry(ContractModel):
    batch_id: str = Field(pattern=r"^run_mirror_batch_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^run_mirror_batch_plan_[0-9a-f]{24}$")
    intent_created_at: datetime
    selection: RunArtifactMirrorSelection
    release_count: int = Field(ge=1)
    copy_bytes: int = Field(ge=0)
    member_record_count: int = Field(ge=0)
    batch_record_present: bool
    resolution_id: str | None = Field(
        default=None,
        pattern=r"^run_mirror_batch_resolution_[0-9a-f]{24}$",
    )
    resolution_kind: RunArtifactMirrorBatchResolutionKind | None = None
    replacement_plan_id: str | None = Field(
        default=None,
        pattern=r"^run_mirror_batch_plan_[0-9a-f]{24}$",
    )
    state: RunArtifactMirrorBatchEvidenceState

    @model_validator(mode="after")
    def validate_entry(self) -> RunArtifactMirrorBatchLedgerEntry:
        if self.intent_created_at.tzinfo is None or self.intent_created_at.utcoffset() is None:
            raise ValueError("mirror ledger intent time must be timezone-aware")
        if self.member_record_count > self.release_count:
            raise ValueError("mirror ledger member count exceeds its release count")
        resolution_present = self.resolution_id is not None
        if resolution_present != (self.resolution_kind is not None):
            raise ValueError("mirror ledger resolution fields are incomplete")
        if self.resolution_kind is RunArtifactMirrorBatchResolutionKind.SUPERSEDED:
            if self.replacement_plan_id is None:
                raise ValueError("superseded mirror ledger entry requires a replacement plan")
        elif self.replacement_plan_id is not None:
            raise ValueError("non-superseded mirror ledger entry has a replacement plan")
        if self.batch_record_present and resolution_present:
            raise ValueError("mirror batch cannot be both completed and resolved")
        if self.batch_record_present and self.member_record_count != self.release_count:
            raise ValueError("completed mirror ledger entry requires every member record")
        expected_state = (
            RunArtifactMirrorBatchEvidenceState.RESOLVED
            if resolution_present
            else (
                RunArtifactMirrorBatchEvidenceState.COMPLETE
                if self.batch_record_present
                else (
                    RunArtifactMirrorBatchEvidenceState.MEMBER_EVIDENCE
                    if self.member_record_count
                    else RunArtifactMirrorBatchEvidenceState.INTENT_ONLY
                )
            )
        )
        if self.state is not expected_state:
            raise ValueError("mirror ledger evidence state is inconsistent")
        return self


class RunArtifactMirrorOperationsLedger(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    observed_at: datetime
    entries: tuple[RunArtifactMirrorBatchLedgerEntry, ...] = ()
    unclassified_keys: tuple[str, ...] = ()
    batch_count: int = Field(ge=0)
    state_counts: dict[str, int]
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_ledger(self) -> RunArtifactMirrorOperationsLedger:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("mirror operations ledger time must be timezone-aware")
        batch_ids = tuple(item.batch_id for item in self.entries)
        if batch_ids != tuple(sorted(set(batch_ids))):
            raise ValueError("mirror ledger batch IDs must be sorted and unique")
        if self.unclassified_keys != tuple(sorted(set(self.unclassified_keys))):
            raise ValueError("mirror ledger unclassified keys must be sorted and unique")
        if self.batch_count != len(self.entries):
            raise ValueError("mirror ledger batch count does not match its entries")
        expected_counts = {
            state.value: sum(item.state is state for item in self.entries)
            for state in RunArtifactMirrorBatchEvidenceState
        }
        if self.state_counts != expected_counts:
            raise ValueError("mirror ledger state counts do not match its entries")
        expected_digest = self.expected_state_digest(
            entries=self.entries,
            unclassified_keys=self.unclassified_keys,
        )
        if self.state_digest != expected_digest:
            raise ValueError("mirror ledger digest does not match its contents")
        return self

    @staticmethod
    def expected_state_digest(
        *,
        entries: tuple[RunArtifactMirrorBatchLedgerEntry, ...],
        unclassified_keys: tuple[str, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "entries": [item.model_dump(mode="json") for item in entries],
                "unclassified_keys": unclassified_keys,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(payload).hexdigest()

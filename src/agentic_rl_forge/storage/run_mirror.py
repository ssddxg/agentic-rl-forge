from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Literal, TypeAlias

from agentic_rl_forge.contracts import (
    BlobInfo,
    RunArtifactMirrorAction,
    RunArtifactMirrorBatchDecision,
    RunArtifactMirrorBatchDecisionKind,
    RunArtifactMirrorBatchEvidenceState,
    RunArtifactMirrorBatchHealth,
    RunArtifactMirrorBatchInspection,
    RunArtifactMirrorBatchIntent,
    RunArtifactMirrorBatchLedgerEntry,
    RunArtifactMirrorBatchMemberState,
    RunArtifactMirrorBatchMemberStatus,
    RunArtifactMirrorBatchPlan,
    RunArtifactMirrorBatchPolicy,
    RunArtifactMirrorBatchRecord,
    RunArtifactMirrorBatchResolution,
    RunArtifactMirrorBatchResolutionBasis,
    RunArtifactMirrorBatchResolutionKind,
    RunArtifactMirrorBatchStatus,
    RunArtifactMirrorObjectPlan,
    RunArtifactMirrorOperationsLedger,
    RunArtifactMirrorPlan,
    RunArtifactMirrorRecord,
    RunArtifactMirrorSelection,
    RunArtifactReleaseState,
    RunArtifactTransportCommit,
    RunArtifactTransportManifest,
    RunArtifactTransportObjectRef,
)
from agentic_rl_forge.storage.blobs import BlobConflictError, ConditionalBlobStore
from agentic_rl_forge.storage.run_transport import RunArtifactTransport

_MirrorRole: TypeAlias = Literal["chunk", "checksum", "attestation", "manifest", "commit"]


class RunArtifactMirrorError(ValueError):
    pass


class RunArtifactMirror:
    MAX_WORKERS = 64
    DEFAULT_MAX_BATCH_RELEASES = 256
    DEFAULT_MAX_BATCH_COPY_BYTES = 1_099_511_627_776
    RECORD_ROOT = "run-release-mirrors/records"
    BATCH_ROOT = "run-release-mirrors/batches"

    def __init__(
        self,
        source: ConditionalBlobStore,
        destination: ConditionalBlobStore,
    ) -> None:
        self.source = source
        self.destination = destination
        self.source_transport = RunArtifactTransport(source)
        self.destination_transport = RunArtifactTransport(destination)

    def plan(
        self,
        archive_id: str,
        *,
        trusted_public_keys: Collection[str] = (),
        require_attestation: bool = False,
        now: datetime | None = None,
    ) -> RunArtifactMirrorPlan:
        observed_at = self._validated_time(now)
        trusted_keys = tuple(trusted_public_keys)
        commit, manifest, _, verification = self.source_transport.inspect_committed_release(
            archive_id,
            trusted_public_keys=trusted_keys,
            require_attestation=require_attestation,
        )
        roles_and_references = self._release_references(commit, manifest)
        object_plans = []
        expected_keys = {reference.key for _, reference in roles_and_references}
        for role, reference in roles_and_references:
            source_info = self._verified_identity(self.source, reference)
            destination_info = self.destination.head(reference.key)
            if destination_info is None:
                action = RunArtifactMirrorAction.COPY
                complete_destination = None
            else:
                try:
                    complete_destination = self._verified_identity(
                        self.destination,
                        reference,
                        initial=destination_info,
                    )
                except BlobConflictError as error:
                    raise BlobConflictError(
                        f"destination object {reference.key!r} conflicts with source release"
                    ) from error
                action = RunArtifactMirrorAction.REUSE
            object_plans.append(
                RunArtifactMirrorObjectPlan(
                    role=role,
                    reference=reference,
                    source=source_info,
                    destination=complete_destination,
                    action=action,
                )
            )
        unexpected = tuple(
            sorted(
                (
                    self._verified_blob_identity(self.destination, key)
                    for key in self.destination.list(
                        f"{RunArtifactTransport.release_root(archive_id)}/"
                    )
                    if key not in expected_keys
                ),
                key=lambda item: item.key,
            )
        )
        object_tuple = tuple(object_plans)
        plan_id = RunArtifactMirrorPlan.expected_plan_id(
            archive_id=archive_id,
            commit=commit,
            manifest=manifest,
            require_attestation=require_attestation,
            attestation_verification=verification,
            objects=object_tuple,
            unexpected_destination_objects=unexpected,
        )
        plan = RunArtifactMirrorPlan(
            plan_id=plan_id,
            observed_at=observed_at,
            archive_id=archive_id,
            commit=commit,
            manifest=manifest,
            require_attestation=require_attestation,
            attestation_verification=verification,
            objects=object_tuple,
            unexpected_destination_objects=unexpected,
            copy_object_count=sum(
                item.action is RunArtifactMirrorAction.COPY for item in object_tuple
            ),
            reuse_object_count=sum(
                item.action is RunArtifactMirrorAction.REUSE for item in object_tuple
            ),
            copy_bytes=sum(
                item.reference.size_bytes
                for item in object_tuple
                if item.action is RunArtifactMirrorAction.COPY
            ),
            reuse_bytes=sum(
                item.reference.size_bytes
                for item in object_tuple
                if item.action is RunArtifactMirrorAction.REUSE
            ),
        )
        self._validate_source(plan, trusted_keys)
        self._validate_destination_snapshot(plan, allow_progress=False)
        return plan

    def execute(
        self,
        plan: RunArtifactMirrorPlan,
        *,
        confirm_plan_id: str,
        operator: str,
        reason: str,
        trusted_public_keys: Collection[str] = (),
        max_workers: int = 1,
        now: datetime | None = None,
    ) -> RunArtifactMirrorRecord:
        if confirm_plan_id.strip() != plan.plan_id:
            raise RunArtifactMirrorError("mirror confirmation does not match the plan ID")
        normalized_operator = operator.strip()
        normalized_reason = reason.strip()
        if not normalized_operator:
            raise ValueError("mirror operator cannot be empty")
        if len(normalized_reason) < 8:
            raise ValueError("mirror reason must contain at least eight characters")
        if max_workers < 1 or max_workers > self.MAX_WORKERS:
            raise ValueError("mirror worker count must be between 1 and 64")
        action_time = self._validated_time(now)
        trusted_keys = tuple(trusted_public_keys)
        mirror_id = RunArtifactMirrorRecord.expected_mirror_id(
            plan_id=plan.plan_id,
            operator=normalized_operator,
            reason=normalized_reason,
        )
        existing = self._try_load_record(mirror_id)
        if existing is not None:
            if (
                existing.plan.canonical_bytes() != plan.canonical_bytes()
                or existing.operator != normalized_operator
                or existing.reason != normalized_reason
            ):
                raise RunArtifactMirrorError("stored mirror evidence differs from this command")
            self._validate_source(plan, trusted_keys)
            self._validate_destination_complete(plan, trusted_keys)
            return existing

        self._validate_source(plan, trusted_keys)
        self._validate_destination_snapshot(plan, allow_progress=True)
        created_by_key: dict[str, bool] = {}
        non_commit = plan.objects[:-1]
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="run-artifact-mirror",
        ) as executor:
            futures = {
                item.reference.key: executor.submit(self._copy_object, item) for item in non_commit
            }
            for item in non_commit:
                created_by_key[item.reference.key] = futures[item.reference.key].result()

        self._validate_destination_snapshot(plan, allow_progress=True)
        commit_item = plan.objects[-1]
        commit_created = self._copy_object(commit_item)
        created_by_key[commit_item.reference.key] = commit_created
        self._validate_destination_complete(plan, trusted_keys)

        created = tuple(item for item in plan.objects if created_by_key[item.reference.key])
        reused = tuple(item for item in plan.objects if not created_by_key[item.reference.key])
        record = RunArtifactMirrorRecord(
            mirror_id=mirror_id,
            plan=plan,
            operator=normalized_operator,
            reason=normalized_reason,
            started_at=action_time,
            completed_at=action_time,
            created_keys=tuple(item.reference.key for item in created),
            reused_keys=tuple(item.reference.key for item in reused),
            created_object_count=len(created),
            reused_object_count=len(reused),
            created_bytes=sum(item.reference.size_bytes for item in created),
            reused_bytes=sum(item.reference.size_bytes for item in reused),
            commit_created=commit_created,
        )
        payload = record.canonical_bytes() + b"\n"
        try:
            self._put_verified(
                self.destination,
                self.record_key(mirror_id),
                payload,
                metadata={
                    "kind": "run-archive-mirror-record",
                    "mirror-id": mirror_id,
                    "plan-id": plan.plan_id,
                    "archive-id": plan.archive_id,
                },
            )
        except BlobConflictError:
            loaded = self._try_load_record(mirror_id)
            if loaded is None:
                raise
            return loaded
        loaded = self._try_load_record(mirror_id)
        if loaded is None:
            raise RunArtifactMirrorError("mirror evidence disappeared after publication")
        return loaded

    def plan_batch(
        self,
        *,
        archive_ids: Collection[str] = (),
        include_all_committed: bool = False,
        trusted_public_keys: Collection[str] = (),
        require_attestation: bool = False,
        max_release_count: int = DEFAULT_MAX_BATCH_RELEASES,
        max_copy_bytes: int = DEFAULT_MAX_BATCH_COPY_BYTES,
        max_workers: int = 1,
        now: datetime | None = None,
    ) -> RunArtifactMirrorBatchPlan:
        if max_workers < 1 or max_workers > self.MAX_WORKERS:
            raise ValueError("mirror batch planning worker count must be between 1 and 64")
        requested = tuple(archive_ids)
        if include_all_committed == bool(requested):
            raise ValueError("select releases with either archive IDs or include_all_committed")
        if len(requested) != len(set(requested)):
            raise ValueError("mirror batch archive IDs must be unique")
        policy = RunArtifactMirrorBatchPolicy(
            max_release_count=max_release_count,
            max_copy_bytes=max_copy_bytes,
        )
        observed_at = self._validated_time(now)
        source_inventory = self.source_transport.inventory(now=observed_at)
        destination_inventory = self.destination_transport.inventory(now=observed_at)
        source_by_id = {item.archive_id: item for item in source_inventory.releases}
        if include_all_committed:
            selection = RunArtifactMirrorSelection.ALL_COMMITTED
            selected = tuple(
                item.archive_id
                for item in source_inventory.releases
                if item.state is RunArtifactReleaseState.COMMITTED
            )
        else:
            selection = RunArtifactMirrorSelection.EXPLICIT
            selected = tuple(sorted(requested))
        if not selected:
            raise RunArtifactMirrorError("mirror batch selection contains no releases")
        if len(selected) > policy.max_release_count:
            raise RunArtifactMirrorError("mirror batch release count exceeds its safety policy")
        for archive_id in selected:
            status = source_by_id.get(archive_id)
            if status is None:
                raise RunArtifactMirrorError(
                    f"selected source release {archive_id!r} was not found"
                )
            if status.state is not RunArtifactReleaseState.COMMITTED:
                raise RunArtifactMirrorError(
                    f"selected source release {archive_id!r} is not committed"
                )
        trusted_keys = tuple(trusted_public_keys)
        plans_by_id: dict[str, RunArtifactMirrorPlan] = {}
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="run-artifact-mirror-plan",
        ) as executor:
            futures = {
                archive_id: executor.submit(
                    self.plan,
                    archive_id,
                    trusted_public_keys=trusted_keys,
                    require_attestation=require_attestation,
                    now=observed_at,
                )
                for archive_id in selected
            }
            for archive_id in selected:
                plans_by_id[archive_id] = futures[archive_id].result()
        releases = tuple(plans_by_id[archive_id] for archive_id in selected)
        copy_bytes = sum(item.copy_bytes for item in releases)
        if copy_bytes > policy.max_copy_bytes:
            raise RunArtifactMirrorError("mirror batch copy bytes exceed its safety policy")
        final_source = self.source_transport.inventory(now=observed_at)
        final_destination = self.destination_transport.inventory(now=observed_at)
        if final_source.state_digest != source_inventory.state_digest:
            raise RunArtifactMirrorError("source inventory changed during mirror batch planning")
        if final_destination.state_digest != destination_inventory.state_digest:
            raise RunArtifactMirrorError(
                "destination inventory changed during mirror batch planning"
            )
        plan_id = RunArtifactMirrorBatchPlan.expected_plan_id(
            source_state_digest=source_inventory.state_digest,
            destination_state_digest=destination_inventory.state_digest,
            selection=selection,
            policy=policy,
            require_attestation=require_attestation,
            selected_archive_ids=selected,
            releases=releases,
        )
        return RunArtifactMirrorBatchPlan(
            plan_id=plan_id,
            source_inventory=source_inventory,
            destination_inventory=destination_inventory,
            selection=selection,
            policy=policy,
            require_attestation=require_attestation,
            selected_archive_ids=selected,
            releases=releases,
            release_count=len(releases),
            object_count=sum(len(item.objects) for item in releases),
            copy_object_count=sum(item.copy_object_count for item in releases),
            reuse_object_count=sum(item.reuse_object_count for item in releases),
            copy_bytes=copy_bytes,
            reuse_bytes=sum(item.reuse_bytes for item in releases),
        )

    def execute_batch(
        self,
        plan: RunArtifactMirrorBatchPlan,
        *,
        confirm_plan_id: str,
        operator: str,
        reason: str,
        trusted_public_keys: Collection[str] = (),
        release_workers: int = 1,
        object_workers: int = 1,
        now: datetime | None = None,
    ) -> RunArtifactMirrorBatchRecord:
        if confirm_plan_id.strip() != plan.plan_id:
            raise RunArtifactMirrorError("mirror batch confirmation does not match the plan ID")
        normalized_operator = operator.strip()
        normalized_reason = reason.strip()
        if not normalized_operator:
            raise ValueError("mirror batch operator cannot be empty")
        if len(normalized_reason) < 8:
            raise ValueError("mirror batch reason must contain at least eight characters")
        self._validate_batch_workers(release_workers, object_workers)
        action_time = self._validated_time(now)
        batch_id = RunArtifactMirrorBatchIntent.expected_batch_id(
            plan_id=plan.plan_id,
            operator=normalized_operator,
            reason=normalized_reason,
        )
        expected_intent = RunArtifactMirrorBatchIntent(
            batch_id=batch_id,
            plan=plan,
            operator=normalized_operator,
            reason=normalized_reason,
            created_at=action_time,
        )
        intent = self._try_load_batch_intent(batch_id)
        if intent is None:
            try:
                self._put_verified(
                    self.destination,
                    self.batch_intent_key(batch_id),
                    expected_intent.canonical_bytes() + b"\n",
                    metadata={
                        "kind": "run-archive-mirror-batch-intent",
                        "batch-id": batch_id,
                        "plan-id": plan.plan_id,
                    },
                )
                intent = expected_intent
            except BlobConflictError:
                intent = self._try_load_batch_intent(batch_id)
                if intent is None:
                    raise
        self._validate_batch_intent(
            intent,
            plan=plan,
            operator=normalized_operator,
            reason=normalized_reason,
        )
        decision, existing, resolution = self._load_batch_terminal_evidence(intent)
        if resolution is not None:
            raise RunArtifactMirrorError("mirror batch intent has been resolved")
        trusted_keys = tuple(trusted_public_keys)
        records = self._execute_batch_members(
            intent,
            trusted_public_keys=trusted_keys,
            release_workers=release_workers,
            object_workers=object_workers,
            action_time=action_time,
        )
        if existing is not None:
            if tuple(item.canonical_bytes() for item in existing.records) != tuple(
                item.canonical_bytes() for item in records
            ):
                raise RunArtifactMirrorError(
                    "stored mirror batch evidence differs from member evidence"
                )
            expected_decision = self._completion_decision(existing)
            if decision is None:
                decision = self._publish_batch_decision(expected_decision)
            if decision.canonical_bytes() != expected_decision.canonical_bytes():
                raise RunArtifactMirrorError(
                    "mirror batch terminal decision differs from completion evidence"
                )
            return self._ensure_batch_record_sidecar(existing)
        completed_at = max((item.completed_at for item in records), default=intent.created_at)
        completed_at = max(intent.created_at, completed_at)
        record = RunArtifactMirrorBatchRecord(
            intent=intent,
            completed_at=completed_at,
            records=records,
            release_count=len(records),
            created_object_count=sum(item.created_object_count for item in records),
            reused_object_count=sum(item.reused_object_count for item in records),
            created_bytes=sum(item.created_bytes for item in records),
            reused_bytes=sum(item.reused_bytes for item in records),
            commit_created_count=sum(item.commit_created for item in records),
        )
        expected_decision = self._completion_decision(record)
        decision = self._publish_batch_decision(expected_decision)
        if decision.kind is RunArtifactMirrorBatchDecisionKind.RESOLVED:
            raise RunArtifactMirrorError(
                "mirror batch was resolved while member execution was in progress"
            )
        if decision.canonical_bytes() != expected_decision.canonical_bytes():
            raise RunArtifactMirrorError("stored mirror batch completion differs from this command")
        return self._ensure_batch_record_sidecar(record)

    def batch_status(
        self,
        plan: RunArtifactMirrorBatchPlan,
        *,
        operator: str,
        reason: str,
        trusted_public_keys: Collection[str] = (),
        now: datetime | None = None,
    ) -> RunArtifactMirrorBatchStatus:
        normalized_operator = operator.strip()
        normalized_reason = reason.strip()
        if not normalized_operator:
            raise ValueError("mirror batch operator cannot be empty")
        if len(normalized_reason) < 8:
            raise ValueError("mirror batch reason must contain at least eight characters")
        observed_at = self._validated_time(now)
        batch_id = RunArtifactMirrorBatchIntent.expected_batch_id(
            plan_id=plan.plan_id,
            operator=normalized_operator,
            reason=normalized_reason,
        )
        intent = self._try_load_batch_intent(batch_id)
        if intent is None:
            raise RunArtifactMirrorError("mirror batch intent was not found")
        self._validate_batch_intent(
            intent,
            plan=plan,
            operator=normalized_operator,
            reason=normalized_reason,
        )
        trusted_keys = tuple(trusted_public_keys)
        members = tuple(
            self._batch_member_status(
                member_plan,
                operator=intent.operator,
                reason=intent.reason,
                trusted_public_keys=trusted_keys,
            )
            for member_plan in plan.releases
        )
        _, record, _ = self._load_batch_terminal_evidence(intent)
        state_counts = {
            state.value: sum(item.state is state for item in members)
            for state in RunArtifactMirrorBatchMemberState
        }
        invalid = state_counts[RunArtifactMirrorBatchMemberState.INVALID.value] > 0
        present_object_count = (
            None if invalid else sum(item.present_copy_object_count or 0 for item in members)
        )
        present_bytes = None if invalid else sum(item.present_copy_bytes or 0 for item in members)
        remaining_object_count = (
            None if invalid else sum(item.remaining_copy_object_count or 0 for item in members)
        )
        remaining_bytes = (
            None if invalid else sum(item.remaining_copy_bytes or 0 for item in members)
        )
        state_digest = RunArtifactMirrorBatchStatus.expected_state_digest(
            intent=intent,
            members=members,
            record=record,
        )
        return RunArtifactMirrorBatchStatus(
            observed_at=observed_at,
            intent=intent,
            members=members,
            state_counts=state_counts,
            release_count=len(members),
            planned_copy_object_count=plan.copy_object_count,
            planned_copy_bytes=plan.copy_bytes,
            present_copy_object_count=present_object_count,
            present_copy_bytes=present_bytes,
            remaining_copy_object_count=remaining_object_count,
            remaining_copy_bytes=remaining_bytes,
            record=record,
            state_digest=state_digest,
        )

    def inspect_batch(
        self,
        batch_id: str,
        *,
        trusted_public_keys: Collection[str] = (),
        now: datetime | None = None,
    ) -> RunArtifactMirrorBatchInspection:
        self._validate_batch_id(batch_id)
        intent = self._try_load_batch_intent(batch_id)
        if intent is None:
            raise RunArtifactMirrorError("mirror batch intent was not found")
        status = self.batch_status(
            intent.plan,
            operator=intent.operator,
            reason=intent.reason,
            trusted_public_keys=trusted_public_keys,
            now=now,
        )
        _, _, resolution = self._load_batch_terminal_evidence(intent)
        counts = status.state_counts
        invalid = counts[RunArtifactMirrorBatchMemberState.INVALID.value] > 0
        active = any(
            counts[state.value] > 0
            for state in (
                RunArtifactMirrorBatchMemberState.PARTIAL,
                RunArtifactMirrorBatchMemberState.DESTINATION_COMPLETE,
                RunArtifactMirrorBatchMemberState.COMPLETED,
            )
        )
        if resolution is not None:
            health = RunArtifactMirrorBatchHealth.RESOLVED
            resolution_allowed = False
            resolution_basis = None
        elif status.record is not None:
            health = (
                RunArtifactMirrorBatchHealth.DEGRADED
                if invalid
                else RunArtifactMirrorBatchHealth.COMPLETE
            )
            resolution_allowed = False
            resolution_basis = None
        elif invalid:
            health = RunArtifactMirrorBatchHealth.BLOCKED
            resolution_allowed = True
            resolution_basis = RunArtifactMirrorBatchResolutionBasis.INVALID
        elif active:
            health = RunArtifactMirrorBatchHealth.ACTIVE
            resolution_allowed = False
            resolution_basis = None
        else:
            health = RunArtifactMirrorBatchHealth.PENDING
            resolution_allowed = (
                counts[RunArtifactMirrorBatchMemberState.PENDING.value] == status.release_count
            )
            resolution_basis = (
                RunArtifactMirrorBatchResolutionBasis.NOT_STARTED if resolution_allowed else None
            )
        state_digest = RunArtifactMirrorBatchInspection.expected_state_digest(
            status_digest=status.state_digest,
            resolution=resolution,
        )
        return RunArtifactMirrorBatchInspection(
            status=status,
            resolution=resolution,
            health=health,
            resolution_allowed=resolution_allowed,
            resolution_basis=resolution_basis,
            state_digest=state_digest,
        )

    def resolve_batch(
        self,
        batch_id: str,
        *,
        kind: RunArtifactMirrorBatchResolutionKind,
        confirm_status_digest: str,
        resolver: str,
        reason: str,
        replacement_plan_id: str | None = None,
        trusted_public_keys: Collection[str] = (),
        now: datetime | None = None,
    ) -> RunArtifactMirrorBatchResolution:
        self._validate_batch_id(batch_id)
        confirmed = confirm_status_digest.strip().lower()
        if len(confirmed) != 64 or any(
            character not in "0123456789abcdef" for character in confirmed
        ):
            raise ValueError("mirror batch resolution requires a 64-character status digest")
        normalized_resolver = resolver.strip()
        normalized_reason = reason.strip()
        if not normalized_resolver:
            raise ValueError("mirror batch resolver cannot be empty")
        if len(normalized_reason) < 8:
            raise ValueError(
                "mirror batch resolution reason must contain at least eight characters"
            )
        action_time = self._validated_time(now)
        intent = self._try_load_batch_intent(batch_id)
        if intent is None:
            raise RunArtifactMirrorError("mirror batch intent was not found")
        decision, record, existing = self._load_batch_terminal_evidence(intent)
        if record is not None:
            raise RunArtifactMirrorError("completed mirror batch cannot be resolved")
        if existing is not None:
            expected_id = RunArtifactMirrorBatchResolution.expected_resolution_id(
                batch_id=batch_id,
                plan_id=intent.plan.plan_id,
                kind=kind,
                basis=existing.basis,
                confirmed_status_digest=confirmed,
                invalid_members=existing.invalid_members,
                resolver=normalized_resolver,
                reason=normalized_reason,
                replacement_plan_id=replacement_plan_id,
            )
            if existing.resolution_id != expected_id:
                raise RunArtifactMirrorError(
                    "stored mirror batch resolution differs from this command"
                )
            expected_decision = self._resolution_decision(existing)
            if decision is None:
                decision = self._publish_batch_decision(expected_decision)
            if decision.canonical_bytes() != expected_decision.canonical_bytes():
                raise RunArtifactMirrorError(
                    "mirror batch terminal decision differs from resolution evidence"
                )
            return self._ensure_batch_resolution_sidecar(existing)
        inspection = self.inspect_batch(
            batch_id,
            trusted_public_keys=trusted_public_keys,
            now=action_time,
        )
        if inspection.status.record is not None:
            raise RunArtifactMirrorError("completed mirror batch cannot be resolved")
        if not inspection.resolution_allowed or inspection.resolution_basis is None:
            raise RunArtifactMirrorError("mirror batch is active and not eligible for resolution")
        if inspection.status.state_digest != confirmed:
            raise RunArtifactMirrorError(
                "mirror batch status digest does not match the resolution confirmation"
            )
        resolution_id = RunArtifactMirrorBatchResolution.expected_resolution_id(
            batch_id=batch_id,
            plan_id=intent.plan.plan_id,
            kind=kind,
            basis=inspection.resolution_basis,
            confirmed_status_digest=confirmed,
            invalid_members={
                item.archive_id: item.detail
                for item in inspection.status.members
                if item.state is RunArtifactMirrorBatchMemberState.INVALID
            },
            resolver=normalized_resolver,
            reason=normalized_reason,
            replacement_plan_id=replacement_plan_id,
        )
        resolution = RunArtifactMirrorBatchResolution(
            resolution_id=resolution_id,
            batch_id=batch_id,
            plan_id=intent.plan.plan_id,
            kind=kind,
            basis=inspection.resolution_basis,
            confirmed_status_digest=confirmed,
            confirmed_state_counts=dict(inspection.status.state_counts),
            invalid_members={
                item.archive_id: item.detail
                for item in inspection.status.members
                if item.state is RunArtifactMirrorBatchMemberState.INVALID
            },
            release_count=inspection.status.release_count,
            resolver=normalized_resolver,
            reason=normalized_reason,
            replacement_plan_id=replacement_plan_id,
            created_at=action_time,
        )
        final_status = self.batch_status(
            intent.plan,
            operator=intent.operator,
            reason=intent.reason,
            trusted_public_keys=trusted_public_keys,
            now=action_time,
        )
        if final_status.state_digest != confirmed:
            raise RunArtifactMirrorError("mirror batch changed during resolution confirmation")
        decision, record, existing_resolution = self._load_batch_terminal_evidence(intent)
        if record is not None:
            raise RunArtifactMirrorError("mirror batch completed during resolution confirmation")
        if existing_resolution is not None:
            raise RunArtifactMirrorError("stored mirror batch resolution differs from this command")
        expected_decision = self._resolution_decision(resolution)
        if decision is None:
            decision = self._publish_batch_decision(expected_decision)
        if decision.kind is RunArtifactMirrorBatchDecisionKind.COMPLETED:
            raise RunArtifactMirrorError("mirror batch completed during resolution confirmation")
        decided_resolution = decision.resolved
        if (
            decided_resolution is None
            or decided_resolution.resolution_id != resolution.resolution_id
        ):
            raise RunArtifactMirrorError("stored mirror batch resolution differs from this command")
        return self._ensure_batch_resolution_sidecar(decided_resolution)

    def operations_ledger(
        self,
        *,
        now: datetime | None = None,
    ) -> RunArtifactMirrorOperationsLedger:
        observed_at = self._validated_time(now)
        prefix = f"{self.BATCH_ROOT}/"
        batch_files: dict[str, set[str]] = {}
        unclassified: set[str] = set()
        for key in self.destination.list(prefix):
            if not key.startswith(prefix):
                unclassified.add(key)
                continue
            remainder = key[len(prefix) :]
            batch_id, separator, filename = remainder.partition("/")
            if (
                not separator
                or not self._is_batch_id(batch_id)
                or filename
                not in {"intent.json", "decision.json", "record.json", "resolution.json"}
            ):
                unclassified.add(key)
                continue
            batch_files.setdefault(batch_id, set()).add(filename)
        entries = []
        for batch_id in sorted(batch_files):
            files = batch_files[batch_id]
            if "intent.json" not in files:
                unclassified.update(f"{prefix}{batch_id}/{filename}" for filename in files)
                continue
            intent = self._try_load_batch_intent(batch_id)
            if intent is None:
                raise RunArtifactMirrorError(
                    f"mirror batch intent {batch_id!r} disappeared during ledger scan"
                )
            member_record_count = 0
            for plan in intent.plan.releases:
                mirror_id = RunArtifactMirrorRecord.expected_mirror_id(
                    plan_id=plan.plan_id,
                    operator=intent.operator,
                    reason=intent.reason,
                )
                member_record = self._try_load_record(mirror_id)
                if member_record is None:
                    continue
                if (
                    member_record.plan.canonical_bytes() != plan.canonical_bytes()
                    or member_record.operator != intent.operator
                    or member_record.reason != intent.reason
                ):
                    raise RunArtifactMirrorError(
                        "mirror ledger member evidence does not match its batch intent"
                    )
                member_record_count += 1
            _, batch_record, resolution = self._load_batch_terminal_evidence(intent)
            state = (
                RunArtifactMirrorBatchEvidenceState.RESOLVED
                if resolution is not None
                else (
                    RunArtifactMirrorBatchEvidenceState.COMPLETE
                    if batch_record is not None
                    else (
                        RunArtifactMirrorBatchEvidenceState.MEMBER_EVIDENCE
                        if member_record_count
                        else RunArtifactMirrorBatchEvidenceState.INTENT_ONLY
                    )
                )
            )
            entries.append(
                RunArtifactMirrorBatchLedgerEntry(
                    batch_id=batch_id,
                    plan_id=intent.plan.plan_id,
                    intent_created_at=intent.created_at,
                    selection=intent.plan.selection,
                    release_count=intent.plan.release_count,
                    copy_bytes=intent.plan.copy_bytes,
                    member_record_count=member_record_count,
                    batch_record_present=batch_record is not None,
                    resolution_id=(resolution.resolution_id if resolution is not None else None),
                    resolution_kind=(resolution.kind if resolution is not None else None),
                    replacement_plan_id=(
                        resolution.replacement_plan_id if resolution is not None else None
                    ),
                    state=state,
                )
            )
        entry_tuple = tuple(entries)
        unclassified_tuple = tuple(sorted(unclassified))
        state_counts = {
            state.value: sum(item.state is state for item in entry_tuple)
            for state in RunArtifactMirrorBatchEvidenceState
        }
        state_digest = RunArtifactMirrorOperationsLedger.expected_state_digest(
            entries=entry_tuple,
            unclassified_keys=unclassified_tuple,
        )
        return RunArtifactMirrorOperationsLedger(
            observed_at=observed_at,
            entries=entry_tuple,
            unclassified_keys=unclassified_tuple,
            batch_count=len(entry_tuple),
            state_counts=state_counts,
            state_digest=state_digest,
        )

    @classmethod
    def record_key(cls, mirror_id: str) -> str:
        return f"{cls.RECORD_ROOT}/{mirror_id}.json"

    @classmethod
    def batch_intent_key(cls, batch_id: str) -> str:
        return f"{cls.BATCH_ROOT}/{batch_id}/intent.json"

    @classmethod
    def batch_record_key(cls, batch_id: str) -> str:
        return f"{cls.BATCH_ROOT}/{batch_id}/record.json"

    @classmethod
    def batch_decision_key(cls, batch_id: str) -> str:
        return f"{cls.BATCH_ROOT}/{batch_id}/decision.json"

    @classmethod
    def batch_resolution_key(cls, batch_id: str) -> str:
        return f"{cls.BATCH_ROOT}/{batch_id}/resolution.json"

    @staticmethod
    def _validated_time(value: datetime | None) -> datetime:
        resolved = value or datetime.now(timezone.utc)
        if resolved.tzinfo is None or resolved.utcoffset() is None:
            raise ValueError("mirror time must be timezone-aware")
        return resolved

    @staticmethod
    def _release_references(
        commit: RunArtifactTransportCommit,
        manifest: RunArtifactTransportManifest,
    ) -> tuple[tuple[_MirrorRole, RunArtifactTransportObjectRef], ...]:
        values: list[tuple[_MirrorRole, RunArtifactTransportObjectRef]] = [
            ("chunk", item) for item in manifest.chunks
        ]
        values.append(("checksum", manifest.checksum))
        if manifest.attestation is not None:
            values.append(("attestation", manifest.attestation))
        values.append(("manifest", commit.manifest))
        commit_payload = commit.canonical_bytes() + b"\n"
        values.append(
            (
                "commit",
                RunArtifactTransportObjectRef(
                    key=RunArtifactTransport.commit_key(commit.archive_id),
                    media_type="application/json",
                    size_bytes=len(commit_payload),
                    content_digest=hashlib.sha256(commit_payload).hexdigest(),
                ),
            )
        )
        return tuple(values)

    def _validate_source(
        self,
        plan: RunArtifactMirrorPlan,
        trusted_public_keys: tuple[str, ...],
    ) -> None:
        commit, manifest, _, verification = self.source_transport.inspect_committed_release(
            plan.archive_id,
            trusted_public_keys=trusted_public_keys,
            require_attestation=plan.require_attestation,
        )
        if (
            commit != plan.commit
            or manifest != plan.manifest
            or verification != plan.attestation_verification
        ):
            raise RunArtifactMirrorError(
                "source release or trust verification changed after planning"
            )
        for item in plan.objects:
            current = self._verified_identity(self.source, item.reference)
            if not self._blob_identity_matches(current, item.source):
                raise RunArtifactMirrorError(
                    f"source object {item.reference.key!r} changed after mirror planning"
                )

    def _validate_destination_snapshot(
        self,
        plan: RunArtifactMirrorPlan,
        *,
        allow_progress: bool,
    ) -> None:
        expected_by_key = {item.reference.key: item for item in plan.objects}
        planned_unexpected = {item.key: item for item in plan.unexpected_destination_objects}
        current_keys = self.destination.list(
            f"{RunArtifactTransport.release_root(plan.archive_id)}/"
        )
        for key in current_keys:
            item = expected_by_key.get(key)
            if item is not None:
                current = self._verified_identity(self.destination, item.reference)
                if item.destination is not None:
                    if not self._blob_identity_matches(current, item.destination):
                        raise RunArtifactMirrorError(
                            f"destination object {key!r} changed after mirror planning"
                        )
                elif not allow_progress:
                    raise RunArtifactMirrorError(
                        f"destination object {key!r} appeared during mirror planning"
                    )
                continue
            planned = planned_unexpected.get(key)
            if planned is None:
                raise RunArtifactMirrorError(f"destination gained unplanned object {key!r}")
            current = self._verified_blob_identity(self.destination, key)
            if not self._blob_identity_matches(current, planned):
                raise RunArtifactMirrorError(
                    f"unexpected destination object {key!r} changed after planning"
                )
        current_key_set = set(current_keys)
        for item in plan.objects:
            if item.destination is not None and item.reference.key not in current_key_set:
                raise RunArtifactMirrorError(
                    f"planned destination object {item.reference.key!r} disappeared"
                )
        missing_unexpected = planned_unexpected.keys() - current_key_set
        if missing_unexpected:
            key = min(missing_unexpected)
            raise RunArtifactMirrorError(
                f"unexpected destination object {key!r} disappeared after planning"
            )

    def _validate_destination_complete(
        self,
        plan: RunArtifactMirrorPlan,
        trusted_public_keys: tuple[str, ...],
    ) -> None:
        self._validate_destination_snapshot(plan, allow_progress=True)
        commit, manifest, _, verification = self.destination_transport.inspect_committed_release(
            plan.archive_id,
            trusted_public_keys=trusted_public_keys,
            require_attestation=plan.require_attestation,
        )
        if (
            commit != plan.commit
            or manifest != plan.manifest
            or verification != plan.attestation_verification
        ):
            raise RunArtifactMirrorError(
                "destination release or trust verification does not match mirror plan"
            )

    def _copy_object(self, item: RunArtifactMirrorObjectPlan) -> bool:
        payload = self._verified_payload(self.source, item.reference, expected=item.source)
        return self._put_verified(
            self.destination,
            item.reference.key,
            payload,
            metadata={
                "kind": f"run-archive-mirror-{item.role}",
                "archive-id": item.reference.key.split("/", 2)[1],
                "content-sha256": item.reference.content_digest,
            },
        )

    def _verified_identity(
        self,
        store: ConditionalBlobStore,
        reference: RunArtifactTransportObjectRef,
        *,
        initial: BlobInfo | None = None,
    ) -> BlobInfo:
        payload = self._verified_payload(store, reference, initial=initial)
        info = store.head(reference.key)
        if info is None:
            raise RunArtifactMirrorError(
                f"mirror object {reference.key!r} disappeared during inspection"
            )
        complete = self._complete_identity(info, payload)
        if (
            complete.size_bytes != reference.size_bytes
            or complete.content_sha256 != reference.content_digest
        ):
            raise BlobConflictError(
                f"mirror object {reference.key!r} conflicts with source release"
            )
        return complete

    def _verified_blob_identity(
        self,
        store: ConditionalBlobStore,
        key: str,
    ) -> BlobInfo:
        before = store.head(key)
        if before is None:
            raise RunArtifactMirrorError(f"mirror inventory object {key!r} disappeared")
        try:
            payload = store.get(key)
        except KeyError as error:
            raise RunArtifactMirrorError(f"mirror inventory object {key!r} disappeared") from error
        after = store.head(key)
        if after is None or not self._blob_identity_matches(before, after):
            raise RunArtifactMirrorError(
                f"mirror inventory object {key!r} changed during inspection"
            )
        return self._complete_identity(after, payload)

    def _verified_payload(
        self,
        store: ConditionalBlobStore,
        reference: RunArtifactTransportObjectRef,
        *,
        initial: BlobInfo | None = None,
        expected: BlobInfo | None = None,
    ) -> bytes:
        before = initial or store.head(reference.key)
        if before is None:
            raise RunArtifactMirrorError(f"mirror object {reference.key!r} is missing")
        try:
            payload = store.get(reference.key)
        except KeyError as error:
            raise RunArtifactMirrorError(
                f"mirror object {reference.key!r} disappeared during read"
            ) from error
        if (
            len(payload) != reference.size_bytes
            or hashlib.sha256(payload).hexdigest() != reference.content_digest
        ):
            raise BlobConflictError(f"mirror object {reference.key!r} failed content verification")
        after = store.head(reference.key)
        if after is None or not self._blob_identity_matches(before, after):
            raise RunArtifactMirrorError(f"mirror object {reference.key!r} changed during read")
        complete = self._complete_identity(after, payload)
        if expected is not None and not self._blob_identity_matches(complete, expected):
            raise RunArtifactMirrorError(f"mirror object {reference.key!r} changed after planning")
        return payload

    @staticmethod
    def _complete_identity(info: BlobInfo, payload: bytes) -> BlobInfo:
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != info.size_bytes:
            raise RunArtifactMirrorError(
                f"mirror object {info.key!r} size changed during inspection"
            )
        if info.content_sha256 is not None and info.content_sha256 != digest:
            raise RunArtifactMirrorError(
                f"mirror object {info.key!r} has inconsistent digest metadata"
            )
        if info.etag is None or info.last_modified is None:
            raise RunArtifactMirrorError(
                f"mirror object {info.key!r} has an incomplete storage identity"
            )
        return info.model_copy(update={"content_sha256": digest})

    @staticmethod
    def _put_verified(
        store: ConditionalBlobStore,
        key: str,
        payload: bytes,
        *,
        metadata: dict[str, object],
    ) -> bool:
        result = store.put_if_absent(key, payload, metadata=metadata)
        try:
            stored = store.get(key)
        except KeyError as error:
            raise RunArtifactMirrorError(
                f"mirrored object {key!r} disappeared after write"
            ) from error
        info = store.head(key)
        digest = hashlib.sha256(payload).hexdigest()
        if stored != payload or info is None or info.size_bytes != len(payload):
            raise RunArtifactMirrorError(f"mirrored object {key!r} failed post-write verification")
        if info.content_sha256 is not None and info.content_sha256 != digest:
            raise RunArtifactMirrorError(f"mirrored object {key!r} has invalid digest metadata")
        return result.created

    @classmethod
    def _validate_batch_workers(cls, release_workers: int, object_workers: int) -> None:
        if release_workers < 1 or release_workers > cls.MAX_WORKERS:
            raise ValueError("mirror batch release worker count must be between 1 and 64")
        if object_workers < 1 or object_workers > cls.MAX_WORKERS:
            raise ValueError("mirror batch object worker count must be between 1 and 64")
        if release_workers * object_workers > cls.MAX_WORKERS:
            raise ValueError("mirror batch worker product cannot exceed 64")

    @staticmethod
    def _validate_batch_intent(
        intent: RunArtifactMirrorBatchIntent,
        *,
        plan: RunArtifactMirrorBatchPlan,
        operator: str,
        reason: str,
    ) -> None:
        if (
            intent.plan.canonical_bytes() != plan.canonical_bytes()
            or intent.operator != operator
            or intent.reason != reason
        ):
            raise RunArtifactMirrorError("stored mirror batch intent differs from this command")

    def _execute_batch_members(
        self,
        intent: RunArtifactMirrorBatchIntent,
        *,
        trusted_public_keys: tuple[str, ...],
        release_workers: int,
        object_workers: int,
        action_time: datetime,
    ) -> tuple[RunArtifactMirrorRecord, ...]:
        records_by_archive: dict[str, RunArtifactMirrorRecord] = {}
        with ThreadPoolExecutor(
            max_workers=release_workers,
            thread_name_prefix="run-artifact-mirror-batch",
        ) as executor:
            futures = {
                plan.archive_id: executor.submit(
                    self.execute,
                    plan,
                    confirm_plan_id=plan.plan_id,
                    operator=intent.operator,
                    reason=intent.reason,
                    trusted_public_keys=trusted_public_keys,
                    max_workers=object_workers,
                    now=action_time,
                )
                for plan in intent.plan.releases
            }
            for plan in intent.plan.releases:
                records_by_archive[plan.archive_id] = futures[plan.archive_id].result()
        return tuple(records_by_archive[item.archive_id] for item in intent.plan.releases)

    def _batch_member_status(
        self,
        plan: RunArtifactMirrorPlan,
        *,
        operator: str,
        reason: str,
        trusted_public_keys: tuple[str, ...],
    ) -> RunArtifactMirrorBatchMemberStatus:
        mirror_id = RunArtifactMirrorRecord.expected_mirror_id(
            plan_id=plan.plan_id,
            operator=operator,
            reason=reason,
        )
        try:
            self._validate_source(plan, trusted_public_keys)
            self._validate_destination_snapshot(plan, allow_progress=True)
            record = self._try_load_record(mirror_id)
            if record is not None:
                if (
                    record.plan.canonical_bytes() != plan.canonical_bytes()
                    or record.operator != operator
                    or record.reason != reason
                ):
                    raise RunArtifactMirrorError(
                        "stored mirror evidence differs from its batch member"
                    )
                self._validate_destination_complete(plan, trusted_public_keys)
                return RunArtifactMirrorBatchMemberStatus(
                    archive_id=plan.archive_id,
                    mirror_id=mirror_id,
                    state=RunArtifactMirrorBatchMemberState.COMPLETED,
                    planned_copy_object_count=plan.copy_object_count,
                    planned_copy_bytes=plan.copy_bytes,
                    present_copy_object_count=plan.copy_object_count,
                    present_copy_bytes=plan.copy_bytes,
                    remaining_copy_object_count=0,
                    remaining_copy_bytes=0,
                    record=record,
                    detail="member evidence and complete destination release are valid",
                )
            copy_items = tuple(
                item for item in plan.objects if item.action is RunArtifactMirrorAction.COPY
            )
            present = tuple(
                item for item in copy_items if self.destination.head(item.reference.key) is not None
            )
            present_keys = {item.reference.key for item in present}
            remaining = tuple(item for item in copy_items if item.reference.key not in present_keys)
            commit_key = plan.objects[-1].reference.key
            if commit_key in present_keys and remaining:
                raise RunArtifactMirrorError(
                    "destination commit is present before its complete planned graph"
                )
            present_count = len(present)
            present_bytes = sum(item.reference.size_bytes for item in present)
            remaining_count = len(remaining)
            remaining_bytes = sum(item.reference.size_bytes for item in remaining)
            if not remaining:
                self._validate_destination_complete(plan, trusted_public_keys)
                state = RunArtifactMirrorBatchMemberState.DESTINATION_COMPLETE
                detail = "destination release is complete but member evidence is pending"
            elif not present:
                state = RunArtifactMirrorBatchMemberState.PENDING
                detail = "no planned copy objects are present at the destination"
            else:
                state = RunArtifactMirrorBatchMemberState.PARTIAL
                detail = "exact planned copy progress is present without a destination commit"
            return RunArtifactMirrorBatchMemberStatus(
                archive_id=plan.archive_id,
                mirror_id=mirror_id,
                state=state,
                planned_copy_object_count=plan.copy_object_count,
                planned_copy_bytes=plan.copy_bytes,
                present_copy_object_count=present_count,
                present_copy_bytes=present_bytes,
                remaining_copy_object_count=remaining_count,
                remaining_copy_bytes=remaining_bytes,
                detail=detail,
            )
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            detail = str(error).strip() or error.__class__.__name__
            return RunArtifactMirrorBatchMemberStatus(
                archive_id=plan.archive_id,
                mirror_id=mirror_id,
                state=RunArtifactMirrorBatchMemberState.INVALID,
                planned_copy_object_count=plan.copy_object_count,
                planned_copy_bytes=plan.copy_bytes,
                detail=detail,
            )

    def _try_load_record(self, mirror_id: str) -> RunArtifactMirrorRecord | None:
        key = self.record_key(mirror_id)
        if self.destination.head(key) is None:
            return None
        try:
            payload = self.destination.get(key)
        except KeyError as error:
            raise RunArtifactMirrorError("mirror evidence disappeared during validation") from error
        try:
            record = RunArtifactMirrorRecord.model_validate_json(payload)
        except ValueError as error:
            raise RunArtifactMirrorError("mirror evidence contract is invalid") from error
        if payload != record.canonical_bytes() + b"\n":
            raise RunArtifactMirrorError("mirror evidence is not canonical")
        if record.mirror_id != mirror_id:
            raise RunArtifactMirrorError("mirror evidence ID does not match its key")
        return record

    def _try_load_batch_intent(
        self,
        batch_id: str,
    ) -> RunArtifactMirrorBatchIntent | None:
        self._validate_batch_id(batch_id)
        key = self.batch_intent_key(batch_id)
        if self.destination.head(key) is None:
            return None
        try:
            payload = self.destination.get(key)
        except KeyError as error:
            raise RunArtifactMirrorError(
                "mirror batch intent disappeared during validation"
            ) from error
        try:
            intent = RunArtifactMirrorBatchIntent.model_validate_json(payload)
        except ValueError as error:
            raise RunArtifactMirrorError("mirror batch intent contract is invalid") from error
        if payload != intent.canonical_bytes() + b"\n":
            raise RunArtifactMirrorError("mirror batch intent is not canonical")
        if intent.batch_id != batch_id:
            raise RunArtifactMirrorError("mirror batch intent ID does not match its key")
        return intent

    def _try_load_batch_record(
        self,
        intent: RunArtifactMirrorBatchIntent,
    ) -> RunArtifactMirrorBatchRecord | None:
        key = self.batch_record_key(intent.batch_id)
        if self.destination.head(key) is None:
            return None
        try:
            payload = self.destination.get(key)
        except KeyError as error:
            raise RunArtifactMirrorError(
                "mirror batch record disappeared during validation"
            ) from error
        try:
            record = RunArtifactMirrorBatchRecord.model_validate_json(payload)
        except ValueError as error:
            raise RunArtifactMirrorError("mirror batch record contract is invalid") from error
        if payload != record.canonical_bytes() + b"\n":
            raise RunArtifactMirrorError("mirror batch record is not canonical")
        if record.intent.canonical_bytes() != intent.canonical_bytes():
            raise RunArtifactMirrorError("mirror batch record does not match its intent")
        return record

    def _try_load_batch_decision(
        self,
        intent: RunArtifactMirrorBatchIntent,
    ) -> RunArtifactMirrorBatchDecision | None:
        key = self.batch_decision_key(intent.batch_id)
        if self.destination.head(key) is None:
            return None
        try:
            payload = self.destination.get(key)
        except KeyError as error:
            raise RunArtifactMirrorError(
                "mirror batch decision disappeared during validation"
            ) from error
        try:
            decision = RunArtifactMirrorBatchDecision.model_validate_json(payload)
        except ValueError as error:
            raise RunArtifactMirrorError("mirror batch decision contract is invalid") from error
        if payload != decision.canonical_bytes() + b"\n":
            raise RunArtifactMirrorError("mirror batch decision is not canonical")
        if decision.batch_id != intent.batch_id or decision.plan_id != intent.plan.plan_id:
            raise RunArtifactMirrorError("mirror batch decision does not match its intent")
        return decision

    def _try_load_batch_resolution(
        self,
        intent: RunArtifactMirrorBatchIntent,
    ) -> RunArtifactMirrorBatchResolution | None:
        key = self.batch_resolution_key(intent.batch_id)
        if self.destination.head(key) is None:
            return None
        try:
            payload = self.destination.get(key)
        except KeyError as error:
            raise RunArtifactMirrorError(
                "mirror batch resolution disappeared during validation"
            ) from error
        try:
            resolution = RunArtifactMirrorBatchResolution.model_validate_json(payload)
        except ValueError as error:
            raise RunArtifactMirrorError("mirror batch resolution contract is invalid") from error
        if payload != resolution.canonical_bytes() + b"\n":
            raise RunArtifactMirrorError("mirror batch resolution is not canonical")
        if resolution.batch_id != intent.batch_id or resolution.plan_id != intent.plan.plan_id:
            raise RunArtifactMirrorError("mirror batch resolution does not match its intent")
        return resolution

    def _load_batch_terminal_evidence(
        self,
        intent: RunArtifactMirrorBatchIntent,
    ) -> tuple[
        RunArtifactMirrorBatchDecision | None,
        RunArtifactMirrorBatchRecord | None,
        RunArtifactMirrorBatchResolution | None,
    ]:
        decision = self._try_load_batch_decision(intent)
        record = self._try_load_batch_record(intent)
        resolution = self._try_load_batch_resolution(intent)
        if record is not None and resolution is not None:
            raise RunArtifactMirrorError("mirror batch has conflicting terminal sidecars")
        if decision is None:
            return decision, record, resolution
        if decision.kind is RunArtifactMirrorBatchDecisionKind.COMPLETED:
            completed = decision.completed
            if completed is None:
                raise RunArtifactMirrorError("completed mirror batch decision has no evidence")
            if resolution is not None:
                raise RunArtifactMirrorError(
                    "completed mirror batch decision conflicts with a resolution sidecar"
                )
            if record is not None and record.canonical_bytes() != completed.canonical_bytes():
                raise RunArtifactMirrorError(
                    "mirror batch record differs from its terminal decision"
                )
            return decision, completed, None
        resolved = decision.resolved
        if resolved is None:
            raise RunArtifactMirrorError("resolved mirror batch decision has no evidence")
        if record is not None:
            raise RunArtifactMirrorError(
                "resolved mirror batch decision conflicts with a completion sidecar"
            )
        if resolution is not None and resolution.canonical_bytes() != resolved.canonical_bytes():
            raise RunArtifactMirrorError(
                "mirror batch resolution differs from its terminal decision"
            )
        return decision, None, resolved

    @staticmethod
    def _completion_decision(
        record: RunArtifactMirrorBatchRecord,
    ) -> RunArtifactMirrorBatchDecision:
        decision_id = RunArtifactMirrorBatchDecision.expected_decision_id(
            batch_id=record.intent.batch_id,
            plan_id=record.intent.plan.plan_id,
            kind=RunArtifactMirrorBatchDecisionKind.COMPLETED,
            outcome=record.canonical_bytes(),
        )
        return RunArtifactMirrorBatchDecision(
            decision_id=decision_id,
            batch_id=record.intent.batch_id,
            plan_id=record.intent.plan.plan_id,
            kind=RunArtifactMirrorBatchDecisionKind.COMPLETED,
            completed=record,
            decided_at=record.completed_at,
        )

    @staticmethod
    def _resolution_decision(
        resolution: RunArtifactMirrorBatchResolution,
    ) -> RunArtifactMirrorBatchDecision:
        decision_id = RunArtifactMirrorBatchDecision.expected_decision_id(
            batch_id=resolution.batch_id,
            plan_id=resolution.plan_id,
            kind=RunArtifactMirrorBatchDecisionKind.RESOLVED,
            outcome=resolution.canonical_bytes(),
        )
        return RunArtifactMirrorBatchDecision(
            decision_id=decision_id,
            batch_id=resolution.batch_id,
            plan_id=resolution.plan_id,
            kind=RunArtifactMirrorBatchDecisionKind.RESOLVED,
            resolved=resolution,
            decided_at=resolution.created_at,
        )

    def _publish_batch_decision(
        self,
        decision: RunArtifactMirrorBatchDecision,
    ) -> RunArtifactMirrorBatchDecision:
        try:
            self._put_verified(
                self.destination,
                self.batch_decision_key(decision.batch_id),
                decision.canonical_bytes() + b"\n",
                metadata={
                    "kind": "run-archive-mirror-batch-decision",
                    "batch-id": decision.batch_id,
                    "plan-id": decision.plan_id,
                    "decision-id": decision.decision_id,
                    "outcome": decision.kind.value,
                },
            )
        except BlobConflictError:
            intent = self._try_load_batch_intent(decision.batch_id)
            if intent is None:
                raise RunArtifactMirrorError(
                    "mirror batch intent disappeared after terminal decision conflict"
                ) from None
            loaded = self._try_load_batch_decision(intent)
            if loaded is None:
                raise
            return loaded
        intent = self._try_load_batch_intent(decision.batch_id)
        if intent is None:
            raise RunArtifactMirrorError(
                "mirror batch intent disappeared after terminal decision publication"
            )
        loaded = self._try_load_batch_decision(intent)
        if loaded is None:
            raise RunArtifactMirrorError(
                "mirror batch terminal decision disappeared after publication"
            )
        return loaded

    def _ensure_batch_record_sidecar(
        self,
        record: RunArtifactMirrorBatchRecord,
    ) -> RunArtifactMirrorBatchRecord:
        try:
            self._put_verified(
                self.destination,
                self.batch_record_key(record.intent.batch_id),
                record.canonical_bytes() + b"\n",
                metadata={
                    "kind": "run-archive-mirror-batch-record",
                    "batch-id": record.intent.batch_id,
                    "plan-id": record.intent.plan.plan_id,
                },
            )
        except BlobConflictError:
            loaded = self._try_load_batch_record(record.intent)
            if loaded is None:
                raise
            if loaded.canonical_bytes() != record.canonical_bytes():
                raise RunArtifactMirrorError(
                    "stored mirror batch record differs from its terminal decision"
                ) from None
            return loaded
        loaded = self._try_load_batch_record(record.intent)
        if loaded is None:
            raise RunArtifactMirrorError("mirror batch record disappeared after publication")
        return loaded

    def _ensure_batch_resolution_sidecar(
        self,
        resolution: RunArtifactMirrorBatchResolution,
    ) -> RunArtifactMirrorBatchResolution:
        intent = self._try_load_batch_intent(resolution.batch_id)
        if intent is None:
            raise RunArtifactMirrorError(
                "mirror batch intent disappeared before resolution publication"
            )
        try:
            self._put_verified(
                self.destination,
                self.batch_resolution_key(resolution.batch_id),
                resolution.canonical_bytes() + b"\n",
                metadata={
                    "kind": "run-archive-mirror-batch-resolution",
                    "batch-id": resolution.batch_id,
                    "plan-id": resolution.plan_id,
                    "resolution-id": resolution.resolution_id,
                },
            )
        except BlobConflictError:
            loaded = self._try_load_batch_resolution(intent)
            if loaded is None:
                raise
            if loaded.canonical_bytes() != resolution.canonical_bytes():
                raise RunArtifactMirrorError(
                    "stored mirror batch resolution differs from its terminal decision"
                ) from None
            return loaded
        loaded = self._try_load_batch_resolution(intent)
        if loaded is None:
            raise RunArtifactMirrorError("mirror batch resolution disappeared after publication")
        return loaded

    @staticmethod
    def _is_batch_id(value: str) -> bool:
        return re.fullmatch(r"run_mirror_batch_[0-9a-f]{24}", value) is not None

    @classmethod
    def _validate_batch_id(cls, value: str) -> None:
        if not cls._is_batch_id(value):
            raise ValueError("invalid mirror batch ID")

    @staticmethod
    def _blob_identity_matches(current: BlobInfo, expected: BlobInfo) -> bool:
        return (
            current.key == expected.key
            and current.size_bytes == expected.size_bytes
            and current.content_sha256 == expected.content_sha256
            and current.etag == expected.etag
            and current.last_modified == expected.last_modified
        )

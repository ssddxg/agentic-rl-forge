from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Collection
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, TypeAlias, TypeVar

from agentic_rl_forge.contracts import (
    BlobInfo,
    ContractModel,
    RunArtifactArchiveAttestationVerification,
    RunArtifactArchiveReceipt,
    RunArtifactFetchResult,
    RunArtifactGcBatchIntent,
    RunArtifactGcBatchRecord,
    RunArtifactGcPlan,
    RunArtifactGcPreview,
    RunArtifactGcReason,
    RunArtifactGcRecord,
    RunArtifactGcTombstone,
    RunArtifactPublicationResult,
    RunArtifactReleaseState,
    RunArtifactStoreInventory,
    RunArtifactTransportChunkRef,
    RunArtifactTransportCommit,
    RunArtifactTransportManifest,
    RunArtifactTransportObjectRef,
    RunArtifactTransportStatus,
    SignedRunArtifactArchiveAttestation,
)
from agentic_rl_forge.storage.blobs import BlobConflictError, ConditionalBlobStore, LocalBlobStore
from agentic_rl_forge.storage.run_archives import RunArtifactArchive, RunArtifactArchiveError

_TransportContract = TypeVar("_TransportContract", bound=ContractModel)
_ReleaseDecision: TypeAlias = RunArtifactTransportCommit | RunArtifactGcTombstone


class RunArtifactTransportError(ValueError):
    pass


class RunArtifactTransport:
    DEFAULT_CHUNK_SIZE_BYTES = 8 * 1024 * 1024
    DEFAULT_MAX_WORKERS = 1
    MAX_WORKERS = 64
    GC_RECORD_ROOT = "run-release-gc/records"
    GC_BATCH_ROOT = "run-release-gc/batches"

    def __init__(
        self,
        store: ConditionalBlobStore,
        *,
        chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        if chunk_size_bytes < 1 or chunk_size_bytes > 536_870_912:
            raise ValueError("transport chunk size must be between 1 byte and 512 MiB")
        if max_workers < 1 or max_workers > self.MAX_WORKERS:
            raise ValueError("transport worker count must be between 1 and 64")
        self.store = store
        self.chunk_size_bytes = chunk_size_bytes
        self.max_workers = max_workers

    def publish(
        self,
        archive: Path | str,
        *,
        expected_sha256: str,
        attestation: SignedRunArtifactArchiveAttestation | None = None,
    ) -> RunArtifactPublicationResult:
        archive_path = Path(archive)
        receipt = RunArtifactArchive.inspect(
            archive_path,
            expected_sha256=expected_sha256,
        )
        attestation_payload = self._attestation_payload(attestation, receipt)

        existing = self._try_load_release(receipt.archive_id)
        if existing is not None:
            commit, manifest, remote_attestation = existing
            if manifest.receipt != receipt:
                raise BlobConflictError("committed release belongs to a different archive receipt")
            if remote_attestation != attestation_payload:
                raise BlobConflictError("committed release has different attestation content")
            self._verify_remote_chunks(manifest)
            object_count = 3 + int(manifest.attestation is not None)
            return RunArtifactPublicationResult(
                commit=commit,
                created_chunk_count=0,
                reused_chunk_count=len(manifest.chunks),
                created_object_count=0,
                reused_object_count=object_count,
                commit_created=False,
            )

        chunks, created_chunks, reused_chunks = self._publish_chunks(
            archive_path,
            receipt.archive_id,
        )
        if not chunks:
            raise RunArtifactTransportError("cannot publish an empty archive")

        created_objects = 0
        reused_objects = 0
        checksum_payload = f"{receipt.content_digest}\n".encode("ascii")
        checksum_ref = self._object_ref(
            self.checksum_key(receipt.archive_id),
            checksum_payload,
            media_type="text/plain",
        )
        checksum_created = self._put_verified(
            checksum_ref.key,
            checksum_payload,
            metadata={
                "kind": "run-archive-checksum",
                "archive-id": receipt.archive_id,
            },
        )
        created_objects += int(checksum_created)
        reused_objects += int(not checksum_created)

        attestation_ref = None
        if attestation_payload is not None:
            attestation_ref = self._object_ref(
                self.attestation_key(receipt.archive_id),
                attestation_payload,
                media_type="application/json",
            )
            attestation_created = self._put_verified(
                attestation_ref.key,
                attestation_payload,
                metadata={
                    "kind": "run-archive-attestation",
                    "archive-id": receipt.archive_id,
                },
            )
            created_objects += int(attestation_created)
            reused_objects += int(not attestation_created)

        chunk_tuple = tuple(chunks)
        transport_id = RunArtifactTransportManifest.expected_transport_id(
            receipt=receipt,
            chunk_size_bytes=self.chunk_size_bytes,
            chunks=chunk_tuple,
            checksum=checksum_ref,
            attestation=attestation_ref,
        )
        manifest = RunArtifactTransportManifest(
            transport_id=transport_id,
            receipt=receipt,
            chunk_size_bytes=self.chunk_size_bytes,
            chunks=chunk_tuple,
            checksum=checksum_ref,
            attestation=attestation_ref,
        )
        manifest_payload = manifest.canonical_bytes() + b"\n"
        manifest_ref = self._object_ref(
            self.manifest_key(receipt.archive_id, transport_id),
            manifest_payload,
            media_type="application/json",
        )
        manifest_created = self._put_verified(
            manifest_ref.key,
            manifest_payload,
            metadata={
                "kind": "run-archive-transport-manifest",
                "archive-id": receipt.archive_id,
                "transport-id": transport_id,
            },
        )
        created_objects += int(manifest_created)
        reused_objects += int(not manifest_created)

        commit_id = RunArtifactTransportCommit.expected_commit_id(
            archive_id=receipt.archive_id,
            transport_id=transport_id,
            manifest=manifest_ref,
        )
        commit = RunArtifactTransportCommit(
            commit_id=commit_id,
            archive_id=receipt.archive_id,
            transport_id=transport_id,
            manifest=manifest_ref,
        )
        commit_payload = commit.canonical_bytes() + b"\n"
        try:
            commit_created = self._put_verified(
                self.commit_key(receipt.archive_id),
                commit_payload,
                metadata={
                    "kind": "run-archive-release-commit",
                    "archive-id": receipt.archive_id,
                    "transport-id": transport_id,
                    "commit-id": commit_id,
                },
            )
        except BlobConflictError as error:
            decision = self._load_decision(receipt.archive_id)
            if isinstance(decision, RunArtifactGcTombstone):
                raise RunArtifactTransportError(
                    f"run artifact release {receipt.archive_id!r} is tombstoned"
                ) from error
            raise
        created_objects += int(commit_created)
        reused_objects += int(not commit_created)

        loaded_commit, loaded_manifest, loaded_attestation = self._load_release(receipt.archive_id)
        if (
            loaded_commit != commit
            or loaded_manifest != manifest
            or loaded_attestation != attestation_payload
        ):
            raise RunArtifactTransportError("committed release changed during publication")
        self._verify_remote_chunks(loaded_manifest)
        return RunArtifactPublicationResult(
            commit=commit,
            created_chunk_count=created_chunks,
            reused_chunk_count=reused_chunks,
            created_object_count=created_objects,
            reused_object_count=reused_objects,
            commit_created=commit_created,
        )

    def fetch(
        self,
        archive_id: str,
        output: Path | str,
        *,
        trusted_public_keys: Collection[str] = (),
        require_attestation: bool = False,
    ) -> RunArtifactFetchResult:
        commit, manifest, attestation_payload = self._load_release(archive_id)
        if manifest.receipt.archive_id != archive_id:
            raise RunArtifactTransportError("release manifest archive ID does not match request")
        attestation_verification = self._verify_fetched_attestation(
            manifest,
            attestation_payload,
            trusted_public_keys=trusted_public_keys,
            require_attestation=require_attestation,
        )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            try:
                existing_receipt = RunArtifactArchive.inspect(
                    output_path,
                    expected_sha256=manifest.receipt.content_digest,
                )
            except RunArtifactArchiveError as error:
                raise BlobConflictError(
                    "archive output already exists with different content"
                ) from error
            if existing_receipt != manifest.receipt:
                raise BlobConflictError("archive output already exists with another receipt")
            self._publish_local_sidecars(output_path, manifest, attestation_payload)
            return RunArtifactFetchResult(
                receipt=manifest.receipt,
                commit=commit,
                downloaded_chunk_count=0,
                reused_local_chunk_count=len(manifest.chunks),
                downloaded_bytes=0,
                attestation_verification=attestation_verification,
            )

        partial = self.partial_path(output_path, archive_id)
        partial.parent.mkdir(parents=True, exist_ok=True)
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise RunArtifactTransportError("archive download partial path must be a regular file")
        if not partial.exists():
            descriptor = os.open(
                partial,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            os.close(descriptor)
        resumed_chunks, offset = self._validated_partial_prefix(partial, manifest)
        with self._open_partial(partial) as destination:
            destination.seek(offset)
            downloaded_chunks, downloaded_bytes = self._write_download_chunks(
                destination,
                manifest.chunks[resumed_chunks:],
            )
        partial.chmod(0o644)

        downloaded_receipt = RunArtifactArchive.inspect(
            partial,
            expected_sha256=manifest.receipt.content_digest,
        )
        if downloaded_receipt != manifest.receipt:
            raise RunArtifactTransportError("downloaded archive receipt does not match release")
        self._publish_local_sidecars(output_path, manifest, attestation_payload)
        self._publish_local_archive(partial, output_path, manifest.receipt.content_digest)
        if partial.exists():
            partial.unlink()
        return RunArtifactFetchResult(
            receipt=manifest.receipt,
            commit=commit,
            downloaded_chunk_count=downloaded_chunks,
            reused_local_chunk_count=resumed_chunks,
            downloaded_bytes=downloaded_bytes,
            attestation_verification=attestation_verification,
        )

    def list_committed_archive_ids(self) -> tuple[str, ...]:
        prefix = f"{RunArtifactTransportManifest.ROOT}/"
        suffix = "/commit.json"
        archive_ids = []
        for key in self.store.list(prefix):
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            archive_id = key[len(prefix) : -len(suffix)]
            if "/" not in archive_id:
                decision = self._load_decision(archive_id)
                if isinstance(decision, RunArtifactTransportCommit):
                    self._load_release(archive_id)
                    archive_ids.append(archive_id)
        return tuple(sorted(archive_ids))

    def inspect_committed_release(
        self,
        archive_id: str,
        *,
        trusted_public_keys: Collection[str] = (),
        require_attestation: bool = False,
    ) -> tuple[
        RunArtifactTransportCommit,
        RunArtifactTransportManifest,
        bytes | None,
        RunArtifactArchiveAttestationVerification | None,
    ]:
        commit, manifest, attestation_payload = self._load_release(archive_id)
        verification = self._verify_fetched_attestation(
            manifest,
            attestation_payload,
            trusted_public_keys=trusted_public_keys,
            require_attestation=require_attestation,
        )
        return commit, manifest, attestation_payload, verification

    def inventory(
        self,
        *,
        now: datetime | None = None,
    ) -> RunArtifactStoreInventory:
        observed_at = self._validated_time(now)
        prefix = f"{RunArtifactTransportManifest.ROOT}/"
        archive_ids: set[str] = set()
        invalid_keys: set[str] = set()
        for key in self.store.list(prefix):
            if not key.startswith(prefix):
                invalid_keys.add(key)
                continue
            remainder = key[len(prefix) :]
            archive_id, separator, _ = remainder.partition("/")
            if not separator:
                invalid_keys.add(key)
                continue
            try:
                self._validate_archive_id(archive_id)
            except ValueError:
                invalid_keys.add(key)
                continue
            archive_ids.add(archive_id)
        releases = tuple(
            self.status(archive_id, now=observed_at) for archive_id in sorted(archive_ids)
        )
        invalid_key_tuple = tuple(sorted(invalid_keys))
        state_counts = {
            state.value: sum(item.state is state for item in releases)
            for state in RunArtifactReleaseState
        }
        state_digest = RunArtifactStoreInventory.expected_state_digest(
            releases=releases,
            invalid_keys=invalid_key_tuple,
        )
        return RunArtifactStoreInventory(
            observed_at=observed_at,
            releases=releases,
            invalid_keys=invalid_key_tuple,
            release_count=len(releases),
            object_count=sum(item.object_count for item in releases),
            total_bytes=sum(item.total_bytes for item in releases),
            state_counts=state_counts,
            state_digest=state_digest,
        )

    def plan_garbage_collection(
        self,
        *,
        min_age_seconds: float,
        now: datetime | None = None,
    ) -> RunArtifactGcPlan:
        if min_age_seconds < 1:
            raise ValueError("GC minimum age must be at least one second")
        observed_at = self._validated_time(now)
        inventory = self.inventory(now=observed_at)
        if inventory.invalid_keys:
            raise RunArtifactTransportError(
                "release inventory contains unclassified keys; no GC plan was created"
            )
        invalid = tuple(
            item.archive_id
            for item in inventory.releases
            if item.state is RunArtifactReleaseState.INVALID
        )
        if invalid:
            raise RunArtifactTransportError(
                f"release inventory contains an invalid prefix: {invalid[0]!r}"
            )
        candidates = []
        for status in inventory.releases:
            if status.state is not RunArtifactReleaseState.STAGED:
                continue
            preview = self.preview_garbage_collection(
                status.archive_id,
                min_age_seconds=min_age_seconds,
                now=observed_at,
            )
            if preview.reason is RunArtifactGcReason.INVALID:
                raise RunArtifactTransportError(
                    f"release {status.archive_id!r} became invalid during GC planning"
                )
            if preview.eligible:
                candidates.append(preview)
        final_inventory = self.inventory(now=observed_at)
        if final_inventory.state_digest != inventory.state_digest:
            raise RunArtifactTransportError("release inventory changed during GC planning")
        candidate_tuple = tuple(sorted(candidates, key=lambda item: item.archive_id))
        normalized_age = float(min_age_seconds)
        plan_id = RunArtifactGcPlan.expected_plan_id(
            inventory_state_digest=inventory.state_digest,
            min_age_seconds=normalized_age,
            candidates=candidate_tuple,
        )
        return RunArtifactGcPlan(
            plan_id=plan_id,
            inventory=inventory,
            min_age_seconds=normalized_age,
            candidates=candidate_tuple,
            candidate_count=len(candidate_tuple),
            reclaimable_bytes=sum(item.total_bytes for item in candidate_tuple),
        )

    def status(
        self,
        archive_id: str,
        *,
        now: datetime | None = None,
    ) -> RunArtifactTransportStatus:
        observed_at = self._validated_time(now)
        try:
            objects = self._release_objects(archive_id)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            return self._status_result(
                archive_id,
                RunArtifactReleaseState.INVALID,
                observed_at,
                (),
                detail=f"release inventory is unstable or invalid: {error}",
            )
        if not objects:
            return self._status_result(
                archive_id,
                RunArtifactReleaseState.ABSENT,
                observed_at,
                objects,
                detail="release prefix does not contain objects",
            )
        if any(item.last_modified is None for item in objects):
            return self._status_result(
                archive_id,
                RunArtifactReleaseState.INVALID,
                observed_at,
                objects,
                detail="one or more release objects have no modification time",
            )
        decision_info = next(
            (item for item in objects if item.key == self.commit_key(archive_id)),
            None,
        )
        if decision_info is None:
            return self._status_result(
                archive_id,
                RunArtifactReleaseState.STAGED,
                observed_at,
                objects,
                detail="release contains staged objects but no commit or GC tombstone",
            )
        try:
            decision_payload = self.store.get(decision_info.key)
            decision = self._parse_decision(decision_payload, archive_id)
            decision_digest = hashlib.sha256(decision_payload).hexdigest()
            if isinstance(decision, RunArtifactTransportCommit):
                self._load_release(archive_id)
                return self._status_result(
                    archive_id,
                    RunArtifactReleaseState.COMMITTED,
                    observed_at,
                    objects,
                    decision_digest=decision_digest,
                    commit_id=decision.commit_id,
                    detail="release commit and complete object graph are valid",
                )
            record = self._try_load_gc_record(decision)
            if record is None:
                return self._status_result(
                    archive_id,
                    RunArtifactReleaseState.GC_IN_PROGRESS,
                    observed_at,
                    objects,
                    decision_digest=decision_digest,
                    gc_id=decision.gc_id,
                    detail="GC tombstone owns the release prefix; completion is pending",
                )
            remaining = tuple(item for item in objects if item.key != self.commit_key(archive_id))
            if remaining:
                raise RunArtifactTransportError(
                    "garbage collection record exists while target objects remain"
                )
            return self._status_result(
                archive_id,
                RunArtifactReleaseState.GARBAGE_COLLECTED,
                observed_at,
                objects,
                decision_digest=decision_digest,
                gc_id=decision.gc_id,
                detail="GC tombstone and immutable completion record are valid",
            )
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            return self._status_result(
                archive_id,
                RunArtifactReleaseState.INVALID,
                observed_at,
                objects,
                detail=f"release decision or object graph is invalid: {error}",
            )

    def preview_garbage_collection(
        self,
        archive_id: str,
        *,
        min_age_seconds: float,
        now: datetime | None = None,
    ) -> RunArtifactGcPreview:
        if min_age_seconds < 1:
            raise ValueError("GC minimum age must be at least one second")
        observed_at = self._validated_time(now)
        status = self.status(archive_id, now=observed_at)
        reason_by_state = {
            RunArtifactReleaseState.ABSENT: RunArtifactGcReason.NOT_FOUND,
            RunArtifactReleaseState.COMMITTED: RunArtifactGcReason.COMMITTED,
            RunArtifactReleaseState.GC_IN_PROGRESS: RunArtifactGcReason.TOMBSTONED,
            RunArtifactReleaseState.GARBAGE_COLLECTED: RunArtifactGcReason.TOMBSTONED,
            RunArtifactReleaseState.INVALID: RunArtifactGcReason.INVALID,
        }
        reason = reason_by_state.get(status.state)
        if reason is not None:
            return self._gc_preview(
                archive_id,
                observed_at=observed_at,
                min_age_seconds=min_age_seconds,
                reason=reason,
            )

        try:
            objects = tuple(
                self._complete_blob_identity(item) for item in self._release_objects(archive_id)
            )
            if not objects:
                return self._gc_preview(
                    archive_id,
                    observed_at=observed_at,
                    min_age_seconds=min_age_seconds,
                    reason=RunArtifactGcReason.NOT_FOUND,
                )
            if any(item.key == self.commit_key(archive_id) for item in objects):
                return self.preview_garbage_collection(
                    archive_id,
                    min_age_seconds=min_age_seconds,
                    now=observed_at,
                )
            current_keys = self.store.list(f"{self.release_root(archive_id)}/")
            if tuple(item.key for item in objects) != current_keys:
                raise RunArtifactTransportError("release inventory changed during GC preview")
            latest = max(item.last_modified for item in objects if item.last_modified is not None)
            eligible_at = latest + timedelta(seconds=min_age_seconds)
            preview_reason = (
                RunArtifactGcReason.ELIGIBLE
                if observed_at >= eligible_at
                else RunArtifactGcReason.TOO_RECENT
            )
            return self._gc_preview(
                archive_id,
                observed_at=observed_at,
                min_age_seconds=min_age_seconds,
                reason=preview_reason,
                objects=objects,
                eligible_at=eligible_at,
            )
        except (KeyError, OSError, RuntimeError, ValueError):
            return self._gc_preview(
                archive_id,
                observed_at=observed_at,
                min_age_seconds=min_age_seconds,
                reason=RunArtifactGcReason.INVALID,
            )

    def garbage_collect(
        self,
        archive_id: str,
        *,
        min_age_seconds: float,
        confirm_state_digest: str,
        operator: str,
        reason: str,
        now: datetime | None = None,
    ) -> RunArtifactGcRecord:
        if min_age_seconds < 1:
            raise ValueError("GC minimum age must be at least one second")
        confirmed = confirm_state_digest.strip().lower()
        if len(confirmed) != 64 or any(
            character not in "0123456789abcdef" for character in confirmed
        ):
            raise ValueError("GC confirmation must be a 64-character state digest")
        normalized_operator = operator.strip()
        normalized_reason = reason.strip()
        if not normalized_operator:
            raise ValueError("GC operator cannot be empty")
        if len(normalized_reason) < 8:
            raise ValueError("GC reason must contain at least eight characters")
        action_time = self._validated_time(now)

        decision = self._load_decision(archive_id)
        if isinstance(decision, RunArtifactTransportCommit):
            raise RunArtifactTransportError(
                "committed releases are protected from garbage collection"
            )
        if isinstance(decision, RunArtifactGcTombstone):
            tombstone = self._validate_gc_resume(
                decision,
                archive_id=archive_id,
                min_age_seconds=min_age_seconds,
                state_digest=confirmed,
                operator=normalized_operator,
                reason=normalized_reason,
            )
        else:
            preview = self.preview_garbage_collection(
                archive_id,
                min_age_seconds=min_age_seconds,
                now=action_time,
            )
            if not preview.eligible:
                raise RunArtifactTransportError(
                    f"release is not eligible for garbage collection: {preview.reason.value}"
                )
            if preview.state_digest != confirmed:
                raise RunArtifactTransportError("GC state digest does not match current preview")
            tombstone = self._create_gc_tombstone(
                preview,
                operator=normalized_operator,
                reason=normalized_reason,
                created_at=action_time,
            )
            try:
                self._put_verified(
                    self.commit_key(archive_id),
                    tombstone.canonical_bytes() + b"\n",
                    metadata={
                        "kind": "run-archive-gc-tombstone",
                        "archive-id": archive_id,
                        "gc-id": tombstone.gc_id,
                        "state-digest": preview.state_digest,
                    },
                )
            except BlobConflictError:
                raced = self._load_decision(archive_id)
                if isinstance(raced, RunArtifactTransportCommit):
                    raise RunArtifactTransportError(
                        "release was committed before garbage collection acquired its tombstone"
                    ) from None
                if not isinstance(raced, RunArtifactGcTombstone):
                    raise RunArtifactTransportError(
                        "release decision changed during garbage collection"
                    ) from None
                tombstone = self._validate_gc_resume(
                    raced,
                    archive_id=archive_id,
                    min_age_seconds=min_age_seconds,
                    state_digest=confirmed,
                    operator=normalized_operator,
                    reason=normalized_reason,
                )

        existing_record = self._try_load_gc_record(tombstone)
        if existing_record is not None:
            self._assert_gc_targets_absent(tombstone)
            return existing_record

        target_by_key = {item.key: item for item in tombstone.preview.objects}
        current = tuple(
            self._complete_blob_identity(item) for item in self._release_objects(archive_id)
        )
        current_targets = tuple(item for item in current if item.key != self.commit_key(archive_id))
        unexpected = tuple(item.key for item in current_targets if item.key not in target_by_key)
        if unexpected:
            raise RunArtifactTransportError(
                f"release gained unconfirmed objects after GC preview: {unexpected[0]!r}"
            )
        for item in current_targets:
            if not self._blob_identity_matches(item, target_by_key[item.key]):
                raise RunArtifactTransportError(
                    f"release object {item.key!r} changed after GC preview"
                )

        deleted_count = 0
        current_keys = {item.key for item in current_targets}
        already_missing_count = len(target_by_key.keys() - current_keys)
        for item in current_targets:
            if self.store.delete_if_match(item.key, target_by_key[item.key]):
                deleted_count += 1
            else:
                already_missing_count += 1
        self._assert_gc_targets_absent(tombstone)
        record = RunArtifactGcRecord(
            gc_id=tombstone.gc_id,
            archive_id=archive_id,
            state_digest=tombstone.preview.state_digest,
            operator=tombstone.operator,
            reason=tombstone.reason,
            started_at=tombstone.created_at,
            completed_at=action_time,
            target_object_count=len(tombstone.preview.objects),
            deleted_object_count=deleted_count,
            already_missing_object_count=already_missing_count,
            target_bytes=sum(item.size_bytes for item in tombstone.preview.objects),
        )
        payload = record.canonical_bytes() + b"\n"
        try:
            self._put_verified(
                self.gc_record_key(record.gc_id),
                payload,
                metadata={
                    "kind": "run-archive-gc-record",
                    "archive-id": archive_id,
                    "gc-id": record.gc_id,
                    "state-digest": record.state_digest,
                },
            )
        except BlobConflictError:
            loaded = self._try_load_gc_record(tombstone)
            if loaded is None:
                raise
            return loaded
        return record

    def execute_garbage_collection_plan(
        self,
        plan: RunArtifactGcPlan,
        *,
        confirm_plan_id: str,
        operator: str,
        reason: str,
        max_workers: int = 1,
        now: datetime | None = None,
    ) -> RunArtifactGcBatchRecord:
        confirmed = confirm_plan_id.strip()
        if confirmed != plan.plan_id:
            raise RunArtifactTransportError("GC batch confirmation does not match the plan ID")
        normalized_operator = operator.strip()
        normalized_reason = reason.strip()
        if not normalized_operator:
            raise ValueError("GC batch operator cannot be empty")
        if len(normalized_reason) < 8:
            raise ValueError("GC batch reason must contain at least eight characters")
        if max_workers < 1 or max_workers > self.MAX_WORKERS:
            raise ValueError("GC batch worker count must be between 1 and 64")
        if not plan.candidates:
            raise RunArtifactTransportError("GC plan contains no eligible candidates")
        action_time = self._validated_time(now)
        batch_id = RunArtifactGcBatchIntent.expected_batch_id(
            plan_id=plan.plan_id,
            operator=normalized_operator,
            reason=normalized_reason,
        )
        expected_intent = RunArtifactGcBatchIntent(
            batch_id=batch_id,
            plan=plan,
            operator=normalized_operator,
            reason=normalized_reason,
            created_at=action_time,
        )
        intent = self._try_load_gc_batch_intent(batch_id)
        if intent is None:
            try:
                self._put_verified(
                    self.gc_batch_intent_key(batch_id),
                    expected_intent.canonical_bytes() + b"\n",
                    metadata={
                        "kind": "run-archive-gc-batch-intent",
                        "batch-id": batch_id,
                        "plan-id": plan.plan_id,
                    },
                )
                intent = expected_intent
            except BlobConflictError:
                intent = self._try_load_gc_batch_intent(batch_id)
                if intent is None:
                    raise
        if (
            intent.plan != plan
            or intent.operator != normalized_operator
            or intent.reason != normalized_reason
        ):
            raise RunArtifactTransportError("stored GC batch intent differs from this command")
        existing = self._try_load_gc_batch_record(intent)
        if existing is not None:
            return existing

        records_by_archive: dict[str, RunArtifactGcRecord] = {}
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="run-artifact-gc-batch",
        ) as executor:
            futures = {
                preview.archive_id: executor.submit(
                    self.garbage_collect,
                    preview.archive_id,
                    min_age_seconds=plan.min_age_seconds,
                    confirm_state_digest=preview.state_digest,
                    operator=intent.operator,
                    reason=intent.reason,
                    now=action_time,
                )
                for preview in plan.candidates
            }
            for preview in plan.candidates:
                records_by_archive[preview.archive_id] = futures[preview.archive_id].result()
        records = tuple(records_by_archive[item.archive_id] for item in plan.candidates)
        record = RunArtifactGcBatchRecord(
            intent=intent,
            completed_at=action_time,
            records=records,
            candidate_count=len(records),
            deleted_object_count=sum(item.deleted_object_count for item in records),
            already_missing_object_count=sum(item.already_missing_object_count for item in records),
            target_bytes=sum(item.target_bytes for item in records),
        )
        try:
            self._put_verified(
                self.gc_batch_record_key(batch_id),
                record.canonical_bytes() + b"\n",
                metadata={
                    "kind": "run-archive-gc-batch-record",
                    "batch-id": batch_id,
                    "plan-id": plan.plan_id,
                },
            )
        except BlobConflictError:
            loaded = self._try_load_gc_batch_record(intent)
            if loaded is None:
                raise
            return loaded
        return record

    @staticmethod
    def release_root(archive_id: str) -> str:
        return f"{RunArtifactTransportManifest.ROOT}/{archive_id}"

    @classmethod
    def chunk_key(cls, archive_id: str, index: int, digest: str) -> str:
        return f"{cls.release_root(archive_id)}/chunks/{index:08d}-{digest}.part"

    @classmethod
    def checksum_key(cls, archive_id: str) -> str:
        return f"{cls.release_root(archive_id)}/archive.sha256"

    @classmethod
    def attestation_key(cls, archive_id: str) -> str:
        return f"{cls.release_root(archive_id)}/attestation.json"

    @classmethod
    def manifest_key(cls, archive_id: str, transport_id: str) -> str:
        return f"{cls.release_root(archive_id)}/manifests/{transport_id}.json"

    @classmethod
    def commit_key(cls, archive_id: str) -> str:
        return f"{cls.release_root(archive_id)}/commit.json"

    @classmethod
    def gc_record_key(cls, gc_id: str) -> str:
        return f"{cls.GC_RECORD_ROOT}/{gc_id}.json"

    @classmethod
    def gc_batch_intent_key(cls, batch_id: str) -> str:
        return f"{cls.GC_BATCH_ROOT}/{batch_id}/intent.json"

    @classmethod
    def gc_batch_record_key(cls, batch_id: str) -> str:
        return f"{cls.GC_BATCH_ROOT}/{batch_id}/record.json"

    @staticmethod
    def partial_path(output: Path, archive_id: str) -> Path:
        return output.with_name(f".{output.name}.{archive_id}.part")

    @staticmethod
    def fetched_attestation_path(output: Path) -> Path:
        return output.with_name(f"{output.name}.attestation.json")

    @staticmethod
    def _validated_time(value: datetime | None) -> datetime:
        resolved = value or datetime.now(timezone.utc)
        if resolved.tzinfo is None or resolved.utcoffset() is None:
            raise ValueError("transport time must be timezone-aware")
        return resolved

    @staticmethod
    def _validate_archive_id(archive_id: str) -> None:
        prefix = "run_archive_"
        suffix = archive_id[len(prefix) :] if archive_id.startswith(prefix) else ""
        if len(suffix) != 24 or any(character not in "0123456789abcdef" for character in suffix):
            raise ValueError("run artifact archive ID is invalid")

    def _release_objects(self, archive_id: str) -> tuple[BlobInfo, ...]:
        self._validate_archive_id(archive_id)
        prefix = f"{self.release_root(archive_id)}/"
        objects = []
        for key in self.store.list(prefix):
            if not key.startswith(prefix):
                raise RunArtifactTransportError("release inventory escaped its prefix")
            info = self.store.head(key)
            if info is None:
                raise RunArtifactTransportError(
                    f"release object {key!r} disappeared during inventory"
                )
            objects.append(info)
        return tuple(sorted(objects, key=lambda item: item.key))

    @staticmethod
    def _status_result(
        archive_id: str,
        state: RunArtifactReleaseState,
        observed_at: datetime,
        objects: tuple[BlobInfo, ...],
        *,
        detail: str,
        decision_digest: str | None = None,
        commit_id: str | None = None,
        gc_id: str | None = None,
    ) -> RunArtifactTransportStatus:
        modified = tuple(item.last_modified for item in objects if item.last_modified is not None)
        return RunArtifactTransportStatus(
            archive_id=archive_id,
            state=state,
            observed_at=observed_at,
            object_count=len(objects),
            total_bytes=sum(item.size_bytes for item in objects),
            oldest_modified_at=min(modified) if modified else None,
            newest_modified_at=max(modified) if modified else None,
            decision_digest=decision_digest,
            commit_id=commit_id,
            gc_id=gc_id,
            detail=detail,
        )

    def _complete_blob_identity(self, info: BlobInfo) -> BlobInfo:
        if info.etag is None or info.last_modified is None:
            raise RunArtifactTransportError(
                f"release object {info.key!r} has an incomplete storage identity"
            )
        content_sha256 = info.content_sha256
        if content_sha256 is None:
            payload = self.store.get(info.key)
            if len(payload) != info.size_bytes:
                raise RunArtifactTransportError(
                    f"release object {info.key!r} changed while hashing"
                )
            content_sha256 = hashlib.sha256(payload).hexdigest()
        return info.model_copy(update={"content_sha256": content_sha256})

    @staticmethod
    def _gc_preview(
        archive_id: str,
        *,
        observed_at: datetime,
        min_age_seconds: float,
        reason: RunArtifactGcReason,
        objects: tuple[BlobInfo, ...] = (),
        eligible_at: datetime | None = None,
    ) -> RunArtifactGcPreview:
        ordered = tuple(sorted(objects, key=lambda item: item.key))
        normalized_age = float(min_age_seconds)
        state_digest = RunArtifactGcPreview.expected_state_digest(
            archive_id=archive_id,
            min_age_seconds=normalized_age,
            eligible_at=eligible_at,
            objects=ordered,
        )
        return RunArtifactGcPreview(
            archive_id=archive_id,
            eligible=reason is RunArtifactGcReason.ELIGIBLE,
            reason=reason,
            observed_at=observed_at,
            min_age_seconds=normalized_age,
            eligible_at=eligible_at,
            objects=ordered,
            object_count=len(ordered),
            total_bytes=sum(item.size_bytes for item in ordered),
            state_digest=state_digest,
        )

    @staticmethod
    def _create_gc_tombstone(
        preview: RunArtifactGcPreview,
        *,
        operator: str,
        reason: str,
        created_at: datetime,
    ) -> RunArtifactGcTombstone:
        gc_id = RunArtifactGcTombstone.expected_gc_id(
            state_digest=preview.state_digest,
            operator=operator,
            reason=reason,
            created_at=created_at,
        )
        return RunArtifactGcTombstone(
            gc_id=gc_id,
            preview=preview,
            operator=operator,
            reason=reason,
            created_at=created_at,
        )

    @staticmethod
    def _validate_gc_resume(
        tombstone: RunArtifactGcTombstone,
        *,
        archive_id: str,
        min_age_seconds: float,
        state_digest: str,
        operator: str,
        reason: str,
    ) -> RunArtifactGcTombstone:
        if tombstone.preview.archive_id != archive_id:
            raise RunArtifactTransportError("GC tombstone belongs to another release")
        if tombstone.preview.min_age_seconds != min_age_seconds:
            raise RunArtifactTransportError("GC retry minimum age differs from tombstone")
        if tombstone.preview.state_digest != state_digest:
            raise RunArtifactTransportError("GC retry state digest differs from tombstone")
        if tombstone.operator != operator or tombstone.reason != reason:
            raise RunArtifactTransportError("GC retry operator or reason differs from tombstone")
        return tombstone

    def _try_load_gc_record(
        self,
        tombstone: RunArtifactGcTombstone,
    ) -> RunArtifactGcRecord | None:
        key = self.gc_record_key(tombstone.gc_id)
        if self.store.head(key) is None:
            return None
        try:
            payload = self.store.get(key)
        except KeyError as error:
            raise RunArtifactTransportError("GC record disappeared during validation") from error
        record = self._parse_canonical(payload, RunArtifactGcRecord)
        if (
            record.gc_id != tombstone.gc_id
            or record.archive_id != tombstone.preview.archive_id
            or record.state_digest != tombstone.preview.state_digest
            or record.operator != tombstone.operator
            or record.reason != tombstone.reason
            or record.started_at != tombstone.created_at
        ):
            raise RunArtifactTransportError("GC record does not match its tombstone")
        return record

    def _try_load_gc_batch_intent(
        self,
        batch_id: str,
    ) -> RunArtifactGcBatchIntent | None:
        key = self.gc_batch_intent_key(batch_id)
        if self.store.head(key) is None:
            return None
        try:
            payload = self.store.get(key)
        except KeyError as error:
            raise RunArtifactTransportError(
                "GC batch intent disappeared during validation"
            ) from error
        intent = self._parse_canonical(payload, RunArtifactGcBatchIntent)
        if intent.batch_id != batch_id:
            raise RunArtifactTransportError("GC batch intent ID does not match its key")
        return intent

    def _try_load_gc_batch_record(
        self,
        intent: RunArtifactGcBatchIntent,
    ) -> RunArtifactGcBatchRecord | None:
        key = self.gc_batch_record_key(intent.batch_id)
        if self.store.head(key) is None:
            return None
        try:
            payload = self.store.get(key)
        except KeyError as error:
            raise RunArtifactTransportError(
                "GC batch record disappeared during validation"
            ) from error
        record = self._parse_canonical(payload, RunArtifactGcBatchRecord)
        if record.intent != intent:
            raise RunArtifactTransportError("GC batch record does not match its intent")
        return record

    def _assert_gc_targets_absent(self, tombstone: RunArtifactGcTombstone) -> None:
        remaining = tuple(
            key
            for key in self.store.list(f"{self.release_root(tombstone.preview.archive_id)}/")
            if key != self.commit_key(tombstone.preview.archive_id)
        )
        if remaining:
            raise RunArtifactTransportError(
                f"release still contains objects after garbage collection: {remaining[0]!r}"
            )

    @staticmethod
    def _blob_identity_matches(current: BlobInfo, expected: BlobInfo) -> bool:
        return (
            current.key == expected.key
            and current.size_bytes == expected.size_bytes
            and current.content_sha256 == expected.content_sha256
            and current.etag == expected.etag
            and current.last_modified == expected.last_modified
        )

    def _load_decision(self, archive_id: str) -> _ReleaseDecision | None:
        try:
            payload = self.store.get(self.commit_key(archive_id))
        except KeyError:
            return None
        return self._parse_decision(payload, archive_id)

    def _parse_decision(self, payload: bytes, archive_id: str) -> _ReleaseDecision:
        try:
            commit = self._parse_canonical(payload, RunArtifactTransportCommit)
        except RunArtifactTransportError:
            commit = None
        if commit is not None:
            if commit.archive_id != archive_id:
                raise RunArtifactTransportError("release commit archive ID does not match its key")
            return commit
        try:
            tombstone = self._parse_canonical(payload, RunArtifactGcTombstone)
        except RunArtifactTransportError as error:
            raise RunArtifactTransportError(
                "release metadata contract is invalid or decision is non-canonical"
            ) from error
        if tombstone.preview.archive_id != archive_id:
            raise RunArtifactTransportError(
                "release GC tombstone archive ID does not match its key"
            )
        return tombstone

    def _try_load_release(
        self,
        archive_id: str,
    ) -> (
        tuple[
            RunArtifactTransportCommit,
            RunArtifactTransportManifest,
            bytes | None,
        ]
        | None
    ):
        decision = self._load_decision(archive_id)
        if decision is None:
            return None
        if isinstance(decision, RunArtifactGcTombstone):
            raise RunArtifactTransportError(f"run artifact release {archive_id!r} is tombstoned")
        return self._load_release(archive_id)

    def _load_release(
        self,
        archive_id: str,
    ) -> tuple[
        RunArtifactTransportCommit,
        RunArtifactTransportManifest,
        bytes | None,
    ]:
        decision = self._load_decision(archive_id)
        if decision is None:
            raise RunArtifactTransportError(f"run artifact release {archive_id!r} is not committed")
        if isinstance(decision, RunArtifactGcTombstone):
            raise RunArtifactTransportError(f"run artifact release {archive_id!r} is tombstoned")
        commit = decision
        manifest_payload = self._get_verified(commit.manifest)
        manifest = self._parse_canonical(manifest_payload, RunArtifactTransportManifest)
        if manifest.transport_id != commit.transport_id:
            raise RunArtifactTransportError("release commit points to another transport manifest")
        if manifest.receipt.archive_id != archive_id:
            raise RunArtifactTransportError("release manifest archive ID does not match its key")
        checksum_payload = self._get_verified(manifest.checksum)
        expected_checksum = f"{manifest.receipt.content_digest}\n".encode("ascii")
        if checksum_payload != expected_checksum:
            raise RunArtifactTransportError("release checksum does not match archive receipt")
        attestation_payload = (
            self._get_verified(manifest.attestation) if manifest.attestation is not None else None
        )
        if attestation_payload is not None:
            try:
                signed = SignedRunArtifactArchiveAttestation.model_validate_json(
                    attestation_payload
                )
            except ValueError as error:
                raise RunArtifactTransportError(
                    "release attestation contract is invalid"
                ) from error
            if attestation_payload != signed.canonical_bytes() + b"\n":
                raise RunArtifactTransportError("release attestation is not canonical")
            if signed.attestation.receipt != manifest.receipt:
                raise RunArtifactTransportError(
                    "release attestation receipt does not match transport manifest"
                )
        return commit, manifest, attestation_payload

    def _publish_chunks(
        self,
        archive: Path,
        archive_id: str,
    ) -> tuple[tuple[RunArtifactTransportChunkRef, ...], int, int]:
        results: dict[int, tuple[RunArtifactTransportChunkRef, bool]] = {}
        pending: set[Future[tuple[RunArtifactTransportChunkRef, bool]]] = set()
        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="run-artifact-upload",
        ) as executor:
            with archive.open("rb") as source:
                index = 0
                while payload := source.read(self.chunk_size_bytes):
                    pending.add(executor.submit(self._publish_chunk, archive_id, index, payload))
                    index += 1
                    if len(pending) >= self.max_workers:
                        self._collect_chunk_uploads(pending, results)
            while pending:
                self._collect_chunk_uploads(pending, results)
        chunks = tuple(results[index][0] for index in sorted(results))
        created = sum(int(item[1]) for item in results.values())
        return chunks, created, len(chunks) - created

    def _publish_chunk(
        self,
        archive_id: str,
        index: int,
        payload: bytes,
    ) -> tuple[RunArtifactTransportChunkRef, bool]:
        digest = hashlib.sha256(payload).hexdigest()
        key = self.chunk_key(archive_id, index, digest)
        created = self._put_verified(
            key,
            payload,
            metadata={
                "kind": "run-archive-chunk",
                "archive-id": archive_id,
                "chunk-index": index,
                "chunk-sha256": digest,
            },
        )
        return (
            RunArtifactTransportChunkRef(
                index=index,
                key=key,
                media_type="application/octet-stream",
                size_bytes=len(payload),
                content_digest=digest,
            ),
            created,
        )

    @staticmethod
    def _collect_chunk_uploads(
        pending: set[Future[tuple[RunArtifactTransportChunkRef, bool]]],
        results: dict[int, tuple[RunArtifactTransportChunkRef, bool]],
    ) -> None:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            pending.remove(future)
            reference, created = future.result()
            results[reference.index] = (reference, created)

    def _write_download_chunks(
        self,
        destination: BinaryIO,
        chunks: tuple[RunArtifactTransportChunkRef, ...],
    ) -> tuple[int, int]:
        if not chunks:
            return 0, 0
        futures: dict[int, Future[bytes]] = {}
        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="run-artifact-download",
        ) as executor:
            next_submit = 0
            while next_submit < min(self.max_workers, len(chunks)):
                futures[next_submit] = executor.submit(
                    self._get_verified,
                    chunks[next_submit],
                )
                next_submit += 1
            downloaded_bytes = 0
            for index in range(len(chunks)):
                payload = futures.pop(index).result()
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
                downloaded_bytes += len(payload)
                if next_submit < len(chunks):
                    futures[next_submit] = executor.submit(
                        self._get_verified,
                        chunks[next_submit],
                    )
                    next_submit += 1
        return len(chunks), downloaded_bytes

    def _verify_remote_chunks(self, manifest: RunArtifactTransportManifest) -> None:
        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="run-artifact-verify",
        ) as executor:
            futures = tuple(executor.submit(self._get_verified, chunk) for chunk in manifest.chunks)
            for future in futures:
                future.result()

    def _put_verified(
        self,
        key: str,
        payload: bytes,
        *,
        metadata: dict[str, object],
    ) -> bool:
        result = self.store.put_if_absent(key, payload, metadata=metadata)
        stored = self.store.get(key)
        if stored != payload:
            raise RunArtifactTransportError(f"stored object {key!r} failed post-write verification")
        info = self.store.head(key)
        digest = hashlib.sha256(payload).hexdigest()
        if info is None or info.size_bytes != len(payload):
            raise RunArtifactTransportError(f"stored object {key!r} has invalid metadata")
        if info.content_sha256 is not None and info.content_sha256 != digest:
            raise RunArtifactTransportError(f"stored object {key!r} has invalid digest metadata")
        return result.created

    def _get_verified(self, reference: RunArtifactTransportObjectRef) -> bytes:
        try:
            payload = self.store.get(reference.key)
        except KeyError as error:
            raise RunArtifactTransportError(
                f"release object {reference.key!r} is missing"
            ) from error
        if (
            len(payload) != reference.size_bytes
            or hashlib.sha256(payload).hexdigest() != reference.content_digest
        ):
            raise RunArtifactTransportError(
                f"release object {reference.key!r} failed content verification"
            )
        return payload

    @staticmethod
    def _object_ref(
        key: str,
        payload: bytes,
        *,
        media_type: str,
    ) -> RunArtifactTransportObjectRef:
        return RunArtifactTransportObjectRef(
            key=key,
            media_type=media_type,
            size_bytes=len(payload),
            content_digest=hashlib.sha256(payload).hexdigest(),
        )

    @staticmethod
    def _parse_canonical(
        payload: bytes,
        model: type[_TransportContract],
    ) -> _TransportContract:
        try:
            parsed = model.model_validate_json(payload)
        except ValueError as error:
            raise RunArtifactTransportError("release metadata contract is invalid") from error
        if payload != parsed.canonical_bytes() + b"\n":
            raise RunArtifactTransportError("release metadata is not canonical")
        return parsed

    @staticmethod
    def _attestation_payload(
        attestation: SignedRunArtifactArchiveAttestation | None,
        receipt: RunArtifactArchiveReceipt,
    ) -> bytes | None:
        if attestation is None:
            return None
        from agentic_rl_forge.data.attestations import RunArtifactArchiveAttestor

        verification = RunArtifactArchiveAttestor.verify(
            attestation,
            receipt,
            trusted_public_keys=(attestation.signature.public_key_base64,),
        )
        if not verification.valid:
            raise RunArtifactTransportError(
                "archive attestation does not match the published receipt"
            )
        return attestation.canonical_bytes() + b"\n"

    @staticmethod
    def _verify_fetched_attestation(
        manifest: RunArtifactTransportManifest,
        payload: bytes | None,
        *,
        trusted_public_keys: Collection[str],
        require_attestation: bool,
    ) -> RunArtifactArchiveAttestationVerification | None:
        from agentic_rl_forge.data.attestations import RunArtifactArchiveAttestor

        trusted_keys = tuple(trusted_public_keys)
        if payload is None:
            if require_attestation or trusted_keys:
                raise RunArtifactTransportError("release does not contain an attestation")
            return None
        try:
            signed = SignedRunArtifactArchiveAttestation.model_validate_json(payload)
        except ValueError as error:
            raise RunArtifactTransportError("release attestation is invalid") from error
        embedded_verification = RunArtifactArchiveAttestor.verify(
            signed,
            manifest.receipt,
            trusted_public_keys=(signed.signature.public_key_base64,),
        )
        if not embedded_verification.valid:
            raise RunArtifactTransportError(
                "release attestation does not match its archive receipt"
            )
        if not trusted_keys:
            if require_attestation:
                raise RunArtifactTransportError(
                    "trusted public keys are required for authenticated fetch"
                )
            return None
        verification = RunArtifactArchiveAttestor.verify(
            signed,
            manifest.receipt,
            trusted_public_keys=trusted_keys,
        )
        if not verification.valid:
            raise RunArtifactTransportError("release attestation publisher is not trusted")
        return verification

    @staticmethod
    def _validated_partial_prefix(
        partial: Path,
        manifest: RunArtifactTransportManifest,
    ) -> tuple[int, int]:
        offset = 0
        resumed = 0
        with RunArtifactTransport._open_partial(partial) as source:
            current_size = os.fstat(source.fileno()).st_size
            for chunk in manifest.chunks:
                end = offset + chunk.size_bytes
                if current_size < end:
                    break
                source.seek(offset)
                payload = source.read(chunk.size_bytes)
                if hashlib.sha256(payload).hexdigest() != chunk.content_digest:
                    break
                offset = end
                resumed += 1
            if current_size != offset:
                source.truncate(offset)
                source.flush()
                os.fsync(source.fileno())
        return resumed, offset

    @staticmethod
    def _open_partial(path: Path) -> BinaryIO:
        descriptor = os.open(
            path,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RunArtifactTransportError("archive download partial path must be a regular file")
        return os.fdopen(descriptor, "r+b")

    @classmethod
    def _publish_local_sidecars(
        cls,
        output: Path,
        manifest: RunArtifactTransportManifest,
        attestation_payload: bytes | None,
    ) -> None:
        local = LocalBlobStore(output.parent)
        checksum_path = RunArtifactArchive.checksum_path(output)
        checksum_payload = f"{manifest.receipt.content_digest}\n".encode("ascii")
        local.put_if_absent(checksum_path.name, checksum_payload)
        checksum_path.chmod(0o644)
        if attestation_payload is not None:
            attestation_path = cls.fetched_attestation_path(output)
            local.put_if_absent(attestation_path.name, attestation_payload)
            attestation_path.chmod(0o644)

    @staticmethod
    def _publish_local_archive(partial: Path, output: Path, digest: str) -> None:
        try:
            os.link(partial, output)
        except FileExistsError:
            try:
                receipt = RunArtifactArchive.inspect(output, expected_sha256=digest)
            except RunArtifactArchiveError as error:
                raise BlobConflictError("archive output appeared with different content") from error
            if receipt.content_digest != digest:
                raise BlobConflictError("archive output appeared with different content") from None
        output.chmod(0o644)

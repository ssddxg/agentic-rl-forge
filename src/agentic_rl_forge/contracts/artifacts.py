from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import ClassVar, Literal

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel
from agentic_rl_forge.contracts.blobs import BlobInfo
from agentic_rl_forge.contracts.datasets import ManifestSignature
from agentic_rl_forge.contracts.shards import ShardVerification


class RunArtifactRef(ContractModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    relative_path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> RunArtifactRef:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ValueError("run artifact path must stay within the collection root")
        return self


class RunArtifactManifest(ContractModel):
    REQUIRED_NAMES: ClassVar[tuple[str, ...]] = (
        "rollout_plan",
        "run_manifest",
        "trajectories",
        "benchmark_report",
        "metrics",
        "summary",
        "shard_manifest",
    )

    schema_version: int = Field(default=1, ge=1)
    manifest_id: str = Field(pattern=r"^run_artifacts_[0-9a-f]{24}$")
    run_id: str = Field(min_length=1)
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{24}$")
    shard_manifest_id: str = Field(pattern=r"^manifest_[0-9a-f]{24}$")
    created_at: datetime
    artifacts: tuple[RunArtifactRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> RunArtifactManifest:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("run artifact manifest time must be timezone-aware")
        names = tuple(item.name for item in self.artifacts)
        if names != self.REQUIRED_NAMES:
            raise ValueError("run artifact manifest has an invalid artifact set or order")
        paths = tuple(item.relative_path for item in self.artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("run artifact paths must be unique")
        expected_paths = {
            "rollout_plan": f"plans/{self.plan_id}.json",
            "run_manifest": f"runs/{self.run_id}/run-manifest.json",
            "trajectories": f"runs/{self.run_id}/trajectories.jsonl",
            "benchmark_report": f"runs/{self.run_id}/benchmark-report.json",
            "metrics": f"runs/{self.run_id}/metrics.prom",
            "summary": f"runs/{self.run_id}/summary.json",
            "shard_manifest": (
                f"shards/runs/{self.run_id}/manifests/{self.shard_manifest_id}.json"
            ),
        }
        if any(item.relative_path != expected_paths[item.name] for item in self.artifacts):
            raise ValueError("run artifact path does not match manifest identities")
        if self.manifest_id != self.expected_manifest_id(
            run_id=self.run_id,
            plan_id=self.plan_id,
            shard_manifest_id=self.shard_manifest_id,
            created_at=self.created_at,
            artifacts=self.artifacts,
        ):
            raise ValueError("run artifact manifest ID does not match its contents")
        return self

    @staticmethod
    def expected_manifest_id(
        *,
        run_id: str,
        plan_id: str,
        shard_manifest_id: str,
        created_at: datetime,
        artifacts: tuple[RunArtifactRef, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "run_id": run_id,
                "plan_id": plan_id,
                "shard_manifest_id": shard_manifest_id,
                "created_at": created_at.isoformat(),
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_artifacts_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactVerification(ContractModel):
    manifest_id: str
    valid: bool
    verified_artifact_count: int = Field(ge=0)
    missing_artifacts: tuple[str, ...] = ()
    mismatched_artifacts: tuple[str, ...] = ()
    semantic_errors: tuple[str, ...] = ()
    shard_verification: ShardVerification | None = None

    @model_validator(mode="after")
    def validate_result(self) -> RunArtifactVerification:
        expected_valid = (
            not self.missing_artifacts
            and not self.mismatched_artifacts
            and not self.semantic_errors
            and self.shard_verification is not None
            and self.shard_verification.valid
            and self.shard_verification.complete
        )
        if self.valid is not expected_valid:
            raise ValueError("run artifact verification validity does not match findings")
        return self


class RunArtifactArchiveReceipt(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    archive_id: str = Field(pattern=r"^run_archive_[0-9a-f]{24}$")
    run_id: str = Field(min_length=1)
    manifest_id: str = Field(pattern=r"^run_artifacts_[0-9a-f]{24}$")
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    member_count: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_archive_id(self) -> RunArtifactArchiveReceipt:
        expected = self.expected_archive_id(self.content_digest)
        if self.archive_id != expected:
            raise ValueError("run artifact archive ID does not match its content digest")
        return self

    @staticmethod
    def expected_archive_id(content_digest: str) -> str:
        return f"run_archive_{content_digest[:24]}"


class RunArtifactArchiveAttestation(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    attestation_id: str = Field(pattern=r"^run_attestation_[0-9a-f]{24}$")
    receipt: RunArtifactArchiveReceipt
    signer_key_id: str = Field(pattern=r"^ed25519_[0-9a-f]{24}$")
    signed_at: datetime

    @model_validator(mode="after")
    def validate_attestation(self) -> RunArtifactArchiveAttestation:
        if self.signed_at.tzinfo is None or self.signed_at.utcoffset() is None:
            raise ValueError("run artifact attestation time must be timezone-aware")
        expected = self.expected_attestation_id(
            receipt=self.receipt,
            signer_key_id=self.signer_key_id,
            signed_at=self.signed_at,
        )
        if self.attestation_id != expected:
            raise ValueError("run artifact attestation ID does not match its contents")
        return self

    @staticmethod
    def expected_attestation_id(
        *,
        receipt: RunArtifactArchiveReceipt,
        signer_key_id: str,
        signed_at: datetime,
    ) -> str:
        payload = orjson.dumps(
            {
                "receipt": receipt.model_dump(mode="json"),
                "signer_key_id": signer_key_id,
                "signed_at": signed_at.isoformat(),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_attestation_{hashlib.sha256(payload).hexdigest()[:24]}"


class SignedRunArtifactArchiveAttestation(ContractModel):
    attestation: RunArtifactArchiveAttestation
    signature: ManifestSignature

    @model_validator(mode="after")
    def validate_signature_identity(self) -> SignedRunArtifactArchiveAttestation:
        if self.signature.algorithm != "ed25519":
            raise ValueError("run artifact attestation requires Ed25519")
        try:
            public_key = base64.b64decode(
                self.signature.public_key_base64,
                validate=True,
            )
        except (ValueError, binascii.Error) as error:
            raise ValueError("run artifact attestation public key is invalid") from error
        if len(public_key) != 32:
            raise ValueError("run artifact attestation public key must contain 32 bytes")
        expected_key_id = f"ed25519_{hashlib.sha256(public_key).hexdigest()[:24]}"
        if self.attestation.signer_key_id != expected_key_id:
            raise ValueError("run artifact attestation signer key ID does not match its public key")
        payload_digest = hashlib.sha256(self.attestation.canonical_bytes()).hexdigest()
        if self.signature.payload_sha256 != payload_digest:
            raise ValueError("run artifact attestation payload digest does not match")
        return self


class RunArtifactArchiveAttestationVerification(ContractModel):
    attestation_id: str
    valid: bool
    archive_receipt_matches: bool
    signer_identity_valid: bool
    payload_digest_valid: bool
    signature_valid: bool
    trusted_signer: bool

    @model_validator(mode="after")
    def validate_result(self) -> RunArtifactArchiveAttestationVerification:
        expected = all(
            (
                self.archive_receipt_matches,
                self.signer_identity_valid,
                self.payload_digest_valid,
                self.signature_valid,
                self.trusted_signer,
            )
        )
        if self.valid is not expected:
            raise ValueError("run artifact attestation validity does not match findings")
        return self


class RunArtifactTransportObjectRef(ContractModel):
    key: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_key(self) -> RunArtifactTransportObjectRef:
        path = PurePosixPath(self.key)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ValueError("run artifact transport key must be a safe relative path")
        return self


class RunArtifactTransportChunkRef(RunArtifactTransportObjectRef):
    index: int = Field(ge=0)


class RunArtifactTransportManifest(ContractModel):
    ROOT: ClassVar[str] = "run-releases"

    schema_version: int = Field(default=1, ge=1)
    transport_id: str = Field(pattern=r"^run_transport_[0-9a-f]{24}$")
    receipt: RunArtifactArchiveReceipt
    chunk_size_bytes: int = Field(ge=1, le=536_870_912)
    chunks: tuple[RunArtifactTransportChunkRef, ...] = Field(min_length=1)
    checksum: RunArtifactTransportObjectRef
    attestation: RunArtifactTransportObjectRef | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> RunArtifactTransportManifest:
        if tuple(chunk.index for chunk in self.chunks) != tuple(range(len(self.chunks))):
            raise ValueError("run artifact transport chunk indexes must be contiguous")
        if sum(chunk.size_bytes for chunk in self.chunks) != self.receipt.size_bytes:
            raise ValueError("run artifact transport chunks do not match archive size")
        if any(chunk.size_bytes != self.chunk_size_bytes for chunk in self.chunks[:-1]):
            raise ValueError("non-final transport chunks must use the configured chunk size")
        if self.chunks[-1].size_bytes > self.chunk_size_bytes:
            raise ValueError("final transport chunk exceeds the configured chunk size")
        release_root = f"{self.ROOT}/{self.receipt.archive_id}"
        for chunk in self.chunks:
            expected = f"{release_root}/chunks/{chunk.index:08d}-{chunk.content_digest}.part"
            if chunk.key != expected or chunk.media_type != "application/octet-stream":
                raise ValueError("run artifact transport chunk metadata is not canonical")
        if (
            self.checksum.key != f"{release_root}/archive.sha256"
            or self.checksum.media_type != "text/plain"
            or self.checksum.size_bytes != 65
        ):
            raise ValueError("run artifact transport checksum metadata is not canonical")
        if self.attestation is not None and (
            self.attestation.key != f"{release_root}/attestation.json"
            or self.attestation.media_type != "application/json"
        ):
            raise ValueError("run artifact transport attestation metadata is not canonical")
        expected_id = self.expected_transport_id(
            receipt=self.receipt,
            chunk_size_bytes=self.chunk_size_bytes,
            chunks=self.chunks,
            checksum=self.checksum,
            attestation=self.attestation,
        )
        if self.transport_id != expected_id:
            raise ValueError("run artifact transport ID does not match its contents")
        return self

    @staticmethod
    def expected_transport_id(
        *,
        receipt: RunArtifactArchiveReceipt,
        chunk_size_bytes: int,
        chunks: tuple[RunArtifactTransportChunkRef, ...],
        checksum: RunArtifactTransportObjectRef,
        attestation: RunArtifactTransportObjectRef | None,
    ) -> str:
        payload = orjson.dumps(
            {
                "receipt": receipt.model_dump(mode="json"),
                "chunk_size_bytes": chunk_size_bytes,
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                "checksum": checksum.model_dump(mode="json"),
                "attestation": (
                    attestation.model_dump(mode="json") if attestation is not None else None
                ),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_transport_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactTransportCommit(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    commit_id: str = Field(pattern=r"^run_release_[0-9a-f]{24}$")
    archive_id: str = Field(pattern=r"^run_archive_[0-9a-f]{24}$")
    transport_id: str = Field(pattern=r"^run_transport_[0-9a-f]{24}$")
    manifest: RunArtifactTransportObjectRef

    @model_validator(mode="after")
    def validate_commit(self) -> RunArtifactTransportCommit:
        expected_key = (
            f"{RunArtifactTransportManifest.ROOT}/{self.archive_id}/"
            f"manifests/{self.transport_id}.json"
        )
        if self.manifest.key != expected_key or self.manifest.media_type != "application/json":
            raise ValueError("run artifact transport manifest reference is not canonical")
        expected_id = self.expected_commit_id(
            archive_id=self.archive_id,
            transport_id=self.transport_id,
            manifest=self.manifest,
        )
        if self.commit_id != expected_id:
            raise ValueError("run artifact release commit ID does not match its contents")
        return self

    @staticmethod
    def expected_commit_id(
        *,
        archive_id: str,
        transport_id: str,
        manifest: RunArtifactTransportObjectRef,
    ) -> str:
        payload = orjson.dumps(
            {
                "archive_id": archive_id,
                "transport_id": transport_id,
                "manifest": manifest.model_dump(mode="json"),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_release_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactPublicationResult(ContractModel):
    commit: RunArtifactTransportCommit
    created_chunk_count: int = Field(ge=0)
    reused_chunk_count: int = Field(ge=0)
    created_object_count: int = Field(ge=0)
    reused_object_count: int = Field(ge=0)
    commit_created: bool


class RunArtifactFetchResult(ContractModel):
    receipt: RunArtifactArchiveReceipt
    commit: RunArtifactTransportCommit
    downloaded_chunk_count: int = Field(ge=0)
    reused_local_chunk_count: int = Field(ge=0)
    downloaded_bytes: int = Field(ge=0)
    attestation_verification: RunArtifactArchiveAttestationVerification | None = None


class RunArtifactReleaseState(str, Enum):
    ABSENT = "absent"
    STAGED = "staged"
    COMMITTED = "committed"
    GC_IN_PROGRESS = "gc_in_progress"
    GARBAGE_COLLECTED = "garbage_collected"
    INVALID = "invalid"


class RunArtifactTransportStatus(ContractModel):
    archive_id: str = Field(pattern=r"^run_archive_[0-9a-f]{24}$")
    state: RunArtifactReleaseState
    observed_at: datetime
    object_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    oldest_modified_at: datetime | None = None
    newest_modified_at: datetime | None = None
    decision_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    commit_id: str | None = Field(default=None, pattern=r"^run_release_[0-9a-f]{24}$")
    gc_id: str | None = Field(default=None, pattern=r"^run_gc_[0-9a-f]{24}$")
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status(self) -> RunArtifactTransportStatus:
        for value in (
            self.observed_at,
            self.oldest_modified_at,
            self.newest_modified_at,
        ):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("run artifact transport status times must be timezone-aware")
        if self.object_count == 0 and (
            self.oldest_modified_at is not None or self.newest_modified_at is not None
        ):
            raise ValueError("empty transport status cannot contain modification times")
        if (
            self.object_count > 0
            and self.state is not RunArtifactReleaseState.INVALID
            and (self.oldest_modified_at is None or self.newest_modified_at is None)
        ):
            raise ValueError("non-empty transport status requires modification times")
        if (
            self.oldest_modified_at is not None
            and self.newest_modified_at is not None
            and self.oldest_modified_at > self.newest_modified_at
        ):
            raise ValueError("transport status modification times are reversed")
        return self


class RunArtifactGcReason(str, Enum):
    ELIGIBLE = "eligible"
    NOT_FOUND = "not_found"
    TOO_RECENT = "too_recent"
    COMMITTED = "committed"
    TOMBSTONED = "tombstoned"
    INVALID = "invalid"


class RunArtifactGcPreview(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    archive_id: str = Field(pattern=r"^run_archive_[0-9a-f]{24}$")
    eligible: bool
    reason: RunArtifactGcReason
    observed_at: datetime
    min_age_seconds: float = Field(ge=1)
    eligible_at: datetime | None = None
    objects: tuple[BlobInfo, ...] = ()
    object_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_preview(self) -> RunArtifactGcPreview:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("run artifact GC observation time must be timezone-aware")
        if self.eligible_at is not None and (
            self.eligible_at.tzinfo is None or self.eligible_at.utcoffset() is None
        ):
            raise ValueError("run artifact GC eligibility time must be timezone-aware")
        if self.object_count != len(self.objects):
            raise ValueError("run artifact GC object count does not match objects")
        if self.total_bytes != sum(item.size_bytes for item in self.objects):
            raise ValueError("run artifact GC byte count does not match objects")
        prefix = f"{RunArtifactTransportManifest.ROOT}/{self.archive_id}/"
        if any(not item.key.startswith(prefix) for item in self.objects):
            raise ValueError("run artifact GC object belongs to another release")
        if any(
            item.content_sha256 is None or item.etag is None or item.last_modified is None
            for item in self.objects
        ):
            raise ValueError("run artifact GC requires complete object identities")
        if self.eligible != (self.reason is RunArtifactGcReason.ELIGIBLE):
            raise ValueError("run artifact GC eligibility does not match its reason")
        if self.objects and self.eligible_at is None:
            raise ValueError("run artifact GC objects require an eligibility time")
        expected_digest = self.expected_state_digest(
            archive_id=self.archive_id,
            min_age_seconds=self.min_age_seconds,
            eligible_at=self.eligible_at,
            objects=self.objects,
        )
        if self.state_digest != expected_digest:
            raise ValueError("run artifact GC state digest does not match preview")
        return self

    @staticmethod
    def expected_state_digest(
        *,
        archive_id: str,
        min_age_seconds: float,
        eligible_at: datetime | None,
        objects: tuple[BlobInfo, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "archive_id": archive_id,
                "min_age_seconds": min_age_seconds,
                "eligible_at": eligible_at.isoformat() if eligible_at is not None else None,
                "objects": [item.model_dump(mode="json") for item in objects],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(payload).hexdigest()


class RunArtifactGcTombstone(ContractModel):
    kind: Literal["garbage_collection"] = "garbage_collection"
    schema_version: int = Field(default=1, ge=1)
    gc_id: str = Field(pattern=r"^run_gc_[0-9a-f]{24}$")
    preview: RunArtifactGcPreview
    operator: str = Field(min_length=1)
    reason: str = Field(min_length=8)
    created_at: datetime

    @model_validator(mode="after")
    def validate_tombstone(self) -> RunArtifactGcTombstone:
        if not self.preview.eligible:
            raise ValueError("run artifact GC tombstone requires an eligible preview")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("run artifact GC tombstone time must be timezone-aware")
        expected_id = self.expected_gc_id(
            state_digest=self.preview.state_digest,
            operator=self.operator,
            reason=self.reason,
            created_at=self.created_at,
        )
        if self.gc_id != expected_id:
            raise ValueError("run artifact GC ID does not match tombstone")
        return self

    @staticmethod
    def expected_gc_id(
        *,
        state_digest: str,
        operator: str,
        reason: str,
        created_at: datetime,
    ) -> str:
        payload = orjson.dumps(
            {
                "state_digest": state_digest,
                "operator": operator,
                "reason": reason,
                "created_at": created_at.isoformat(),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"run_gc_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunArtifactGcRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    gc_id: str = Field(pattern=r"^run_gc_[0-9a-f]{24}$")
    archive_id: str = Field(pattern=r"^run_archive_[0-9a-f]{24}$")
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator: str = Field(min_length=1)
    reason: str = Field(min_length=8)
    started_at: datetime
    completed_at: datetime
    target_object_count: int = Field(ge=1)
    deleted_object_count: int = Field(ge=0)
    already_missing_object_count: int = Field(ge=0)
    target_bytes: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_record(self) -> RunArtifactGcRecord:
        for value in (self.started_at, self.completed_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("run artifact GC record times must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("run artifact GC completion precedes start")
        if (
            self.deleted_object_count + self.already_missing_object_count
            != self.target_object_count
        ):
            raise ValueError("run artifact GC result counts do not cover all targets")
        return self

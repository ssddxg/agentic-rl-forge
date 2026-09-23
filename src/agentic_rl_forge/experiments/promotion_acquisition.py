from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import orjson
import yaml
from pydantic import Field, model_validator

from agentic_rl_forge.contracts import (
    CheckpointManifest,
    ContractModel,
    DatasetCollectionManifest,
    DatasetManifest,
    RolloutPlan,
    TrainerBatchManifest,
)
from agentic_rl_forge.data.manifests import Ed25519ManifestSigner
from agentic_rl_forge.evaluation import BenchmarkComparisonReport, BenchmarkReport
from agentic_rl_forge.experiments.models import ExperimentArtifactKind
from agentic_rl_forge.experiments.promotion import (
    ExperimentPromotionArtifactRef,
    ExperimentPromotionArtifactRole,
    ExperimentPromotionArtifactScope,
    ExperimentReproducibilityManifest,
)
from agentic_rl_forge.experiments.promotion_archives import (
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestationVerification,
    ExperimentPromotionArchiveAttestor,
    ExperimentPromotionArchiveReceipt,
    SignedExperimentPromotionArchiveAttestation,
)
from agentic_rl_forge.experiments.promotion_remote import (
    ExperimentPromotionRemoteArtifactPlan,
    ExperimentPromotionRemoteArtifactReceipt,
    ExperimentPromotionRemoteFetcher,
    ExperimentPromotionRemoteFetchRecord,
    remote_artifact_output_path,
)
from agentic_rl_forge.pipelines import SearchR1CollectionConfig
from agentic_rl_forge.storage.blobs import BlobConflictError, LocalBlobStore

_NATIVE_CONTRACT_LIMIT = 16 * 1024 * 1024
_SIGNER_KEY_ID = re.compile(r"^ed25519_[0-9a-f]{24}$")


def _filesystem_path(path: Path) -> Path:
    """Return a Windows extended path without changing the portable record path."""
    if os.name != "nt":
        return path
    value = os.path.abspath(os.fspath(path))
    if value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{value[2:]}")
    return Path(f"\\\\?\\{value}")


class ExperimentPromotionAcquisitionError(ValueError):
    pass


@dataclass(frozen=True)
class _RemoteCacheArtifact:
    record: ExperimentPromotionRemoteFetchRecord
    receipt: ExperimentPromotionRemoteArtifactReceipt
    root: Path


class ExperimentPromotionArtifactAvailability(str, Enum):
    VERIFIED = "verified"
    MISSING = "missing"
    MISMATCHED = "mismatched"
    REMOTE = "remote"
    UNSAFE = "unsafe"


class ExperimentPromotionNativeStatus(str, Enum):
    VERIFIED = "verified"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class ExperimentPromotionAcquisitionPolicy(ContractModel):
    allow_unresolved_remote: bool = False
    require_native_contracts: bool = True
    require_checkpoint_payloads: bool = True
    require_checkpoint_ancestry: bool = True
    require_dataset_lineage: bool = True
    max_materialized_bytes: int = Field(default=1 << 40, ge=1)


class ExperimentPromotionAcquisitionArtifact(ContractModel):
    reference: ExperimentPromotionArtifactRef
    output_path: str | None = None
    availability: ExperimentPromotionArtifactAvailability
    actual_size_bytes: int | None = Field(default=None, ge=0)
    actual_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    native_status: ExperimentPromotionNativeStatus
    native_content_id: str | None = None
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_resolution(self) -> ExperimentPromotionAcquisitionArtifact:
        reference = self.reference
        if reference.scope is ExperimentPromotionArtifactScope.REMOTE:
            if self.availability is ExperimentPromotionArtifactAvailability.REMOTE:
                if (
                    self.output_path is not None
                    or self.actual_size_bytes is not None
                    or self.actual_sha256 is not None
                    or self.native_status is not ExperimentPromotionNativeStatus.UNAVAILABLE
                    or self.native_content_id is not None
                ):
                    raise ValueError("unresolved remote acquisition artifact is inconsistent")
                return self
            expected_output = remote_artifact_output_path(reference)
        else:
            expected_output = f"{reference.scope.value}/{reference.locator}"
        if self.output_path != expected_output:
            raise ValueError("acquisition artifact output path is inconsistent")
        if (
            reference.scope is not ExperimentPromotionArtifactScope.REMOTE
            and self.availability is ExperimentPromotionArtifactAvailability.REMOTE
        ):
            raise ValueError("local acquisition artifact cannot have remote availability")
        if self.availability in {
            ExperimentPromotionArtifactAvailability.MISSING,
            ExperimentPromotionArtifactAvailability.UNSAFE,
        }:
            if self.actual_size_bytes is not None or self.actual_sha256 is not None:
                raise ValueError("unreadable acquisition artifact cannot have observed bytes")
        else:
            if self.actual_size_bytes is None or self.actual_sha256 is None:
                raise ValueError("read acquisition artifact requires observed bytes")
        if self.availability is ExperimentPromotionArtifactAvailability.VERIFIED:
            if (
                self.actual_size_bytes != reference.size_bytes
                or self.actual_sha256 != reference.sha256
            ):
                raise ValueError("verified acquisition artifact does not match its reference")
            if self.native_status is ExperimentPromotionNativeStatus.UNAVAILABLE:
                raise ValueError("verified acquisition artifact must have a native result")
        elif self.native_status is not ExperimentPromotionNativeStatus.UNAVAILABLE:
            raise ValueError("unverified acquisition artifact cannot have a native result")
        if self.native_status is ExperimentPromotionNativeStatus.VERIFIED:
            if self.native_content_id is None:
                raise ValueError("native verification requires a content ID")
        elif self.native_content_id is not None:
            raise ValueError("only verified native contracts can expose a content ID")
        return self


class ExperimentPromotionMaterializedFile(ContractModel):
    scope: ExperimentPromotionArtifactScope
    locator: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_file(self) -> ExperimentPromotionMaterializedFile:
        if self.scope is ExperimentPromotionArtifactScope.REMOTE:
            parsed_reference = ExperimentPromotionArtifactRef(
                scope=self.scope,
                locator=self.locator,
                kind=ExperimentArtifactKind.OTHER,
                size_bytes=self.size_bytes,
                sha256=self.sha256,
                roles=(ExperimentPromotionArtifactRole.OUTPUT,),
            )
            expected_output = remote_artifact_output_path(parsed_reference)
        else:
            path = PurePosixPath(self.locator)
            if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
                raise ValueError("materialized artifact locator must be safe and relative")
            expected_output = f"{self.scope.value}/{self.locator}"
        if self.output_path != expected_output:
            raise ValueError("materialized artifact output path is inconsistent")
        return self


class ExperimentPromotionAcquisitionCheck(ContractModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    passed: bool
    detail: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_check(self) -> ExperimentPromotionAcquisitionCheck:
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise ValueError("acquisition check evidence must be sorted and unique")
        return self


class ExperimentPromotionAcquisitionPlan(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    plan_id: str = Field(pattern=r"^promotion_acquisition_plan_[0-9a-f]{24}$")
    receipt: ExperimentPromotionArchiveReceipt
    attestation: SignedExperimentPromotionArchiveAttestation
    attestation_verification: ExperimentPromotionArchiveAttestationVerification
    trusted_signer_key_ids: tuple[str, ...]
    manifest: ExperimentReproducibilityManifest
    destination_root: str = Field(min_length=1)
    policy: ExperimentPromotionAcquisitionPolicy
    artifacts: tuple[ExperimentPromotionAcquisitionArtifact, ...]
    materialized_files: tuple[ExperimentPromotionMaterializedFile, ...]
    materialization_issues: tuple[str, ...] = ()
    checkpoint_payload_issues: tuple[str, ...] = ()
    checkpoint_ancestry_issues: tuple[str, ...] = ()
    dataset_lineage_issues: tuple[str, ...] = ()
    verified_artifact_count: int = Field(ge=0)
    unresolved_remote_count: int = Field(ge=0)
    materialized_file_count: int = Field(ge=0)
    materialized_size_bytes: int = Field(ge=0)
    checks: tuple[ExperimentPromotionAcquisitionCheck, ...]
    failed_check_count: int = Field(ge=0)
    eligible: bool

    @model_validator(mode="after")
    def validate_plan(self) -> ExperimentPromotionAcquisitionPlan:
        destination = Path(self.destination_root)
        if not destination.is_absolute() or destination.as_posix() != self.destination_root:
            raise ValueError("acquisition destination root must be a normalized absolute path")
        _validate_trust_evidence(
            self.receipt,
            self.attestation,
            self.attestation_verification,
            self.trusted_signer_key_ids,
        )
        if (
            self.receipt.promotion_name != self.manifest.promotion_name
            or self.receipt.reproducibility_manifest_id != self.manifest.manifest_id
        ):
            raise ValueError("acquisition archive receipt does not match its manifest")
        reference_keys = tuple(
            (
                item.reference.scope.value,
                item.reference.locator,
                item.reference.kind.value,
            )
            for item in self.artifacts
        )
        if reference_keys != tuple(sorted(set(reference_keys))):
            raise ValueError("acquisition artifacts must be sorted and unique")
        if tuple(item.reference for item in self.artifacts) != self.manifest.artifacts:
            raise ValueError("acquisition artifacts do not cover the promotion manifest")
        expected_files, expected_materialization_issues = _materialization(self.artifacts)
        if self.materialized_files != expected_files:
            raise ValueError("acquisition materialized files are inconsistent")
        if self.materialization_issues != expected_materialization_issues:
            raise ValueError("acquisition materialization issues are inconsistent")
        for issues in (
            self.checkpoint_payload_issues,
            self.checkpoint_ancestry_issues,
            self.dataset_lineage_issues,
        ):
            if issues != tuple(sorted(set(issues))):
                raise ValueError("acquisition issues must be sorted and unique")
        verified_count = sum(
            item.availability is ExperimentPromotionArtifactAvailability.VERIFIED
            for item in self.artifacts
        )
        remote_count = sum(
            item.availability is ExperimentPromotionArtifactAvailability.REMOTE
            for item in self.artifacts
        )
        materialized_size = sum(item.size_bytes for item in self.materialized_files)
        if (
            self.verified_artifact_count != verified_count
            or self.unresolved_remote_count != remote_count
            or self.materialized_file_count != len(self.materialized_files)
            or self.materialized_size_bytes != materialized_size
        ):
            raise ValueError("acquisition plan counts are inconsistent")
        expected_checks = self.expected_checks(
            verification=self.attestation_verification,
            policy=self.policy,
            artifacts=self.artifacts,
            materialization_issues=self.materialization_issues,
            checkpoint_payload_issues=self.checkpoint_payload_issues,
            checkpoint_ancestry_issues=self.checkpoint_ancestry_issues,
            dataset_lineage_issues=self.dataset_lineage_issues,
            materialized_size_bytes=self.materialized_size_bytes,
        )
        if self.checks != expected_checks:
            raise ValueError("acquisition plan checks do not match its evidence")
        failed = sum(not item.passed for item in self.checks)
        if self.failed_check_count != failed or self.eligible != (failed == 0):
            raise ValueError("acquisition plan eligibility is inconsistent")
        expected_id = self.expected_plan_id(
            receipt=self.receipt,
            attestation=self.attestation,
            trusted_signer_key_ids=self.trusted_signer_key_ids,
            manifest=self.manifest,
            destination_root=self.destination_root,
            policy=self.policy,
            artifacts=self.artifacts,
            materialized_files=self.materialized_files,
            materialization_issues=self.materialization_issues,
            checkpoint_payload_issues=self.checkpoint_payload_issues,
            checkpoint_ancestry_issues=self.checkpoint_ancestry_issues,
            dataset_lineage_issues=self.dataset_lineage_issues,
            checks=self.checks,
        )
        if self.plan_id != expected_id:
            raise ValueError("acquisition plan ID does not match its contents")
        return self

    @staticmethod
    def expected_checks(
        *,
        verification: ExperimentPromotionArchiveAttestationVerification,
        policy: ExperimentPromotionAcquisitionPolicy,
        artifacts: tuple[ExperimentPromotionAcquisitionArtifact, ...],
        materialization_issues: tuple[str, ...],
        checkpoint_payload_issues: tuple[str, ...],
        checkpoint_ancestry_issues: tuple[str, ...],
        dataset_lineage_issues: tuple[str, ...],
        materialized_size_bytes: int,
    ) -> tuple[ExperimentPromotionAcquisitionCheck, ...]:
        local_failures = tuple(
            sorted(
                f"{item.reference.scope.value}:{item.reference.locator}:{item.availability.value}"
                for item in artifacts
                if item.availability is not ExperimentPromotionArtifactAvailability.VERIFIED
                and not (
                    item.reference.scope is ExperimentPromotionArtifactScope.REMOTE
                    and item.availability is ExperimentPromotionArtifactAvailability.REMOTE
                )
            )
        )
        remotes = tuple(
            sorted(
                item.reference.locator
                for item in artifacts
                if item.availability is ExperimentPromotionArtifactAvailability.REMOTE
            )
        )
        invalid_native = tuple(
            sorted(
                f"{item.reference.scope.value}:{item.reference.locator}"
                for item in artifacts
                if item.native_status is ExperimentPromotionNativeStatus.INVALID
            )
        )
        checks = (
            ExperimentPromotionAcquisitionCheck(
                code="archive.signature",
                passed=(
                    verification.archive_receipt_matches
                    and verification.signer_identity_valid
                    and verification.payload_digest_valid
                    and verification.signature_valid
                ),
                detail="publisher attestation cryptographically matches the archive receipt",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionAcquisitionCheck(
                code="archive.trusted",
                passed=verification.trusted_signer,
                detail="publisher key belongs to the supplied acquisition trust set",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionAcquisitionCheck(
                code="artifacts.local",
                passed=not local_failures,
                detail="every materializable manifest reference matches verified source bytes",
                evidence=local_failures,
            ),
            ExperimentPromotionAcquisitionCheck(
                code="artifacts.remote",
                passed=policy.allow_unresolved_remote or not remotes,
                detail="unresolved remote references satisfy the explicit acquisition policy",
                evidence=remotes,
            ),
            ExperimentPromotionAcquisitionCheck(
                code="checkpoint.ancestry",
                passed=(not policy.require_checkpoint_ancestry or not checkpoint_ancestry_issues),
                detail="checkpoint parent links form a complete valid ancestry graph",
                evidence=checkpoint_ancestry_issues,
            ),
            ExperimentPromotionAcquisitionCheck(
                code="checkpoint.payloads",
                passed=(not policy.require_checkpoint_payloads or not checkpoint_payload_issues),
                detail="checkpoint manifests resolve every declared payload by name and digest",
                evidence=checkpoint_payload_issues,
            ),
            ExperimentPromotionAcquisitionCheck(
                code="dataset.lineage",
                passed=not policy.require_dataset_lineage or not dataset_lineage_issues,
                detail="checkpoint dataset lineage is present in the acquired artifact graph",
                evidence=dataset_lineage_issues,
            ),
            ExperimentPromotionAcquisitionCheck(
                code="materialization.bytes",
                passed=materialized_size_bytes <= policy.max_materialized_bytes,
                detail="verified local bytes fit within the materialization limit",
                evidence=tuple(
                    sorted(
                        {
                            str(materialized_size_bytes),
                            str(policy.max_materialized_bytes),
                        }
                    )
                ),
            ),
            ExperimentPromotionAcquisitionCheck(
                code="materialization.paths",
                passed=not materialization_issues,
                detail="materialized output paths have one consistent byte identity",
                evidence=materialization_issues,
            ),
            ExperimentPromotionAcquisitionCheck(
                code="native.contracts",
                passed=not policy.require_native_contracts or not invalid_native,
                detail="recognized artifact types satisfy their native canonical contracts",
                evidence=invalid_native,
            ),
        )
        return tuple(sorted(checks, key=lambda item: item.code))

    @staticmethod
    def expected_plan_id(
        *,
        receipt: ExperimentPromotionArchiveReceipt,
        attestation: SignedExperimentPromotionArchiveAttestation,
        trusted_signer_key_ids: tuple[str, ...],
        manifest: ExperimentReproducibilityManifest,
        destination_root: str,
        policy: ExperimentPromotionAcquisitionPolicy,
        artifacts: tuple[ExperimentPromotionAcquisitionArtifact, ...],
        materialized_files: tuple[ExperimentPromotionMaterializedFile, ...],
        materialization_issues: tuple[str, ...],
        checkpoint_payload_issues: tuple[str, ...],
        checkpoint_ancestry_issues: tuple[str, ...],
        dataset_lineage_issues: tuple[str, ...],
        checks: tuple[ExperimentPromotionAcquisitionCheck, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "receipt": receipt.model_dump(mode="json"),
                "attestation_id": attestation.attestation.attestation_id,
                "trusted_signer_key_ids": trusted_signer_key_ids,
                "manifest": manifest.model_dump(mode="json"),
                "destination_root": destination_root,
                "policy": policy.model_dump(mode="json"),
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "materialized_files": [item.model_dump(mode="json") for item in materialized_files],
                "materialization_issues": materialization_issues,
                "checkpoint_payload_issues": checkpoint_payload_issues,
                "checkpoint_ancestry_issues": checkpoint_ancestry_issues,
                "dataset_lineage_issues": dataset_lineage_issues,
                "checks": [item.model_dump(mode="json") for item in checks],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"promotion_acquisition_plan_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionAcquisitionRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    acquisition_id: str = Field(pattern=r"^promotion_acquisition_[0-9a-f]{24}$")
    plan: ExperimentPromotionAcquisitionPlan
    materialized_file_count: int = Field(ge=0)
    materialized_size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_record(self) -> ExperimentPromotionAcquisitionRecord:
        if not self.plan.eligible:
            raise ValueError("ineligible acquisition plan cannot become a record")
        expected = self.expected_acquisition_id(self.plan.plan_id)
        if self.acquisition_id != expected:
            raise ValueError("promotion acquisition ID does not match its plan")
        if (
            self.materialized_file_count != self.plan.materialized_file_count
            or self.materialized_size_bytes != self.plan.materialized_size_bytes
        ):
            raise ValueError("promotion acquisition record counts are inconsistent")
        return self

    @staticmethod
    def expected_acquisition_id(plan_id: str) -> str:
        digest = hashlib.sha256(plan_id.encode("ascii")).hexdigest()
        return f"promotion_acquisition_{digest[:24]}"


class ExperimentPromotionAcquirer:
    def __init__(
        self,
        project_root: Path | str,
        state_root: Path | str,
        *,
        remote_records: Collection[Path | str] = (),
    ) -> None:
        self.project_root = self._existing_root(project_root, "project")
        self.state_root = self._existing_root(state_root, "experiment state")
        self.remote_artifacts = self._load_remote_records(remote_records)
        self.remote_materializations = {
            (
                item.receipt.artifact.reference.locator,
                item.receipt.size_bytes,
                item.receipt.sha256,
            ): item
            for item in self.remote_artifacts.values()
        }

    def preview(
        self,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        destination_root: Path | str,
        *,
        trusted_public_keys: Collection[str],
        policy: ExperimentPromotionAcquisitionPolicy | None = None,
    ) -> ExperimentPromotionAcquisitionPlan:
        destination = self._destination_root(destination_root)
        receipt, record = ExperimentPromotionArchive.inspect_record(
            archive,
            expected_sha256=attestation.attestation.receipt.content_digest,
        )
        trusted_keys = tuple(sorted(set(trusted_public_keys)))
        trusted_ids = tuple(
            sorted(Ed25519ManifestSigner.public_key_id(item) for item in trusted_keys)
        )
        verification = ExperimentPromotionArchiveAttestor.verify(
            attestation,
            receipt,
            trusted_public_keys=trusted_keys,
        )
        manifest = record.preview.manifest
        resolutions: list[ExperimentPromotionAcquisitionArtifact] = []
        checkpoints: dict[tuple[ExperimentPromotionArtifactScope, str], CheckpointManifest] = {}
        for reference in manifest.artifacts:
            resolution, native_model = self._inspect_reference(reference)
            resolutions.append(resolution)
            if isinstance(native_model, CheckpointManifest):
                checkpoints[(reference.scope, reference.locator)] = native_model
        artifacts = tuple(resolutions)
        materialized_files, materialization_issues = _materialization(artifacts)
        payload_issues, ancestry_issues, lineage_issues = self._checkpoint_issues(
            artifacts,
            checkpoints,
        )
        resolved_policy = policy or ExperimentPromotionAcquisitionPolicy()
        materialized_size = sum(item.size_bytes for item in materialized_files)
        checks = ExperimentPromotionAcquisitionPlan.expected_checks(
            verification=verification,
            policy=resolved_policy,
            artifacts=artifacts,
            materialization_issues=materialization_issues,
            checkpoint_payload_issues=payload_issues,
            checkpoint_ancestry_issues=ancestry_issues,
            dataset_lineage_issues=lineage_issues,
            materialized_size_bytes=materialized_size,
        )
        plan_id = ExperimentPromotionAcquisitionPlan.expected_plan_id(
            receipt=receipt,
            attestation=attestation,
            trusted_signer_key_ids=trusted_ids,
            manifest=manifest,
            destination_root=destination.as_posix(),
            policy=resolved_policy,
            artifacts=artifacts,
            materialized_files=materialized_files,
            materialization_issues=materialization_issues,
            checkpoint_payload_issues=payload_issues,
            checkpoint_ancestry_issues=ancestry_issues,
            dataset_lineage_issues=lineage_issues,
            checks=checks,
        )
        failed = sum(not item.passed for item in checks)
        return ExperimentPromotionAcquisitionPlan(
            plan_id=plan_id,
            receipt=receipt,
            attestation=attestation,
            attestation_verification=verification,
            trusted_signer_key_ids=trusted_ids,
            manifest=manifest,
            destination_root=destination.as_posix(),
            policy=resolved_policy,
            artifacts=artifacts,
            materialized_files=materialized_files,
            materialization_issues=materialization_issues,
            checkpoint_payload_issues=payload_issues,
            checkpoint_ancestry_issues=ancestry_issues,
            dataset_lineage_issues=lineage_issues,
            verified_artifact_count=sum(
                item.availability is ExperimentPromotionArtifactAvailability.VERIFIED
                for item in artifacts
            ),
            unresolved_remote_count=sum(
                item.availability is ExperimentPromotionArtifactAvailability.REMOTE
                for item in artifacts
            ),
            materialized_file_count=len(materialized_files),
            materialized_size_bytes=materialized_size,
            checks=checks,
            failed_check_count=failed,
            eligible=failed == 0,
        )

    def execute(
        self,
        plan: ExperimentPromotionAcquisitionPlan,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        *,
        trusted_public_keys: Collection[str],
        confirm_plan_id: str,
    ) -> ExperimentPromotionAcquisitionRecord:
        if confirm_plan_id.strip() != plan.plan_id:
            raise ExperimentPromotionAcquisitionError(
                "acquisition confirmation does not match the plan ID"
            )
        current = self.preview(
            archive,
            attestation,
            plan.destination_root,
            trusted_public_keys=trusted_public_keys,
            policy=plan.policy,
        )
        if current != plan:
            raise ExperimentPromotionAcquisitionError(
                "promotion acquisition evidence changed after preview"
            )
        if not plan.eligible:
            raise ExperimentPromotionAcquisitionError("promotion acquisition plan is not eligible")
        record = ExperimentPromotionAcquisitionRecord(
            acquisition_id=ExperimentPromotionAcquisitionRecord.expected_acquisition_id(
                plan.plan_id
            ),
            plan=plan,
            materialized_file_count=plan.materialized_file_count,
            materialized_size_bytes=plan.materialized_size_bytes,
        )
        destination = self._prepare_destination(plan.destination_root)
        prefix = destination / "acquisitions" / record.acquisition_id
        expected_paths = {item.output_path for item in plan.materialized_files}
        expected_paths.add("record.json")
        self._validate_existing_prefix(prefix, expected_paths, record)
        for item in plan.materialized_files:
            self._publish_file(destination, record.acquisition_id, item)
        record_path = prefix / "record.json"
        try:
            LocalBlobStore(record_path.parent).put_if_absent(
                record_path.name,
                record.canonical_bytes() + b"\n",
            )
        except BlobConflictError as error:
            raise ExperimentPromotionAcquisitionError(
                "promotion acquisition record conflicts with existing evidence"
            ) from error
        record_path.chmod(0o644)
        self._verify_materialization(prefix, record)
        return record

    @staticmethod
    def _existing_root(value: Path | str, label: str) -> Path:
        path = Path(value)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"promotion acquisition {label} root must be a regular directory")
        return path.resolve(strict=True)

    def _destination_root(self, value: Path | str) -> Path:
        path = Path(value)
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError("promotion acquisition destination root must be a regular directory")
        destination = path.resolve(strict=False)
        remote_roots = {item.root for item in self.remote_artifacts.values()}
        for source in (self.project_root, self.state_root, *sorted(remote_roots)):
            if (
                destination == source
                or destination.is_relative_to(source)
                or source.is_relative_to(destination)
            ):
                raise ValueError("promotion acquisition destination must be separate from sources")
        return destination

    def _prepare_destination(self, value: str) -> Path:
        destination = self._destination_root(value)
        destination.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or not destination.is_dir():
            raise ExperimentPromotionAcquisitionError(
                "promotion acquisition destination is not a regular directory"
            )
        if destination.resolve(strict=True).as_posix() != value:
            raise ExperimentPromotionAcquisitionError(
                "promotion acquisition destination identity changed after preview"
            )
        return destination

    def _inspect_reference(
        self,
        reference: ExperimentPromotionArtifactRef,
    ) -> tuple[ExperimentPromotionAcquisitionArtifact, ContractModel | None]:
        if reference.scope is ExperimentPromotionArtifactScope.REMOTE:
            cached = self.remote_artifacts.get(
                ExperimentPromotionRemoteArtifactPlan.expected_artifact_id(reference)
            )
            if cached is not None:
                return self._inspect_remote_reference(reference, cached)
            return (
                ExperimentPromotionAcquisitionArtifact(
                    reference=reference,
                    availability=ExperimentPromotionArtifactAvailability.REMOTE,
                    native_status=ExperimentPromotionNativeStatus.UNAVAILABLE,
                    detail="remote reference requires an explicit external transport",
                ),
                None,
            )
        root = (
            self.project_root
            if reference.scope is ExperimentPromotionArtifactScope.PROJECT
            else self.state_root
        )
        path, path_error = self._source_path(root, reference.locator)
        output_path = f"{reference.scope.value}/{reference.locator}"
        if path_error == "missing":
            return (
                ExperimentPromotionAcquisitionArtifact(
                    reference=reference,
                    output_path=output_path,
                    availability=ExperimentPromotionArtifactAvailability.MISSING,
                    native_status=ExperimentPromotionNativeStatus.UNAVAILABLE,
                    detail="declared local artifact is missing",
                ),
                None,
            )
        if path_error is not None:
            return (
                ExperimentPromotionAcquisitionArtifact(
                    reference=reference,
                    output_path=output_path,
                    availability=ExperimentPromotionArtifactAvailability.UNSAFE,
                    native_status=ExperimentPromotionNativeStatus.UNAVAILABLE,
                    detail="declared local artifact path is unsafe",
                ),
                None,
            )
        actual_size, actual_digest, captured, changed = self._read_source(
            path,
            capture=reference.kind is not ExperimentArtifactKind.OTHER,
        )
        if changed or actual_size != reference.size_bytes or actual_digest != reference.sha256:
            return (
                ExperimentPromotionAcquisitionArtifact(
                    reference=reference,
                    output_path=output_path,
                    availability=ExperimentPromotionArtifactAvailability.MISMATCHED,
                    actual_size_bytes=actual_size,
                    actual_sha256=actual_digest,
                    native_status=ExperimentPromotionNativeStatus.UNAVAILABLE,
                    detail="declared local artifact bytes do not match the manifest",
                ),
                None,
            )
        native_status, native_id, native_detail, native_model = self._inspect_native(
            reference,
            captured,
        )
        return (
            ExperimentPromotionAcquisitionArtifact(
                reference=reference,
                output_path=output_path,
                availability=ExperimentPromotionArtifactAvailability.VERIFIED,
                actual_size_bytes=actual_size,
                actual_sha256=actual_digest,
                native_status=native_status,
                native_content_id=native_id,
                detail=native_detail,
            ),
            native_model,
        )

    def _inspect_remote_reference(
        self,
        reference: ExperimentPromotionArtifactRef,
        cached: _RemoteCacheArtifact,
    ) -> tuple[ExperimentPromotionAcquisitionArtifact, ContractModel | None]:
        output_path = remote_artifact_output_path(reference)
        if cached.receipt.artifact.reference != reference:
            return (
                ExperimentPromotionAcquisitionArtifact(
                    reference=reference,
                    output_path=output_path,
                    availability=ExperimentPromotionArtifactAvailability.MISMATCHED,
                    actual_size_bytes=cached.receipt.size_bytes,
                    actual_sha256=cached.receipt.sha256,
                    native_status=ExperimentPromotionNativeStatus.UNAVAILABLE,
                    detail="remote fetch evidence does not match the promotion reference",
                ),
                None,
            )
        try:
            actual_size, actual_digest, captured = self._read_remote_cache(
                cached,
                capture=reference.kind is not ExperimentArtifactKind.OTHER,
            )
        except ExperimentPromotionAcquisitionError:
            return (
                ExperimentPromotionAcquisitionArtifact(
                    reference=reference,
                    output_path=output_path,
                    availability=ExperimentPromotionArtifactAvailability.UNSAFE,
                    native_status=ExperimentPromotionNativeStatus.UNAVAILABLE,
                    detail="remote fetch cache is missing, unsafe, or inconsistent",
                ),
                None,
            )
        if actual_size != reference.size_bytes or actual_digest != reference.sha256:
            return (
                ExperimentPromotionAcquisitionArtifact(
                    reference=reference,
                    output_path=output_path,
                    availability=ExperimentPromotionArtifactAvailability.MISMATCHED,
                    actual_size_bytes=actual_size,
                    actual_sha256=actual_digest,
                    native_status=ExperimentPromotionNativeStatus.UNAVAILABLE,
                    detail="cached remote artifact bytes do not match the manifest",
                ),
                None,
            )
        native_status, native_id, native_detail, native_model = self._inspect_native(
            reference,
            captured,
        )
        return (
            ExperimentPromotionAcquisitionArtifact(
                reference=reference,
                output_path=output_path,
                availability=ExperimentPromotionArtifactAvailability.VERIFIED,
                actual_size_bytes=actual_size,
                actual_sha256=actual_digest,
                native_status=native_status,
                native_content_id=native_id,
                detail=f"{native_detail}; remote receipt {cached.receipt.receipt_id}",
            ),
            native_model,
        )

    @staticmethod
    def _source_path(root: Path, locator: str) -> tuple[Path, str | None]:
        candidate = root
        parts = PurePosixPath(locator).parts
        for index, part in enumerate(parts):
            candidate /= part
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                return candidate, "missing"
            except OSError:
                return candidate, "unsafe"
            if stat.S_ISLNK(info.st_mode):
                return candidate, "unsafe"
            if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
                return candidate, "unsafe"
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            return candidate, "unsafe"
        if not candidate.is_file():
            return candidate, "unsafe"
        return candidate, None

    @staticmethod
    def _read_source(path: Path, *, capture: bool) -> tuple[int, str, bytes, bool]:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ExperimentPromotionAcquisitionError(
                "cannot open a declared promotion artifact"
            ) from error
        digest = hashlib.sha256()
        captured = bytearray()
        size = 0
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ExperimentPromotionAcquisitionError(
                    "declared promotion artifact must be a regular file"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
                    if capture and len(captured) <= _NATIVE_CONTRACT_LIMIT:
                        captured.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        changed = (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        )
        return size, digest.hexdigest(), bytes(captured), changed

    @staticmethod
    def _inspect_native(
        reference: ExperimentPromotionArtifactRef,
        payload: bytes,
    ) -> tuple[
        ExperimentPromotionNativeStatus,
        str | None,
        str,
        ContractModel | None,
    ]:
        kind = reference.kind
        if kind is ExperimentArtifactKind.OTHER:
            return (
                ExperimentPromotionNativeStatus.NOT_APPLICABLE,
                None,
                "artifact kind has no native contract",
                None,
            )
        if len(payload) > _NATIVE_CONTRACT_LIMIT:
            return (
                ExperimentPromotionNativeStatus.INVALID,
                None,
                "native contract exceeds the acquisition safety limit",
                None,
            )
        try:
            model: ContractModel | None = None
            canonical_required = True
            if kind is ExperimentArtifactKind.COLLECTION_CONFIG:
                raw = yaml.safe_load(payload.decode("utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("collection config must contain an object")
                model = SearchR1CollectionConfig.model_validate(raw)
                native_id = model.plan_config_digest
                canonical_required = False
            elif kind is ExperimentArtifactKind.DATASET:
                native_id = reference.sha256
                canonical_required = False
            elif kind is ExperimentArtifactKind.DATASET_MANIFEST:
                model = DatasetManifest.model_validate_json(payload)
                native_id = model.manifest_id
            elif kind is ExperimentArtifactKind.DATASET_COLLECTION_MANIFEST:
                model = DatasetCollectionManifest.model_validate_json(payload)
                native_id = model.collection_id
            elif kind is ExperimentArtifactKind.ROLLOUT_PLAN:
                model = RolloutPlan.model_validate_json(payload)
                native_id = model.plan_id
            elif kind is ExperimentArtifactKind.CHECKPOINT_MANIFEST:
                model = CheckpointManifest.model_validate_json(payload)
                native_id = model.checkpoint_id
            elif kind is ExperimentArtifactKind.TRAINER_BATCH_MANIFEST:
                model = TrainerBatchManifest.model_validate_json(payload)
                native_id = model.batch_id
            elif kind is ExperimentArtifactKind.BENCHMARK_REPORT:
                model = BenchmarkReport.model_validate_json(payload)
                native_id = model.report_id
            elif kind is ExperimentArtifactKind.COMPARISON_REPORT:
                model = BenchmarkComparisonReport.model_validate_json(payload)
                native_id = model.comparison_id
            else:
                raise ValueError("unsupported native artifact kind")
            if (
                canonical_required
                and model is not None
                and payload != model.canonical_bytes() + b"\n"
            ):
                raise ValueError("native contract is not canonical")
            if reference.content_id != native_id:
                raise ValueError("native content ID does not match the manifest")
        except (UnicodeDecodeError, ValueError, yaml.YAMLError):
            return (
                ExperimentPromotionNativeStatus.INVALID,
                None,
                "artifact does not satisfy its native canonical contract",
                None,
            )
        return (
            ExperimentPromotionNativeStatus.VERIFIED,
            native_id,
            "artifact satisfies its native contract and content identity",
            model,
        )

    @staticmethod
    def _checkpoint_issues(
        artifacts: tuple[ExperimentPromotionAcquisitionArtifact, ...],
        checkpoints: dict[tuple[ExperimentPromotionArtifactScope, str], CheckpointManifest],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        payload_issues: set[str] = set()
        ancestry_issues: set[str] = set()
        lineage_issues: set[str] = set()
        by_id: dict[str, CheckpointManifest] = {}
        for checkpoint in checkpoints.values():
            existing = by_id.setdefault(checkpoint.checkpoint_id, checkpoint)
            if existing != checkpoint:
                ancestry_issues.add(f"duplicate_checkpoint:{checkpoint.checkpoint_id}")

        payload_refs = tuple(
            item
            for item in artifacts
            if ExperimentPromotionArtifactRole.CHECKPOINT_PAYLOAD in item.reference.roles
        )
        for checkpoint in checkpoints.values():
            for artifact in checkpoint.artifacts:
                matches = tuple(
                    item
                    for item in payload_refs
                    if item.reference.content_id == artifact.name
                    and item.reference.size_bytes == artifact.size_bytes
                    and item.reference.sha256 == artifact.sha256
                )
                label = f"{checkpoint.checkpoint_id}:{artifact.name}"
                if len(matches) != 1:
                    payload_issues.add(f"payload_reference_count:{label}:{len(matches)}")
                elif (
                    matches[0].availability is not ExperimentPromotionArtifactAvailability.VERIFIED
                ):
                    payload_issues.add(
                        f"payload_unverified:{label}:{matches[0].availability.value}"
                    )

            parent_id = checkpoint.parent_checkpoint_id
            if parent_id is None:
                continue
            parent = by_id.get(parent_id)
            if parent is None:
                ancestry_issues.add(f"missing_parent:{checkpoint.checkpoint_id}:{parent_id}")
                continue
            if parent.run_id != checkpoint.run_id:
                ancestry_issues.add(f"parent_run_mismatch:{checkpoint.checkpoint_id}:{parent_id}")
            if parent.step >= checkpoint.step:
                ancestry_issues.add(f"parent_step_invalid:{checkpoint.checkpoint_id}:{parent_id}")
            if parent.config_digest != checkpoint.config_digest:
                ancestry_issues.add(
                    f"parent_config_mismatch:{checkpoint.checkpoint_id}:{parent_id}"
                )
            if (
                parent.dataset_manifest_digest is not None
                and checkpoint.dataset_manifest_digest is not None
                and parent.dataset_manifest_digest != checkpoint.dataset_manifest_digest
            ):
                ancestry_issues.add(
                    f"parent_dataset_mismatch:{checkpoint.checkpoint_id}:{parent_id}"
                )

        for checkpoint_id in by_id:
            seen: set[str] = set()
            current = checkpoint_id
            while current in by_id and by_id[current].parent_checkpoint_id is not None:
                if current in seen:
                    ancestry_issues.add(f"checkpoint_cycle:{checkpoint_id}")
                    break
                seen.add(current)
                parent_link = by_id[current].parent_checkpoint_id
                if parent_link is None:
                    break
                current = parent_link

        lineage_digests = {
            value
            for item in artifacts
            if item.reference.kind
            in {
                ExperimentArtifactKind.DATASET,
                ExperimentArtifactKind.DATASET_MANIFEST,
                ExperimentArtifactKind.DATASET_COLLECTION_MANIFEST,
            }
            for value in (item.reference.sha256, item.reference.content_id)
            if value is not None and re.fullmatch(r"[0-9a-f]{64}", value)
        }
        for checkpoint in checkpoints.values():
            digest = checkpoint.dataset_manifest_digest
            if digest is None:
                lineage_issues.add(f"missing_dataset_lineage:{checkpoint.checkpoint_id}")
            elif digest not in lineage_digests:
                lineage_issues.add(
                    f"unresolved_dataset_lineage:{checkpoint.checkpoint_id}:{digest}"
                )
        return (
            tuple(sorted(payload_issues)),
            tuple(sorted(ancestry_issues)),
            tuple(sorted(lineage_issues)),
        )

    @staticmethod
    def _validate_existing_prefix(
        prefix: Path,
        expected_paths: set[str],
        record: ExperimentPromotionAcquisitionRecord,
    ) -> None:
        filesystem_prefix = _filesystem_path(prefix)
        if not filesystem_prefix.exists():
            return
        if filesystem_prefix.is_symlink() or not filesystem_prefix.is_dir():
            raise ExperimentPromotionAcquisitionError(
                "promotion acquisition prefix is not a regular directory"
            )
        for entry in filesystem_prefix.rglob("*"):
            if entry.is_symlink():
                raise ExperimentPromotionAcquisitionError(
                    "promotion acquisition prefix contains a symbolic link"
                )
            if entry.is_dir():
                continue
            relative = entry.relative_to(filesystem_prefix).as_posix()
            if relative not in expected_paths:
                raise ExperimentPromotionAcquisitionError(
                    f"promotion acquisition prefix contains unexpected file {relative!r}"
                )
        record_path = filesystem_prefix / "record.json"
        if record_path.exists():
            try:
                existing = ExperimentPromotionAcquisitionRecord.model_validate_json(
                    record_path.read_bytes()
                )
            except (OSError, ValueError) as error:
                raise ExperimentPromotionAcquisitionError(
                    "existing promotion acquisition record is invalid"
                ) from error
            if existing != record or record_path.read_bytes() != existing.canonical_bytes() + b"\n":
                raise ExperimentPromotionAcquisitionError(
                    "existing promotion acquisition record conflicts with the plan"
                )

    def _publish_file(
        self,
        destination: Path,
        acquisition_id: str,
        item: ExperimentPromotionMaterializedFile,
    ) -> None:
        if item.scope is ExperimentPromotionArtifactScope.REMOTE:
            self._publish_remote_file(destination, acquisition_id, item)
            return
        root = (
            self.project_root
            if item.scope is ExperimentPromotionArtifactScope.PROJECT
            else self.state_root
        )
        source, error = self._source_path(root, item.locator)
        if error is not None:
            raise ExperimentPromotionAcquisitionError(
                f"materialization source became {error}: {item.output_path}"
            )
        target = destination / "acquisitions" / acquisition_id / item.output_path
        self._ensure_safe_parent(destination, target.parent)
        filesystem_target = _filesystem_path(target)
        if filesystem_target.exists() or filesystem_target.is_symlink():
            self._verify_file(target, item.size_bytes, item.sha256)
            return
        staging = destination / "acquisitions" / acquisition_id
        descriptor, temporary_name = tempfile.mkstemp(
            dir=staging,
            prefix=".arf-acquire-",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            size = 0
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            source_descriptor: int | None = None
            try:
                source_descriptor = os.open(source, flags)
                before = os.fstat(source_descriptor)
                with (
                    os.fdopen(source_descriptor, "rb", closefd=False) as input_file,
                    os.fdopen(descriptor, "wb", closefd=False) as output_file,
                ):
                    while chunk := input_file.read(1024 * 1024):
                        size += len(chunk)
                        digest.update(chunk)
                        output_file.write(chunk)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                after = os.fstat(source_descriptor)
            finally:
                if source_descriptor is not None:
                    os.close(source_descriptor)
                os.close(descriptor)
            if (
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or size != item.size_bytes
                or digest.hexdigest() != item.sha256
            ):
                raise ExperimentPromotionAcquisitionError(
                    f"materialization source changed after preview: {item.output_path}"
                )
            temporary.chmod(0o644)
            try:
                os.link(_filesystem_path(temporary), filesystem_target)
            except FileExistsError:
                self._verify_file(target, item.size_bytes, item.sha256)
            filesystem_target.chmod(0o644)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _publish_remote_file(
        self,
        destination: Path,
        acquisition_id: str,
        item: ExperimentPromotionMaterializedFile,
    ) -> None:
        cached = self.remote_materializations.get((item.locator, item.size_bytes, item.sha256))
        if cached is None:
            raise ExperimentPromotionAcquisitionError(
                f"remote materialization evidence disappeared: {item.output_path}"
            )
        target = destination / "acquisitions" / acquisition_id / item.output_path
        self._ensure_safe_parent(destination, target.parent)
        filesystem_target = _filesystem_path(target)
        if filesystem_target.exists() or filesystem_target.is_symlink():
            self._verify_file(target, item.size_bytes, item.sha256)
            return
        staging = destination / "acquisitions" / acquisition_id
        descriptor, temporary_name = tempfile.mkstemp(
            dir=staging,
            prefix=".arf-acquire-",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                size, digest, _ = self._read_remote_cache(
                    cached,
                    capture=False,
                    output=output,
                )
                output.flush()
                os.fsync(output.fileno())
            if size != item.size_bytes or digest != item.sha256:
                raise ExperimentPromotionAcquisitionError(
                    f"remote materialization evidence changed: {item.output_path}"
                )
            temporary.chmod(0o644)
            try:
                os.link(_filesystem_path(temporary), filesystem_target)
            except FileExistsError:
                self._verify_file(target, item.size_bytes, item.sha256)
            filesystem_target.chmod(0o644)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_remote_cache(
        self,
        cached: _RemoteCacheArtifact,
        *,
        capture: bool,
        output: BinaryIO | None = None,
    ) -> tuple[int, str, bytes]:
        digest = hashlib.sha256()
        captured = bytearray()
        size = 0
        for chunk in cached.receipt.chunks:
            path = cached.root / chunk.key
            if path.is_symlink() or not path.is_file():
                raise ExperimentPromotionAcquisitionError(
                    "cached remote artifact chunk is not a regular file"
                )
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if (
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or len(data) != chunk.size_bytes
                or hashlib.sha256(data).hexdigest() != chunk.sha256
            ):
                raise ExperimentPromotionAcquisitionError(
                    "cached remote artifact chunk changed after verification"
                )
            size += len(data)
            digest.update(data)
            if output is not None:
                output.write(data)
            if capture and len(captured) <= _NATIVE_CONTRACT_LIMIT:
                captured.extend(data)
        actual_digest = digest.hexdigest()
        if size != cached.receipt.size_bytes or actual_digest != cached.receipt.sha256:
            raise ExperimentPromotionAcquisitionError(
                "cached remote artifact does not match its receipt"
            )
        return size, actual_digest, bytes(captured)

    @staticmethod
    def _load_remote_records(
        paths: Collection[Path | str],
    ) -> dict[str, _RemoteCacheArtifact]:
        artifacts: dict[str, _RemoteCacheArtifact] = {}
        for raw_path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
            record = ExperimentPromotionRemoteFetcher.verify_record(raw_path)
            root = Path(record.plan.cache_root) / "promotion-fetches" / record.fetch_id
            for receipt in record.artifact_receipts:
                artifact_id = receipt.artifact.artifact_id
                if artifact_id in artifacts:
                    raise ValueError(
                        f"multiple remote fetch records cover artifact {artifact_id!r}"
                    )
                artifacts[artifact_id] = _RemoteCacheArtifact(
                    record=record,
                    receipt=receipt,
                    root=root,
                )
        return artifacts

    @staticmethod
    def _ensure_safe_parent(destination: Path, parent: Path) -> None:
        relative = parent.relative_to(destination)
        current = destination
        for part in relative.parts:
            current /= part
            filesystem_current = _filesystem_path(current)
            try:
                filesystem_current.mkdir()
            except FileExistsError:
                if filesystem_current.is_symlink() or not filesystem_current.is_dir():
                    raise ExperimentPromotionAcquisitionError(
                        "promotion acquisition output parent is unsafe"
                    ) from None
            filesystem_current.chmod(0o755)

    @staticmethod
    def _verify_file(path: Path, size_bytes: int, sha256: str) -> None:
        filesystem_path = _filesystem_path(path)
        if filesystem_path.is_symlink() or not filesystem_path.is_file():
            raise ExperimentPromotionAcquisitionError(
                "materialized acquisition artifact is not a regular file"
            )
        digest = hashlib.sha256()
        size = 0
        with filesystem_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        if size != size_bytes or digest.hexdigest() != sha256:
            raise ExperimentPromotionAcquisitionError(
                "materialized acquisition artifact conflicts with the plan"
            )

    @classmethod
    def _verify_materialization(
        cls,
        prefix: Path,
        record: ExperimentPromotionAcquisitionRecord,
    ) -> None:
        expected = {item.output_path for item in record.plan.materialized_files}
        expected.add("record.json")
        cls._validate_existing_prefix(prefix, expected, record)
        for item in record.plan.materialized_files:
            cls._verify_file(prefix / item.output_path, item.size_bytes, item.sha256)


def _materialization(
    artifacts: tuple[ExperimentPromotionAcquisitionArtifact, ...],
) -> tuple[tuple[ExperimentPromotionMaterializedFile, ...], tuple[str, ...]]:
    files: dict[str, ExperimentPromotionMaterializedFile] = {}
    issues: set[str] = set()
    for artifact in artifacts:
        if (
            artifact.availability is not ExperimentPromotionArtifactAvailability.VERIFIED
            or artifact.output_path is None
        ):
            continue
        candidate = ExperimentPromotionMaterializedFile(
            scope=artifact.reference.scope,
            locator=artifact.reference.locator,
            output_path=artifact.output_path,
            size_bytes=artifact.reference.size_bytes,
            sha256=artifact.reference.sha256,
        )
        existing = files.setdefault(candidate.output_path, candidate)
        if existing != candidate:
            issues.add(f"conflicting_output:{candidate.output_path}")
    return (
        tuple(files[key] for key in sorted(files)),
        tuple(sorted(issues)),
    )


def _validate_trust_evidence(
    receipt: ExperimentPromotionArchiveReceipt,
    attestation: SignedExperimentPromotionArchiveAttestation,
    verification: ExperimentPromotionArchiveAttestationVerification,
    trusted_signer_key_ids: tuple[str, ...],
) -> None:
    if trusted_signer_key_ids != tuple(sorted(set(trusted_signer_key_ids))) or any(
        _SIGNER_KEY_ID.fullmatch(item) is None for item in trusted_signer_key_ids
    ):
        raise ValueError("trusted acquisition signer key IDs must be sorted and unique")
    if attestation.attestation.receipt != receipt:
        raise ValueError("promotion acquisition attestation receipt is inconsistent")
    embedded_key = attestation.signature.public_key_base64
    embedded_id = Ed25519ManifestSigner.public_key_id(embedded_key)
    derived = ExperimentPromotionArchiveAttestor.verify(
        attestation,
        receipt,
        trusted_public_keys=(embedded_key,) if embedded_id in trusted_signer_key_ids else (),
    )
    if verification != derived:
        raise ValueError("promotion acquisition attestation verification is inconsistent")


__all__ = [
    "ExperimentPromotionAcquirer",
    "ExperimentPromotionAcquisitionArtifact",
    "ExperimentPromotionAcquisitionCheck",
    "ExperimentPromotionAcquisitionError",
    "ExperimentPromotionAcquisitionPlan",
    "ExperimentPromotionAcquisitionPolicy",
    "ExperimentPromotionAcquisitionRecord",
    "ExperimentPromotionArtifactAvailability",
    "ExperimentPromotionMaterializedFile",
    "ExperimentPromotionNativeStatus",
]

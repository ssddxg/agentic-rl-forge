from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from enum import Enum
from pathlib import Path

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts import ContractModel
from agentic_rl_forge.data.manifests import Ed25519ManifestSigner
from agentic_rl_forge.experiments.promotion_archives import (
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestationVerification,
    ExperimentPromotionArchiveAttestor,
    ExperimentPromotionArchiveReceipt,
    SignedExperimentPromotionArchiveAttestation,
)
from agentic_rl_forge.storage.blobs import BlobConflictError, ConditionalBlobStore

_PROMOTION_ID = re.compile(r"^experiment_promotion_[0-9a-f]{24}$")
_ENVIRONMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_LIFECYCLE_KEY = re.compile(r"^promotions/(experiment_promotion_[0-9a-f]{24})/([0-9]{8})\.json$")
_ALIAS_KEY = re.compile(r"^environments/([A-Za-z][A-Za-z0-9_.-]{0,127})/([0-9]{8})\.json$")


class ExperimentPromotionRegistryError(ValueError):
    pass


class ExperimentPromotionStage(str, Enum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"


class ExperimentPromotionAliasAction(str, Enum):
    ASSIGN = "assign"
    ROLLBACK = "rollback"


class ExperimentPromotionGovernancePolicy(ContractModel):
    minimum_authorizers: int = Field(default=1, ge=1, le=16)
    operator_may_authorize: bool = False


class ExperimentPromotionAliasPolicy(ContractModel):
    allowed_stages: tuple[ExperimentPromotionStage, ...] = (ExperimentPromotionStage.PRODUCTION,)
    governance: ExperimentPromotionGovernancePolicy = Field(
        default_factory=ExperimentPromotionGovernancePolicy
    )

    @model_validator(mode="after")
    def validate_policy(self) -> ExperimentPromotionAliasPolicy:
        expected = tuple(sorted(set(self.allowed_stages), key=lambda item: item.value))
        if not expected or self.allowed_stages != expected:
            raise ValueError("alias policy stages must be nonempty, sorted, and unique")
        if ExperimentPromotionStage.RETIRED in self.allowed_stages:
            raise ValueError("retired promotions cannot satisfy an environment alias policy")
        return self


class ExperimentPromotionRegistryCheck(ContractModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    passed: bool
    detail: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_check(self) -> ExperimentPromotionRegistryCheck:
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise ValueError("promotion registry check evidence must be sorted and unique")
        return self


class ExperimentPromotionLifecyclePreview(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    preview_id: str = Field(pattern=r"^promotion_lifecycle_preview_[0-9a-f]{24}$")
    receipt: ExperimentPromotionArchiveReceipt
    attestation: SignedExperimentPromotionArchiveAttestation
    attestation_verification: ExperimentPromotionArchiveAttestationVerification
    trusted_signer_key_ids: tuple[str, ...]
    sequence: int = Field(ge=1)
    current_event_id: str | None = Field(
        default=None,
        pattern=r"^promotion_lifecycle_event_[0-9a-f]{24}$",
    )
    source_stage: ExperimentPromotionStage | None = None
    target_stage: ExperimentPromotionStage
    current_archive_id: str | None = Field(
        default=None,
        pattern=r"^experiment_promotion_archive_[0-9a-f]{24}$",
    )
    active_environments: tuple[str, ...] = ()
    operator: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=8, max_length=4096)
    authorizers: tuple[str, ...] = Field(min_length=1)
    policy: ExperimentPromotionGovernancePolicy
    checks: tuple[ExperimentPromotionRegistryCheck, ...]
    failed_check_count: int = Field(ge=0)
    eligible: bool

    @model_validator(mode="after")
    def validate_preview(self) -> ExperimentPromotionLifecyclePreview:
        _validate_governance(
            self.operator,
            self.reason,
            self.authorizers,
        )
        _validate_trust_evidence(
            self.receipt,
            self.attestation,
            self.attestation_verification,
            self.trusted_signer_key_ids,
        )
        if self.sequence == 1:
            if (
                self.current_event_id is not None
                or self.source_stage is not None
                or self.current_archive_id is not None
            ):
                raise ValueError("initial lifecycle preview cannot contain prior state")
        elif (
            self.current_event_id is None
            or self.source_stage is None
            or self.current_archive_id is None
        ):
            raise ValueError("continued lifecycle preview requires complete prior state")
        if self.active_environments != tuple(sorted(set(self.active_environments))):
            raise ValueError("active promotion environments must be sorted and unique")
        if any(_ENVIRONMENT.fullmatch(item) is None for item in self.active_environments):
            raise ValueError("active promotion environment is invalid")
        expected_checks = self.expected_checks(
            receipt=self.receipt,
            verification=self.attestation_verification,
            sequence=self.sequence,
            source_stage=self.source_stage,
            target_stage=self.target_stage,
            current_archive_id=self.current_archive_id,
            active_environments=self.active_environments,
            operator=self.operator,
            authorizers=self.authorizers,
            policy=self.policy,
        )
        if self.checks != expected_checks:
            raise ValueError("lifecycle preview checks do not match its evidence")
        failed = sum(not item.passed for item in self.checks)
        if self.failed_check_count != failed or self.eligible != (failed == 0):
            raise ValueError("lifecycle preview eligibility is inconsistent")
        expected_id = self.expected_preview_id(
            receipt=self.receipt,
            attestation=self.attestation,
            trusted_signer_key_ids=self.trusted_signer_key_ids,
            sequence=self.sequence,
            current_event_id=self.current_event_id,
            source_stage=self.source_stage,
            target_stage=self.target_stage,
            current_archive_id=self.current_archive_id,
            active_environments=self.active_environments,
            operator=self.operator,
            reason=self.reason,
            authorizers=self.authorizers,
            policy=self.policy,
            checks=self.checks,
        )
        if self.preview_id != expected_id:
            raise ValueError("lifecycle preview ID does not match its contents")
        return self

    @staticmethod
    def expected_checks(
        *,
        receipt: ExperimentPromotionArchiveReceipt,
        verification: ExperimentPromotionArchiveAttestationVerification,
        sequence: int,
        source_stage: ExperimentPromotionStage | None,
        target_stage: ExperimentPromotionStage,
        current_archive_id: str | None,
        active_environments: tuple[str, ...],
        operator: str,
        authorizers: tuple[str, ...],
        policy: ExperimentPromotionGovernancePolicy,
    ) -> tuple[ExperimentPromotionRegistryCheck, ...]:
        allowed = _allowed_lifecycle_targets(source_stage)
        values = (
            ExperimentPromotionRegistryCheck(
                code="archive.continuity",
                passed=sequence == 1 or current_archive_id == receipt.archive_id,
                detail="archive identity matches prior lifecycle evidence",
                evidence=tuple(
                    sorted(
                        {
                            item
                            for item in (current_archive_id, receipt.archive_id)
                            if item is not None
                        }
                    )
                ),
            ),
            ExperimentPromotionRegistryCheck(
                code="attestation.signature",
                passed=(
                    verification.archive_receipt_matches
                    and verification.signer_identity_valid
                    and verification.payload_digest_valid
                    and verification.signature_valid
                ),
                detail="publisher attestation cryptographically matches the archive receipt",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionRegistryCheck(
                code="attestation.trusted",
                passed=verification.trusted_signer,
                detail="publisher key belongs to the supplied transition trust set",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionRegistryCheck(
                code="authorization.independent",
                passed=policy.operator_may_authorize or operator not in authorizers,
                detail="deployment authorization is independent from the operator",
                evidence=authorizers,
            ),
            ExperimentPromotionRegistryCheck(
                code="authorization.quorum",
                passed=len(authorizers) >= policy.minimum_authorizers,
                detail="deployment authorization satisfies the configured quorum",
                evidence=authorizers,
            ),
            ExperimentPromotionRegistryCheck(
                code="retirement.detached",
                passed=(
                    target_stage is not ExperimentPromotionStage.RETIRED or not active_environments
                ),
                detail="retirement does not leave an active environment alias",
                evidence=active_environments,
            ),
            ExperimentPromotionRegistryCheck(
                code="transition.allowed",
                passed=target_stage in allowed,
                detail="target stage is reachable from the current lifecycle stage",
                evidence=tuple(
                    sorted(
                        [target_stage.value]
                        + ([] if source_stage is None else [source_stage.value])
                    )
                ),
            ),
        )
        return tuple(sorted(values, key=lambda item: item.code))

    @staticmethod
    def expected_preview_id(
        *,
        receipt: ExperimentPromotionArchiveReceipt,
        attestation: SignedExperimentPromotionArchiveAttestation,
        trusted_signer_key_ids: tuple[str, ...],
        sequence: int,
        current_event_id: str | None,
        source_stage: ExperimentPromotionStage | None,
        target_stage: ExperimentPromotionStage,
        current_archive_id: str | None,
        active_environments: tuple[str, ...],
        operator: str,
        reason: str,
        authorizers: tuple[str, ...],
        policy: ExperimentPromotionGovernancePolicy,
        checks: tuple[ExperimentPromotionRegistryCheck, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "receipt": receipt.model_dump(mode="json"),
                "attestation_id": attestation.attestation.attestation_id,
                "trusted_signer_key_ids": trusted_signer_key_ids,
                "sequence": sequence,
                "current_event_id": current_event_id,
                "source_stage": source_stage.value if source_stage is not None else None,
                "target_stage": target_stage.value,
                "current_archive_id": current_archive_id,
                "active_environments": active_environments,
                "operator": operator,
                "reason": reason,
                "authorizers": authorizers,
                "policy": policy.model_dump(mode="json"),
                "checks": [item.model_dump(mode="json") for item in checks],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"promotion_lifecycle_preview_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionLifecycleEvent(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    event_id: str = Field(pattern=r"^promotion_lifecycle_event_[0-9a-f]{24}$")
    preview: ExperimentPromotionLifecyclePreview

    @model_validator(mode="after")
    def validate_event(self) -> ExperimentPromotionLifecycleEvent:
        if not self.preview.eligible:
            raise ValueError("ineligible lifecycle preview cannot become an event")
        expected = self.expected_event_id(self.preview.preview_id)
        if self.event_id != expected:
            raise ValueError("lifecycle event ID does not match its preview")
        return self

    @staticmethod
    def expected_event_id(preview_id: str) -> str:
        digest = hashlib.sha256(preview_id.encode("ascii")).hexdigest()
        return f"promotion_lifecycle_event_{digest[:24]}"


class ExperimentPromotionAliasPreview(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    preview_id: str = Field(pattern=r"^promotion_alias_preview_[0-9a-f]{24}$")
    environment: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    generation: int = Field(ge=1)
    current_event_id: str | None = Field(
        default=None,
        pattern=r"^promotion_alias_event_[0-9a-f]{24}$",
    )
    current_promotion_id: str | None = Field(
        default=None,
        pattern=r"^experiment_promotion_[0-9a-f]{24}$",
    )
    action: ExperimentPromotionAliasAction
    rollback_of_event_id: str | None = Field(
        default=None,
        pattern=r"^promotion_alias_event_[0-9a-f]{24}$",
    )
    rollback_target_event_id: str | None = Field(
        default=None,
        pattern=r"^promotion_alias_event_[0-9a-f]{24}$",
    )
    receipt: ExperimentPromotionArchiveReceipt
    attestation: SignedExperimentPromotionArchiveAttestation
    attestation_verification: ExperimentPromotionArchiveAttestationVerification
    trusted_signer_key_ids: tuple[str, ...]
    lifecycle_event_id: str = Field(pattern=r"^promotion_lifecycle_event_[0-9a-f]{24}$")
    lifecycle_stage: ExperimentPromotionStage
    operator: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=8, max_length=4096)
    authorizers: tuple[str, ...] = Field(min_length=1)
    policy: ExperimentPromotionAliasPolicy
    rollback_ancestor_verified: bool
    checks: tuple[ExperimentPromotionRegistryCheck, ...]
    failed_check_count: int = Field(ge=0)
    eligible: bool

    @model_validator(mode="after")
    def validate_preview(self) -> ExperimentPromotionAliasPreview:
        _validate_governance(
            self.operator,
            self.reason,
            self.authorizers,
        )
        _validate_trust_evidence(
            self.receipt,
            self.attestation,
            self.attestation_verification,
            self.trusted_signer_key_ids,
        )
        if self.generation == 1:
            if self.current_event_id is not None or self.current_promotion_id is not None:
                raise ValueError("initial alias preview cannot contain current state")
        elif self.current_event_id is None or self.current_promotion_id is None:
            raise ValueError("continued alias preview requires complete current state")
        if self.action is ExperimentPromotionAliasAction.ASSIGN:
            if self.rollback_of_event_id is not None or self.rollback_target_event_id is not None:
                raise ValueError("normal alias assignment cannot contain rollback ancestry")
        elif self.rollback_of_event_id != self.current_event_id:
            raise ValueError("rollback must identify the current alias event")
        expected_checks = self.expected_checks(
            receipt=self.receipt,
            verification=self.attestation_verification,
            current_promotion_id=self.current_promotion_id,
            action=self.action,
            rollback_target_event_id=self.rollback_target_event_id,
            rollback_ancestor_verified=self.rollback_ancestor_verified,
            lifecycle_stage=self.lifecycle_stage,
            operator=self.operator,
            authorizers=self.authorizers,
            policy=self.policy,
        )
        if self.checks != expected_checks:
            raise ValueError("alias preview checks do not match its evidence")
        failed = sum(not item.passed for item in self.checks)
        if self.failed_check_count != failed or self.eligible != (failed == 0):
            raise ValueError("alias preview eligibility is inconsistent")
        expected_id = self.expected_preview_id(
            environment=self.environment,
            generation=self.generation,
            current_event_id=self.current_event_id,
            current_promotion_id=self.current_promotion_id,
            action=self.action,
            rollback_of_event_id=self.rollback_of_event_id,
            rollback_target_event_id=self.rollback_target_event_id,
            receipt=self.receipt,
            attestation=self.attestation,
            trusted_signer_key_ids=self.trusted_signer_key_ids,
            lifecycle_event_id=self.lifecycle_event_id,
            lifecycle_stage=self.lifecycle_stage,
            operator=self.operator,
            reason=self.reason,
            authorizers=self.authorizers,
            policy=self.policy,
            rollback_ancestor_verified=self.rollback_ancestor_verified,
            checks=self.checks,
        )
        if self.preview_id != expected_id:
            raise ValueError("alias preview ID does not match its contents")
        return self

    @staticmethod
    def expected_checks(
        *,
        receipt: ExperimentPromotionArchiveReceipt,
        verification: ExperimentPromotionArchiveAttestationVerification,
        current_promotion_id: str | None,
        action: ExperimentPromotionAliasAction,
        rollback_target_event_id: str | None,
        rollback_ancestor_verified: bool,
        lifecycle_stage: ExperimentPromotionStage,
        operator: str,
        authorizers: tuple[str, ...],
        policy: ExperimentPromotionAliasPolicy,
    ) -> tuple[ExperimentPromotionRegistryCheck, ...]:
        values = (
            ExperimentPromotionRegistryCheck(
                code="alias.changed",
                passed=current_promotion_id != receipt.promotion_id,
                detail="environment alias changes to another promotion",
                evidence=tuple(
                    sorted(
                        {
                            item
                            for item in (current_promotion_id, receipt.promotion_id)
                            if item is not None
                        }
                    )
                ),
            ),
            ExperimentPromotionRegistryCheck(
                code="attestation.signature",
                passed=(
                    verification.archive_receipt_matches
                    and verification.signer_identity_valid
                    and verification.payload_digest_valid
                    and verification.signature_valid
                ),
                detail="publisher attestation cryptographically matches the archive receipt",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionRegistryCheck(
                code="attestation.trusted",
                passed=verification.trusted_signer,
                detail="publisher key belongs to the supplied alias trust set",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionRegistryCheck(
                code="authorization.independent",
                passed=(policy.governance.operator_may_authorize or operator not in authorizers),
                detail="deployment authorization is independent from the operator",
                evidence=authorizers,
            ),
            ExperimentPromotionRegistryCheck(
                code="authorization.quorum",
                passed=len(authorizers) >= policy.governance.minimum_authorizers,
                detail="deployment authorization satisfies the configured quorum",
                evidence=authorizers,
            ),
            ExperimentPromotionRegistryCheck(
                code="lifecycle.stage",
                passed=lifecycle_stage in policy.allowed_stages,
                detail="promotion lifecycle stage satisfies the environment policy",
                evidence=(lifecycle_stage.value,),
            ),
            ExperimentPromotionRegistryCheck(
                code="rollback.ancestry",
                passed=(
                    action is ExperimentPromotionAliasAction.ASSIGN
                    or (rollback_target_event_id is not None and rollback_ancestor_verified)
                ),
                detail="rollback target appeared in this environment's alias history",
                evidence=(() if rollback_target_event_id is None else (rollback_target_event_id,)),
            ),
        )
        return tuple(sorted(values, key=lambda item: item.code))

    @staticmethod
    def expected_preview_id(
        *,
        environment: str,
        generation: int,
        current_event_id: str | None,
        current_promotion_id: str | None,
        action: ExperimentPromotionAliasAction,
        rollback_of_event_id: str | None,
        rollback_target_event_id: str | None,
        receipt: ExperimentPromotionArchiveReceipt,
        attestation: SignedExperimentPromotionArchiveAttestation,
        trusted_signer_key_ids: tuple[str, ...],
        lifecycle_event_id: str,
        lifecycle_stage: ExperimentPromotionStage,
        operator: str,
        reason: str,
        authorizers: tuple[str, ...],
        policy: ExperimentPromotionAliasPolicy,
        rollback_ancestor_verified: bool,
        checks: tuple[ExperimentPromotionRegistryCheck, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "environment": environment,
                "generation": generation,
                "current_event_id": current_event_id,
                "current_promotion_id": current_promotion_id,
                "action": action.value,
                "rollback_of_event_id": rollback_of_event_id,
                "rollback_target_event_id": rollback_target_event_id,
                "receipt": receipt.model_dump(mode="json"),
                "attestation_id": attestation.attestation.attestation_id,
                "trusted_signer_key_ids": trusted_signer_key_ids,
                "lifecycle_event_id": lifecycle_event_id,
                "lifecycle_stage": lifecycle_stage.value,
                "operator": operator,
                "reason": reason,
                "authorizers": authorizers,
                "policy": policy.model_dump(mode="json"),
                "rollback_ancestor_verified": rollback_ancestor_verified,
                "checks": [item.model_dump(mode="json") for item in checks],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"promotion_alias_preview_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionAliasEvent(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    event_id: str = Field(pattern=r"^promotion_alias_event_[0-9a-f]{24}$")
    preview: ExperimentPromotionAliasPreview

    @model_validator(mode="after")
    def validate_event(self) -> ExperimentPromotionAliasEvent:
        if not self.preview.eligible:
            raise ValueError("ineligible alias preview cannot become an event")
        expected = self.expected_event_id(self.preview.preview_id)
        if self.event_id != expected:
            raise ValueError("alias event ID does not match its preview")
        return self

    @staticmethod
    def expected_event_id(preview_id: str) -> str:
        return (
            f"promotion_alias_event_{hashlib.sha256(preview_id.encode('ascii')).hexdigest()[:24]}"
        )


class ExperimentPromotionLifecycleSummary(ContractModel):
    promotion_id: str = Field(pattern=r"^experiment_promotion_[0-9a-f]{24}$")
    promotion_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    archive_id: str = Field(pattern=r"^experiment_promotion_archive_[0-9a-f]{24}$")
    stage: ExperimentPromotionStage
    event_id: str = Field(pattern=r"^promotion_lifecycle_event_[0-9a-f]{24}$")
    event_count: int = Field(ge=1)


class ExperimentPromotionAliasSummary(ContractModel):
    environment: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    promotion_id: str = Field(pattern=r"^experiment_promotion_[0-9a-f]{24}$")
    promotion_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    archive_id: str = Field(pattern=r"^experiment_promotion_archive_[0-9a-f]{24}$")
    event_id: str = Field(pattern=r"^promotion_alias_event_[0-9a-f]{24}$")
    generation: int = Field(ge=1)
    action: ExperimentPromotionAliasAction


class ExperimentPromotionRegistryIssue(ContractModel):
    path: str = Field(min_length=1)
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> ExperimentPromotionRegistryIssue:
        parts = self.path.replace("\\", "/").split("/")
        if self.path.startswith(("/", "\\")) or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("promotion registry issue path must be safe and relative")
        return self


class ExperimentPromotionRegistryStatus(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    status_id: str = Field(pattern=r"^promotion_registry_status_[0-9a-f]{24}$")
    promotions: tuple[ExperimentPromotionLifecycleSummary, ...]
    aliases: tuple[ExperimentPromotionAliasSummary, ...]
    issues: tuple[ExperimentPromotionRegistryIssue, ...]
    valid: bool

    @model_validator(mode="after")
    def validate_status(self) -> ExperimentPromotionRegistryStatus:
        promotion_ids = tuple(item.promotion_id for item in self.promotions)
        environments = tuple(item.environment for item in self.aliases)
        issue_keys = tuple((item.path, item.detail) for item in self.issues)
        if promotion_ids != tuple(sorted(set(promotion_ids))):
            raise ValueError("registry promotion summaries must be sorted and unique")
        if environments != tuple(sorted(set(environments))):
            raise ValueError("registry alias summaries must be sorted and unique")
        if issue_keys != tuple(sorted(set(issue_keys))):
            raise ValueError("registry issues must be sorted and unique")
        if self.valid is not (not self.issues):
            raise ValueError("registry status validity does not match its issues")
        expected = self.expected_status_id(self.promotions, self.aliases, self.issues)
        if self.status_id != expected:
            raise ValueError("registry status ID does not match its contents")
        return self

    @staticmethod
    def expected_status_id(
        promotions: tuple[ExperimentPromotionLifecycleSummary, ...],
        aliases: tuple[ExperimentPromotionAliasSummary, ...],
        issues: tuple[ExperimentPromotionRegistryIssue, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "promotions": [item.model_dump(mode="json") for item in promotions],
                "aliases": [item.model_dump(mode="json") for item in aliases],
                "issues": [item.model_dump(mode="json") for item in issues],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"promotion_registry_status_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionRegistry:
    def __init__(self, store: ConditionalBlobStore) -> None:
        self.store = store

    def preview_lifecycle(
        self,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        *,
        trusted_public_keys: Collection[str],
        target_stage: ExperimentPromotionStage,
        operator: str,
        reason: str,
        authorizers: tuple[str, ...],
        policy: ExperimentPromotionGovernancePolicy | None = None,
    ) -> ExperimentPromotionLifecyclePreview:
        self._require_clean_registry()
        receipt, verification, trusted_ids = self._archive_evidence(
            archive,
            attestation,
            trusted_public_keys,
        )
        history = self.lifecycle(receipt.promotion_id)
        current = history[-1] if history else None
        active_environments = tuple(
            item.environment
            for item in self.status().aliases
            if item.promotion_id == receipt.promotion_id
        )
        resolved_policy = policy or ExperimentPromotionGovernancePolicy()
        sorted_authorizers = tuple(sorted(set(authorizers)))
        sequence = len(history) + 1
        source_stage = current.preview.target_stage if current is not None else None
        current_event_id = current.event_id if current is not None else None
        current_archive_id = current.preview.receipt.archive_id if current is not None else None
        checks = ExperimentPromotionLifecyclePreview.expected_checks(
            receipt=receipt,
            verification=verification,
            sequence=sequence,
            source_stage=source_stage,
            target_stage=target_stage,
            current_archive_id=current_archive_id,
            active_environments=active_environments,
            operator=operator,
            authorizers=sorted_authorizers,
            policy=resolved_policy,
        )
        preview_id = ExperimentPromotionLifecyclePreview.expected_preview_id(
            receipt=receipt,
            attestation=attestation,
            trusted_signer_key_ids=trusted_ids,
            sequence=sequence,
            current_event_id=current_event_id,
            source_stage=source_stage,
            target_stage=target_stage,
            current_archive_id=current_archive_id,
            active_environments=active_environments,
            operator=operator,
            reason=reason,
            authorizers=sorted_authorizers,
            policy=resolved_policy,
            checks=checks,
        )
        failed = sum(not item.passed for item in checks)
        return ExperimentPromotionLifecyclePreview(
            preview_id=preview_id,
            receipt=receipt,
            attestation=attestation,
            attestation_verification=verification,
            trusted_signer_key_ids=trusted_ids,
            sequence=sequence,
            current_event_id=current_event_id,
            source_stage=source_stage,
            target_stage=target_stage,
            current_archive_id=current_archive_id,
            active_environments=active_environments,
            operator=operator,
            reason=reason,
            authorizers=sorted_authorizers,
            policy=resolved_policy,
            checks=checks,
            failed_check_count=failed,
            eligible=failed == 0,
        )

    def execute_lifecycle(
        self,
        preview: ExperimentPromotionLifecyclePreview,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        *,
        trusted_public_keys: Collection[str],
        confirm_preview_id: str,
    ) -> ExperimentPromotionLifecycleEvent:
        if confirm_preview_id.strip() != preview.preview_id:
            raise ExperimentPromotionRegistryError(
                "lifecycle confirmation does not match the preview ID"
            )
        self._require_clean_registry()
        key = self.lifecycle_key(preview.receipt.promotion_id, preview.sequence)
        existing = self._try_load_lifecycle_event(key)
        if existing is not None:
            if existing.preview != preview:
                raise ExperimentPromotionRegistryError(
                    "lifecycle generation already contains another decision"
                )
            self._assert_archive_matches_preview(
                preview.receipt,
                preview.attestation_verification,
                preview.trusted_signer_key_ids,
                archive,
                attestation,
                trusted_public_keys,
            )
            return existing
        current = self.preview_lifecycle(
            archive,
            attestation,
            trusted_public_keys=trusted_public_keys,
            target_stage=preview.target_stage,
            operator=preview.operator,
            reason=preview.reason,
            authorizers=preview.authorizers,
            policy=preview.policy,
        )
        if current != preview:
            raise ExperimentPromotionRegistryError(
                "promotion lifecycle state changed after preview"
            )
        if not preview.eligible:
            raise ExperimentPromotionRegistryError("promotion lifecycle preview is not eligible")
        event = ExperimentPromotionLifecycleEvent(
            event_id=ExperimentPromotionLifecycleEvent.expected_event_id(preview.preview_id),
            preview=preview,
        )
        self._put_event(key, event.canonical_bytes() + b"\n", "lifecycle generation")
        return event

    def preview_alias(
        self,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        *,
        trusted_public_keys: Collection[str],
        environment: str,
        action: ExperimentPromotionAliasAction,
        operator: str,
        reason: str,
        authorizers: tuple[str, ...],
        policy: ExperimentPromotionAliasPolicy | None = None,
    ) -> ExperimentPromotionAliasPreview:
        if _ENVIRONMENT.fullmatch(environment) is None:
            raise ValueError("invalid promotion environment name")
        self._require_clean_registry()
        receipt, verification, trusted_ids = self._archive_evidence(
            archive,
            attestation,
            trusted_public_keys,
        )
        lifecycle = self.lifecycle(receipt.promotion_id)
        if not lifecycle:
            raise ExperimentPromotionRegistryError("target promotion has no lifecycle evidence")
        lifecycle_event = lifecycle[-1]
        aliases = self.alias_history(environment)
        current = aliases[-1] if aliases else None
        rollback_target = None
        rollback_verified = action is ExperimentPromotionAliasAction.ASSIGN
        if action is ExperimentPromotionAliasAction.ROLLBACK:
            rollback_target = next(
                (
                    item
                    for item in reversed(aliases[:-1])
                    if item.preview.receipt.promotion_id == receipt.promotion_id
                ),
                None,
            )
            rollback_verified = rollback_target is not None
        resolved_policy = policy or ExperimentPromotionAliasPolicy()
        sorted_authorizers = tuple(sorted(set(authorizers)))
        generation = len(aliases) + 1
        current_event_id = current.event_id if current is not None else None
        current_promotion_id = current.preview.receipt.promotion_id if current is not None else None
        rollback_of_event_id = (
            current.event_id
            if current is not None and action is ExperimentPromotionAliasAction.ROLLBACK
            else None
        )
        rollback_target_event_id = rollback_target.event_id if rollback_target is not None else None
        checks = ExperimentPromotionAliasPreview.expected_checks(
            receipt=receipt,
            verification=verification,
            current_promotion_id=current_promotion_id,
            action=action,
            rollback_target_event_id=rollback_target_event_id,
            rollback_ancestor_verified=rollback_verified,
            lifecycle_stage=lifecycle_event.preview.target_stage,
            operator=operator,
            authorizers=sorted_authorizers,
            policy=resolved_policy,
        )
        preview_id = ExperimentPromotionAliasPreview.expected_preview_id(
            environment=environment,
            generation=generation,
            current_event_id=current_event_id,
            current_promotion_id=current_promotion_id,
            action=action,
            rollback_of_event_id=rollback_of_event_id,
            rollback_target_event_id=rollback_target_event_id,
            receipt=receipt,
            attestation=attestation,
            trusted_signer_key_ids=trusted_ids,
            lifecycle_event_id=lifecycle_event.event_id,
            lifecycle_stage=lifecycle_event.preview.target_stage,
            operator=operator,
            reason=reason,
            authorizers=sorted_authorizers,
            policy=resolved_policy,
            rollback_ancestor_verified=rollback_verified,
            checks=checks,
        )
        failed = sum(not item.passed for item in checks)
        return ExperimentPromotionAliasPreview(
            preview_id=preview_id,
            environment=environment,
            generation=generation,
            current_event_id=current_event_id,
            current_promotion_id=current_promotion_id,
            action=action,
            rollback_of_event_id=rollback_of_event_id,
            rollback_target_event_id=rollback_target_event_id,
            receipt=receipt,
            attestation=attestation,
            attestation_verification=verification,
            trusted_signer_key_ids=trusted_ids,
            lifecycle_event_id=lifecycle_event.event_id,
            lifecycle_stage=lifecycle_event.preview.target_stage,
            operator=operator,
            reason=reason,
            authorizers=sorted_authorizers,
            policy=resolved_policy,
            rollback_ancestor_verified=rollback_verified,
            checks=checks,
            failed_check_count=failed,
            eligible=failed == 0,
        )

    def execute_alias(
        self,
        preview: ExperimentPromotionAliasPreview,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        *,
        trusted_public_keys: Collection[str],
        confirm_preview_id: str,
    ) -> ExperimentPromotionAliasEvent:
        if confirm_preview_id.strip() != preview.preview_id:
            raise ExperimentPromotionRegistryError(
                "alias confirmation does not match the preview ID"
            )
        self._require_clean_registry()
        key = self.alias_key(preview.environment, preview.generation)
        existing = self._try_load_alias_event(key)
        if existing is not None:
            if existing.preview != preview:
                raise ExperimentPromotionRegistryError(
                    "alias generation already contains another decision"
                )
            self._assert_archive_matches_preview(
                preview.receipt,
                preview.attestation_verification,
                preview.trusted_signer_key_ids,
                archive,
                attestation,
                trusted_public_keys,
            )
            return existing
        current = self.preview_alias(
            archive,
            attestation,
            trusted_public_keys=trusted_public_keys,
            environment=preview.environment,
            action=preview.action,
            operator=preview.operator,
            reason=preview.reason,
            authorizers=preview.authorizers,
            policy=preview.policy,
        )
        if current != preview:
            raise ExperimentPromotionRegistryError("environment alias state changed after preview")
        if not preview.eligible:
            raise ExperimentPromotionRegistryError("environment alias preview is not eligible")
        event = ExperimentPromotionAliasEvent(
            event_id=ExperimentPromotionAliasEvent.expected_event_id(preview.preview_id),
            preview=preview,
        )
        self._put_event(key, event.canonical_bytes() + b"\n", "alias generation")
        return event

    def lifecycle(self, promotion_id: str) -> tuple[ExperimentPromotionLifecycleEvent, ...]:
        if _PROMOTION_ID.fullmatch(promotion_id) is None:
            raise ValueError("invalid experiment promotion ID")
        prefix = f"promotions/{promotion_id}/"
        keys = self.store.list(prefix)
        events: list[ExperimentPromotionLifecycleEvent] = []
        previous: ExperimentPromotionLifecycleEvent | None = None
        first_receipt = None
        for expected_sequence, key in enumerate(keys, 1):
            match = _LIFECYCLE_KEY.fullmatch(key)
            if (
                match is None
                or match.group(1) != promotion_id
                or int(match.group(2)) != expected_sequence
            ):
                raise ExperimentPromotionRegistryError(f"invalid lifecycle stream member {key!r}")
            event = self._load_lifecycle_event(key)
            preview = event.preview
            if (
                preview.receipt.promotion_id != promotion_id
                or preview.sequence != expected_sequence
            ):
                raise ExperimentPromotionRegistryError(
                    "lifecycle event identity does not match its stream"
                )
            if previous is None:
                if preview.current_event_id is not None or preview.source_stage is not None:
                    raise ExperimentPromotionRegistryError(
                        "initial lifecycle event contains prior state"
                    )
                first_receipt = preview.receipt
            elif (
                preview.current_event_id != previous.event_id
                or preview.source_stage is not previous.preview.target_stage
                or preview.current_archive_id != previous.preview.receipt.archive_id
                or preview.receipt != first_receipt
            ):
                raise ExperimentPromotionRegistryError(
                    "lifecycle event does not continue the previous generation"
                )
            events.append(event)
            previous = event
        return tuple(events)

    def alias_history(self, environment: str) -> tuple[ExperimentPromotionAliasEvent, ...]:
        if _ENVIRONMENT.fullmatch(environment) is None:
            raise ValueError("invalid promotion environment name")
        prefix = f"environments/{environment}/"
        keys = self.store.list(prefix)
        events: list[ExperimentPromotionAliasEvent] = []
        previous: ExperimentPromotionAliasEvent | None = None
        for expected_generation, key in enumerate(keys, 1):
            match = _ALIAS_KEY.fullmatch(key)
            if (
                match is None
                or match.group(1) != environment
                or int(match.group(2)) != expected_generation
            ):
                raise ExperimentPromotionRegistryError(f"invalid alias stream member {key!r}")
            event = self._load_alias_event(key)
            preview = event.preview
            if preview.environment != environment or preview.generation != expected_generation:
                raise ExperimentPromotionRegistryError(
                    "alias event identity does not match its stream"
                )
            if previous is None:
                if preview.current_event_id is not None or preview.current_promotion_id is not None:
                    raise ExperimentPromotionRegistryError(
                        "initial alias event contains prior state"
                    )
            elif (
                preview.current_event_id != previous.event_id
                or preview.current_promotion_id != previous.preview.receipt.promotion_id
            ):
                raise ExperimentPromotionRegistryError(
                    "alias event does not continue the previous generation"
                )
            if preview.action is ExperimentPromotionAliasAction.ROLLBACK:
                ancestors = {item.event_id: item for item in events}
                target = ancestors.get(preview.rollback_target_event_id or "")
                if (
                    preview.rollback_of_event_id != (previous.event_id if previous else None)
                    or target is None
                    or target.preview.receipt.promotion_id != preview.receipt.promotion_id
                ):
                    raise ExperimentPromotionRegistryError("alias rollback ancestry is invalid")
            events.append(event)
            previous = event
        return tuple(events)

    def status(self) -> ExperimentPromotionRegistryStatus:
        keys = self.store.list()
        lifecycle_ids = sorted(
            {match.group(1) for key in keys if (match := _LIFECYCLE_KEY.fullmatch(key)) is not None}
        )
        environments = sorted(
            {match.group(1) for key in keys if (match := _ALIAS_KEY.fullmatch(key)) is not None}
        )
        recognized = {
            key
            for key in keys
            if _LIFECYCLE_KEY.fullmatch(key) is not None or _ALIAS_KEY.fullmatch(key) is not None
        }
        issues = [
            ExperimentPromotionRegistryIssue(
                path=key,
                detail="unrecognized promotion registry entry",
            )
            for key in keys
            if key not in recognized
        ]
        promotions: list[ExperimentPromotionLifecycleSummary] = []
        lifecycle_events: dict[str, ExperimentPromotionLifecycleEvent] = {}
        for promotion_id in lifecycle_ids:
            try:
                lifecycle_history = self.lifecycle(promotion_id)
            except (ExperimentPromotionRegistryError, KeyError, ValueError) as error:
                issues.append(
                    ExperimentPromotionRegistryIssue(
                        path=f"promotions/{promotion_id}",
                        detail=str(error),
                    )
                )
                continue
            if not lifecycle_history:
                continue
            latest_lifecycle = lifecycle_history[-1]
            lifecycle_events.update((item.event_id, item) for item in lifecycle_history)
            promotions.append(
                ExperimentPromotionLifecycleSummary(
                    promotion_id=promotion_id,
                    promotion_name=latest_lifecycle.preview.receipt.promotion_name,
                    archive_id=latest_lifecycle.preview.receipt.archive_id,
                    stage=latest_lifecycle.preview.target_stage,
                    event_id=latest_lifecycle.event_id,
                    event_count=len(lifecycle_history),
                )
            )
        aliases: list[ExperimentPromotionAliasSummary] = []
        for environment in environments:
            try:
                alias_history = self.alias_history(environment)
                if not alias_history:
                    continue
                for event in alias_history:
                    lifecycle_event = lifecycle_events.get(event.preview.lifecycle_event_id)
                    if (
                        lifecycle_event is None
                        or lifecycle_event.preview.receipt != event.preview.receipt
                        or lifecycle_event.preview.target_stage is not event.preview.lifecycle_stage
                    ):
                        raise ExperimentPromotionRegistryError(
                            "alias references missing or inconsistent lifecycle evidence"
                        )
                latest_alias = alias_history[-1]
                current_lifecycle = next(
                    (
                        item
                        for item in promotions
                        if item.promotion_id == latest_alias.preview.receipt.promotion_id
                    ),
                    None,
                )
                if (
                    current_lifecycle is None
                    or current_lifecycle.stage is ExperimentPromotionStage.RETIRED
                ):
                    raise ExperimentPromotionRegistryError(
                        "active alias points to a missing or retired promotion"
                    )
            except (ExperimentPromotionRegistryError, KeyError, ValueError) as error:
                issues.append(
                    ExperimentPromotionRegistryIssue(
                        path=f"environments/{environment}",
                        detail=str(error),
                    )
                )
                continue
            aliases.append(
                ExperimentPromotionAliasSummary(
                    environment=environment,
                    promotion_id=latest_alias.preview.receipt.promotion_id,
                    promotion_name=latest_alias.preview.receipt.promotion_name,
                    archive_id=latest_alias.preview.receipt.archive_id,
                    event_id=latest_alias.event_id,
                    generation=latest_alias.preview.generation,
                    action=latest_alias.preview.action,
                )
            )
        promotion_tuple = tuple(sorted(promotions, key=lambda item: item.promotion_id))
        alias_tuple = tuple(sorted(aliases, key=lambda item: item.environment))
        issue_tuple = tuple(sorted(set(issues), key=lambda item: (item.path, item.detail)))
        status_id = ExperimentPromotionRegistryStatus.expected_status_id(
            promotion_tuple,
            alias_tuple,
            issue_tuple,
        )
        return ExperimentPromotionRegistryStatus(
            status_id=status_id,
            promotions=promotion_tuple,
            aliases=alias_tuple,
            issues=issue_tuple,
            valid=not issue_tuple,
        )

    @staticmethod
    def lifecycle_key(promotion_id: str, sequence: int) -> str:
        if _PROMOTION_ID.fullmatch(promotion_id) is None or sequence < 1:
            raise ValueError("invalid lifecycle event key")
        return f"promotions/{promotion_id}/{sequence:08d}.json"

    @staticmethod
    def alias_key(environment: str, generation: int) -> str:
        if _ENVIRONMENT.fullmatch(environment) is None or generation < 1:
            raise ValueError("invalid environment alias key")
        return f"environments/{environment}/{generation:08d}.json"

    def _require_clean_registry(self) -> None:
        status = self.status()
        if not status.valid:
            raise ExperimentPromotionRegistryError(
                "promotion registry contains invalid or unrecognized state"
            )

    @staticmethod
    def _archive_evidence(
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        trusted_public_keys: Collection[str],
    ) -> tuple[
        ExperimentPromotionArchiveReceipt,
        ExperimentPromotionArchiveAttestationVerification,
        tuple[str, ...],
    ]:
        trusted_keys = tuple(trusted_public_keys)
        trusted_ids = tuple(
            sorted({Ed25519ManifestSigner.public_key_id(item) for item in trusted_keys})
        )
        expected = attestation.attestation.receipt.content_digest
        receipt = ExperimentPromotionArchive.inspect(archive, expected_sha256=expected)
        verification = ExperimentPromotionArchiveAttestor.verify(
            attestation,
            receipt,
            trusted_public_keys=trusted_keys,
        )
        return receipt, verification, trusted_ids

    @classmethod
    def _assert_archive_matches_preview(
        cls,
        expected_receipt: ExperimentPromotionArchiveReceipt,
        expected_verification: ExperimentPromotionArchiveAttestationVerification,
        expected_trusted_ids: tuple[str, ...],
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        trusted_public_keys: Collection[str],
    ) -> None:
        receipt, verification, trusted_ids = cls._archive_evidence(
            archive,
            attestation,
            trusted_public_keys,
        )
        if (
            receipt != expected_receipt
            or verification != expected_verification
            or trusted_ids != expected_trusted_ids
        ):
            raise ExperimentPromotionRegistryError(
                "archive trust evidence differs from the confirmed preview"
            )

    def _put_event(self, key: str, payload: bytes, label: str) -> None:
        try:
            self.store.put_if_absent(key, payload)
        except BlobConflictError as error:
            raise ExperimentPromotionRegistryError(
                f"{label} was claimed by another decision"
            ) from error

    def _try_load_lifecycle_event(
        self,
        key: str,
    ) -> ExperimentPromotionLifecycleEvent | None:
        if self.store.head(key) is None:
            return None
        return self._load_lifecycle_event(key)

    def _load_lifecycle_event(self, key: str) -> ExperimentPromotionLifecycleEvent:
        payload = self.store.get(key)
        try:
            event = ExperimentPromotionLifecycleEvent.model_validate_json(payload)
        except ValueError as error:
            raise ExperimentPromotionRegistryError(
                f"invalid lifecycle event {key!r}: {error}"
            ) from error
        if payload != event.canonical_bytes() + b"\n":
            raise ExperimentPromotionRegistryError(f"lifecycle event {key!r} is not canonical")
        return event

    def _try_load_alias_event(self, key: str) -> ExperimentPromotionAliasEvent | None:
        if self.store.head(key) is None:
            return None
        return self._load_alias_event(key)

    def _load_alias_event(self, key: str) -> ExperimentPromotionAliasEvent:
        payload = self.store.get(key)
        try:
            event = ExperimentPromotionAliasEvent.model_validate_json(payload)
        except ValueError as error:
            raise ExperimentPromotionRegistryError(
                f"invalid alias event {key!r}: {error}"
            ) from error
        if payload != event.canonical_bytes() + b"\n":
            raise ExperimentPromotionRegistryError(f"alias event {key!r} is not canonical")
        return event


def _allowed_lifecycle_targets(
    source: ExperimentPromotionStage | None,
) -> tuple[ExperimentPromotionStage, ...]:
    transitions = {
        None: (ExperimentPromotionStage.CANDIDATE,),
        ExperimentPromotionStage.CANDIDATE: (
            ExperimentPromotionStage.RETIRED,
            ExperimentPromotionStage.STAGING,
        ),
        ExperimentPromotionStage.STAGING: (
            ExperimentPromotionStage.PRODUCTION,
            ExperimentPromotionStage.RETIRED,
        ),
        ExperimentPromotionStage.PRODUCTION: (ExperimentPromotionStage.RETIRED,),
        ExperimentPromotionStage.RETIRED: (),
    }
    return transitions[source]


def _validate_governance(
    operator: str,
    reason: str,
    authorizers: tuple[str, ...],
) -> None:
    if operator != operator.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in operator
    ):
        raise ValueError("promotion registry operator must be a printable identity")
    if "\x00" in reason:
        raise ValueError("promotion registry reason cannot contain NUL")
    if authorizers != tuple(sorted(set(authorizers))):
        raise ValueError("promotion registry authorizers must be sorted and unique")
    if any(
        not identity
        or len(identity) > 256
        or identity != identity.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in identity)
        for identity in authorizers
    ):
        raise ValueError("promotion registry authorizers must be printable identities")


def _validate_trust_evidence(
    receipt: ExperimentPromotionArchiveReceipt,
    attestation: SignedExperimentPromotionArchiveAttestation,
    verification: ExperimentPromotionArchiveAttestationVerification,
    trusted_signer_key_ids: tuple[str, ...],
) -> None:
    if trusted_signer_key_ids != tuple(sorted(set(trusted_signer_key_ids))) or any(
        re.fullmatch(r"ed25519_[0-9a-f]{24}", item) is None for item in trusted_signer_key_ids
    ):
        raise ValueError("trusted promotion signer key IDs must be sorted and unique")
    if attestation.attestation.receipt != receipt:
        raise ValueError("promotion registry attestation receipt is inconsistent")
    embedded_key = attestation.signature.public_key_base64
    embedded_id = Ed25519ManifestSigner.public_key_id(embedded_key)
    derived = ExperimentPromotionArchiveAttestor.verify(
        attestation,
        receipt,
        trusted_public_keys=(embedded_key,) if embedded_id in trusted_signer_key_ids else (),
    )
    if verification != derived:
        raise ValueError("promotion registry attestation verification is inconsistent")

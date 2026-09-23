from __future__ import annotations

import hashlib
import html
import re
from enum import Enum
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts import (
    ArtifactLocation,
    CheckpointManifest,
    ContractModel,
)
from agentic_rl_forge.experiments.models import (
    ExperimentArtifactEvidence,
    ExperimentArtifactKind,
    ExperimentPlan,
    ExperimentReport,
    ExperimentScalar,
    ExperimentStageState,
    ExperimentTrialState,
)
from agentic_rl_forge.experiments.operations import (
    ExperimentAnalysis,
    ExperimentIndexBuilder,
    ExperimentOperationsIndex,
    ExperimentRankedTrial,
)
from agentic_rl_forge.experiments.runner import ExperimentRunner
from agentic_rl_forge.pipelines import load_search_r1_collection_config
from agentic_rl_forge.storage import BlobConflictError, LocalBlobStore

_PROMOTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_CHECK_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class ExperimentPromotionArtifactScope(str, Enum):
    PROJECT = "project"
    STATE = "state"
    REMOTE = "remote"


class ExperimentPromotionArtifactRole(str, Enum):
    CONFIG = "config"
    INPUT = "input"
    OUTPUT = "output"
    CHECKPOINT_PAYLOAD = "checkpoint_payload"


class ExperimentPromotionArtifactRef(ContractModel):
    scope: ExperimentPromotionArtifactScope
    locator: str = Field(min_length=1)
    kind: ExperimentArtifactKind
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_id: str | None = None
    stages: tuple[str, ...] = ()
    roles: tuple[ExperimentPromotionArtifactRole, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference(self) -> ExperimentPromotionArtifactRef:
        if self.scope is ExperimentPromotionArtifactScope.REMOTE:
            parsed = urlsplit(self.locator)
            if (
                parsed.scheme not in {"https", "s3", "gs", "az"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("remote promotion artifact locator must be a secret-free URI")
        else:
            path = PurePosixPath(self.locator)
            if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
                raise ValueError("promotion artifact path must be safe and relative")
        if self.stages != tuple(sorted(set(self.stages))):
            raise ValueError("promotion artifact stages must be sorted and unique")
        if self.roles != tuple(sorted(set(self.roles), key=lambda item: item.value)):
            raise ValueError("promotion artifact roles must be sorted and unique")
        return self


class ExperimentReproducibilityManifest(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    manifest_id: str = Field(pattern=r"^experiment_repro_[0-9a-f]{24}$")
    promotion_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    index_id: str = Field(pattern=r"^experiment_index_[0-9a-f]{24}$")
    analysis_id: str = Field(pattern=r"^experiment_analysis_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    report_id: str = Field(pattern=r"^experiment_report_[0-9a-f]{24}$")
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    experiment_name: str = Field(min_length=1)
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    parameters: dict[str, ExperimentScalar]
    artifacts: tuple[ExperimentPromotionArtifactRef, ...]
    artifact_count: int = Field(ge=1)
    total_size_bytes: int = Field(ge=0)
    checkpoint_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> ExperimentReproducibilityManifest:
        if tuple(self.parameters) != tuple(sorted(self.parameters)):
            raise ValueError("promotion manifest parameters must be sorted")
        artifact_keys = tuple(
            (item.scope.value, item.locator, item.kind.value) for item in self.artifacts
        )
        if artifact_keys != tuple(sorted(set(artifact_keys))):
            raise ValueError("promotion manifest artifacts must be sorted and unique")
        if self.artifact_count != len(self.artifacts):
            raise ValueError("promotion manifest artifact count is inconsistent")
        if self.total_size_bytes != sum(item.size_bytes for item in self.artifacts):
            raise ValueError("promotion manifest byte count is inconsistent")
        expected_checkpoints = tuple(
            sorted(
                item.content_id
                for item in self.artifacts
                if item.kind is ExperimentArtifactKind.CHECKPOINT_MANIFEST
                and item.content_id is not None
            )
        )
        if self.checkpoint_ids != expected_checkpoints:
            raise ValueError("promotion manifest checkpoint IDs are inconsistent")
        expected = self.expected_manifest_id(
            promotion_name=self.promotion_name,
            index_id=self.index_id,
            analysis_id=self.analysis_id,
            plan_id=self.plan_id,
            report_id=self.report_id,
            trial_id=self.trial_id,
            experiment_name=self.experiment_name,
            config_digest=self.config_digest,
            model=self.model,
            policy_version=self.policy_version,
            dataset_name=self.dataset_name,
            parameters=self.parameters,
            artifacts=self.artifacts,
        )
        if self.manifest_id != expected:
            raise ValueError("promotion manifest ID does not match its contents")
        return self

    @staticmethod
    def expected_manifest_id(
        *,
        promotion_name: str,
        index_id: str,
        analysis_id: str,
        plan_id: str,
        report_id: str,
        trial_id: str,
        experiment_name: str,
        config_digest: str,
        model: str,
        policy_version: str,
        dataset_name: str,
        parameters: dict[str, ExperimentScalar],
        artifacts: tuple[ExperimentPromotionArtifactRef, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "promotion_name": promotion_name,
                "index_id": index_id,
                "analysis_id": analysis_id,
                "plan_id": plan_id,
                "report_id": report_id,
                "trial_id": trial_id,
                "experiment_name": experiment_name,
                "config_digest": config_digest,
                "model": model,
                "policy_version": policy_version,
                "dataset_name": dataset_name,
                "parameters": parameters,
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_repro_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionPolicy(ContractModel):
    required_artifact_kinds: tuple[ExperimentArtifactKind, ...] = (
        ExperimentArtifactKind.BENCHMARK_REPORT,
        ExperimentArtifactKind.CHECKPOINT_MANIFEST,
    )
    require_clean_index: bool = True
    allow_regression: bool = False
    require_pareto_front: bool = False
    maximum_rank: int | None = Field(default=None, ge=1)
    require_single_checkpoint: bool = True
    require_fully_verified_checkpoint: bool = True
    require_dataset_lineage: bool = True
    minimum_approvals: int = Field(default=1, ge=1, le=16)
    operator_may_approve: bool = False

    @model_validator(mode="after")
    def validate_policy(self) -> ExperimentPromotionPolicy:
        expected = tuple(sorted(set(self.required_artifact_kinds), key=lambda item: item.value))
        if self.required_artifact_kinds != expected:
            raise ValueError("required promotion artifact kinds must be sorted and unique")
        return self


class ExperimentPromotionCheck(ContractModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    passed: bool
    detail: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_check(self) -> ExperimentPromotionCheck:
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise ValueError("promotion check evidence must be sorted and unique")
        return self


class ExperimentPromotionPreview(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    preview_id: str = Field(pattern=r"^experiment_promotion_preview_[0-9a-f]{24}$")
    promotion_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    index_id: str = Field(pattern=r"^experiment_index_[0-9a-f]{24}$")
    analysis_id: str = Field(pattern=r"^experiment_analysis_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    report_id: str = Field(pattern=r"^experiment_report_[0-9a-f]{24}$")
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    state: ExperimentTrialState
    rank: int | None = Field(default=None, ge=1)
    pareto_front: bool
    policy: ExperimentPromotionPolicy
    manifest: ExperimentReproducibilityManifest
    checks: tuple[ExperimentPromotionCheck, ...]
    failed_check_count: int = Field(ge=0)
    eligible: bool

    @model_validator(mode="after")
    def validate_preview(self) -> ExperimentPromotionPreview:
        check_codes = tuple(item.code for item in self.checks)
        if check_codes != tuple(sorted(set(check_codes))):
            raise ValueError("promotion preview checks must be sorted and unique")
        failed = sum(not item.passed for item in self.checks)
        if self.failed_check_count != failed or self.eligible != (failed == 0):
            raise ValueError("promotion preview eligibility is inconsistent")
        if (
            self.manifest.promotion_name != self.promotion_name
            or self.manifest.index_id != self.index_id
            or self.manifest.analysis_id != self.analysis_id
            or self.manifest.plan_id != self.plan_id
            or self.manifest.report_id != self.report_id
            or self.manifest.trial_id != self.trial_id
        ):
            raise ValueError("promotion preview manifest identity is inconsistent")
        expected = self.expected_preview_id(
            promotion_name=self.promotion_name,
            index_id=self.index_id,
            analysis_id=self.analysis_id,
            plan_id=self.plan_id,
            report_id=self.report_id,
            trial_id=self.trial_id,
            state=self.state,
            rank=self.rank,
            pareto_front=self.pareto_front,
            policy=self.policy,
            manifest=self.manifest,
            checks=self.checks,
        )
        if self.preview_id != expected:
            raise ValueError("promotion preview ID does not match its contents")
        return self

    @staticmethod
    def expected_preview_id(
        *,
        promotion_name: str,
        index_id: str,
        analysis_id: str,
        plan_id: str,
        report_id: str,
        trial_id: str,
        state: ExperimentTrialState,
        rank: int | None,
        pareto_front: bool,
        policy: ExperimentPromotionPolicy,
        manifest: ExperimentReproducibilityManifest,
        checks: tuple[ExperimentPromotionCheck, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "promotion_name": promotion_name,
                "index_id": index_id,
                "analysis_id": analysis_id,
                "plan_id": plan_id,
                "report_id": report_id,
                "trial_id": trial_id,
                "state": state.value,
                "rank": rank,
                "pareto_front": pareto_front,
                "policy": policy.model_dump(mode="json"),
                "manifest": manifest.model_dump(mode="json"),
                "checks": [item.model_dump(mode="json") for item in checks],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_promotion_preview_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    promotion_id: str = Field(pattern=r"^experiment_promotion_[0-9a-f]{24}$")
    preview: ExperimentPromotionPreview
    operator: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=8, max_length=4096)
    approvers: tuple[str, ...] = Field(min_length=1)
    model_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> ExperimentPromotionRecord:
        if not self.preview.eligible:
            raise ValueError("ineligible experiment cannot be promoted")
        if self.operator != self.operator.strip() or any(
            ord(character) < 32 or ord(character) == 127 for character in self.operator
        ):
            raise ValueError("promotion operator must be a single printable identity")
        if "\x00" in self.reason:
            raise ValueError("promotion reason cannot contain NUL")
        if self.approvers != tuple(sorted(set(self.approvers))):
            raise ValueError("promotion approvers must be sorted and unique")
        if any(
            not identity
            or len(identity) > 256
            or identity != identity.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in identity)
            for identity in self.approvers
        ):
            raise ValueError("promotion approvers must be printable identities")
        if len(self.approvers) < self.preview.policy.minimum_approvals:
            raise ValueError("promotion does not satisfy its minimum approval count")
        if not self.preview.policy.operator_may_approve and self.operator in self.approvers:
            raise ValueError("promotion operator cannot satisfy an independent approval")
        expected = self.expected_promotion_id(
            preview=self.preview,
            operator=self.operator,
            reason=self.reason,
            approvers=self.approvers,
            model_card_sha256=self.model_card_sha256,
        )
        if self.promotion_id != expected:
            raise ValueError("promotion ID does not match its contents")
        return self

    @staticmethod
    def expected_promotion_id(
        *,
        preview: ExperimentPromotionPreview,
        operator: str,
        reason: str,
        approvers: tuple[str, ...],
        model_card_sha256: str,
    ) -> str:
        payload = orjson.dumps(
            {
                "preview_id": preview.preview_id,
                "operator": operator,
                "reason": reason,
                "approvers": approvers,
                "model_card_sha256": model_card_sha256,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_promotion_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionError(ValueError):
    pass


class ExperimentPromoter:
    def __init__(
        self,
        root: Path | str,
        state_root: Path | str,
        promotion_root: Path | str,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        self.state_root = Path(state_root).resolve(strict=True)
        self.promotion_root = Path(promotion_root).resolve()
        if not self.root.is_dir() or not self.state_root.is_dir():
            raise ValueError("experiment root and state directory must be directories")
        if self.promotion_root == self.state_root or self.promotion_root.is_relative_to(
            self.state_root
        ):
            raise ValueError("promotion output must stay outside experiment state")

    def preview(
        self,
        index: ExperimentOperationsIndex,
        analysis: ExperimentAnalysis,
        *,
        promotion_name: str,
        plan_id: str,
        trial_id: str,
        policy: ExperimentPromotionPolicy,
    ) -> ExperimentPromotionPreview:
        if _PROMOTION_NAME.fullmatch(promotion_name) is None:
            raise ValueError("invalid experiment promotion name")
        if analysis.index_id != index.index_id:
            raise ValueError("experiment analysis was produced from another index")
        selected = self._selected_trial(analysis, plan_id, trial_id)
        plan = self._load_plan(plan_id)
        plan_trial = next((item for item in plan.trials if item.trial_id == trial_id), None)
        if plan_trial is None:
            raise ExperimentPromotionError("promotion trial is not present in its plan")
        report = ExperimentRunner(self.root, self.state_root).report(plan)
        report_trial = next(item for item in report.trials if item.trial_id == trial_id)
        refs, checkpoint_manifests, checkpoint_errors = self._collect_artifacts(
            plan,
            report,
            trial_id,
        )
        manifest = self._manifest(
            promotion_name=promotion_name,
            index=index,
            analysis=analysis,
            plan=plan,
            report=report,
            trial_id=trial_id,
            refs=refs,
        )
        current_index = ExperimentIndexBuilder(self.root, self.state_root).build()
        checks = self._checks(
            index=index,
            current_index=current_index,
            selected=selected,
            report=report,
            report_trial_state=report_trial.state,
            policy=policy,
            refs=refs,
            checkpoint_manifests=checkpoint_manifests,
            checkpoint_errors=checkpoint_errors,
        )
        check_tuple = tuple(sorted(checks, key=lambda item: item.code))
        preview_id = ExperimentPromotionPreview.expected_preview_id(
            promotion_name=promotion_name,
            index_id=index.index_id,
            analysis_id=analysis.analysis_id,
            plan_id=plan_id,
            report_id=report.report_id,
            trial_id=trial_id,
            state=report_trial.state,
            rank=selected.rank,
            pareto_front=selected.pareto_front,
            policy=policy,
            manifest=manifest,
            checks=check_tuple,
        )
        failed = sum(not item.passed for item in check_tuple)
        return ExperimentPromotionPreview(
            preview_id=preview_id,
            promotion_name=promotion_name,
            index_id=index.index_id,
            analysis_id=analysis.analysis_id,
            plan_id=plan_id,
            report_id=report.report_id,
            trial_id=trial_id,
            state=report_trial.state,
            rank=selected.rank,
            pareto_front=selected.pareto_front,
            policy=policy,
            manifest=manifest,
            checks=check_tuple,
            failed_check_count=failed,
            eligible=failed == 0,
        )

    def promote(
        self,
        index: ExperimentOperationsIndex,
        analysis: ExperimentAnalysis,
        preview: ExperimentPromotionPreview,
        *,
        confirm_preview_id: str,
        operator: str,
        reason: str,
        approvers: tuple[str, ...],
    ) -> ExperimentPromotionRecord:
        if confirm_preview_id.strip() != preview.preview_id:
            raise ExperimentPromotionError("promotion confirmation does not match the preview ID")
        current = self.preview(
            index,
            analysis,
            promotion_name=preview.promotion_name,
            plan_id=preview.plan_id,
            trial_id=preview.trial_id,
            policy=preview.policy,
        )
        if current != preview:
            raise ExperimentPromotionError("experiment promotion state changed after preview")
        if not preview.eligible:
            raise ExperimentPromotionError("experiment promotion preview is not eligible")
        sorted_approvers = tuple(sorted(set(approvers)))
        model_card = render_experiment_model_card(
            index, analysis, preview, operator, reason, sorted_approvers
        )
        model_card_sha256 = hashlib.sha256(model_card.encode("utf-8")).hexdigest()
        promotion_id = ExperimentPromotionRecord.expected_promotion_id(
            preview=preview,
            operator=operator,
            reason=reason,
            approvers=sorted_approvers,
            model_card_sha256=model_card_sha256,
        )
        record = ExperimentPromotionRecord(
            promotion_id=promotion_id,
            preview=preview,
            operator=operator,
            reason=reason,
            approvers=sorted_approvers,
            model_card_sha256=model_card_sha256,
        )
        prefix = preview.promotion_name
        store = LocalBlobStore(self.promotion_root)
        try:
            store.put_if_absent(
                f"{prefix}/decision.json",
                record.canonical_bytes() + b"\n",
            )
        except BlobConflictError as error:
            raise ExperimentPromotionError(
                "promotion name already has a different immutable decision"
            ) from error
        self._publish_sidecars(store, index, analysis, preview, record, model_card)
        return record

    @staticmethod
    def _selected_trial(
        analysis: ExperimentAnalysis,
        plan_id: str,
        trial_id: str,
    ) -> ExperimentRankedTrial:
        selected = next(
            (
                item
                for item in analysis.trials
                if item.plan_id == plan_id and item.trial_id == trial_id
            ),
            None,
        )
        if selected is None:
            raise ExperimentPromotionError("promotion trial is not present in the analysis")
        return selected

    def _load_plan(self, plan_id: str) -> ExperimentPlan:
        if re.fullmatch(r"experiment_plan_[0-9a-f]{24}", plan_id) is None:
            raise ValueError("invalid experiment plan ID")
        path = self.state_root / plan_id / "plan.json"
        payload = path.read_bytes()
        plan = ExperimentPlan.model_validate_json(payload)
        if plan.plan_id != plan_id or payload != plan.canonical_bytes() + b"\n":
            raise ExperimentPromotionError("persisted promotion plan is invalid")
        return plan

    def _collect_artifacts(
        self,
        plan: ExperimentPlan,
        report: ExperimentReport,
        trial_id: str,
    ) -> tuple[
        tuple[ExperimentPromotionArtifactRef, ...],
        tuple[CheckpointManifest, ...],
        tuple[str, ...],
    ]:
        summary = next(item for item in report.trials if item.trial_id == trial_id)
        merged: dict[
            tuple[ExperimentPromotionArtifactScope, str, ExperimentArtifactKind],
            tuple[ExperimentArtifactEvidence, set[str], set[ExperimentPromotionArtifactRole]],
        ] = {}

        config_path = self.state_root / plan.plan_id / trial_id / "resolved-config.json"
        config_payload = config_path.read_bytes()
        config = load_search_r1_collection_config(config_path)
        config_evidence = ExperimentArtifactEvidence(
            kind=ExperimentArtifactKind.COLLECTION_CONFIG,
            path="{config_path}",
            size_bytes=len(config_payload),
            sha256=hashlib.sha256(config_payload).hexdigest(),
            content_id=config.plan_config_digest,
        )
        self._merge_evidence(
            merged,
            evidence=config_evidence,
            scope=ExperimentPromotionArtifactScope.STATE,
            locator=config_path.relative_to(self.state_root).as_posix(),
            stage=None,
            role=ExperimentPromotionArtifactRole.CONFIG,
        )

        for stage in summary.stages:
            if stage.state is not ExperimentStageState.SUCCEEDED:
                continue
            for evidence in stage.inputs:
                scope, locator = self._evidence_locator(plan, trial_id, evidence)
                self._merge_evidence(
                    merged,
                    evidence=evidence,
                    scope=scope,
                    locator=locator,
                    stage=stage.name,
                    role=ExperimentPromotionArtifactRole.INPUT,
                )
            for evidence in stage.outputs:
                scope, locator = self._evidence_locator(plan, trial_id, evidence)
                self._merge_evidence(
                    merged,
                    evidence=evidence,
                    scope=scope,
                    locator=locator,
                    stage=stage.name,
                    role=ExperimentPromotionArtifactRole.OUTPUT,
                )

        refs = [
            ExperimentPromotionArtifactRef(
                scope=scope,
                locator=locator,
                kind=kind,
                size_bytes=evidence.size_bytes,
                sha256=evidence.sha256,
                content_id=evidence.content_id,
                stages=tuple(sorted(stages)),
                roles=tuple(sorted(roles, key=lambda item: item.value)),
            )
            for (scope, locator, kind), (evidence, stages, roles) in merged.items()
        ]
        checkpoints, payload_refs, checkpoint_errors = self._checkpoint_payload_refs(refs)
        refs.extend(payload_refs)
        unique_refs: dict[
            tuple[ExperimentPromotionArtifactScope, str, ExperimentArtifactKind],
            ExperimentPromotionArtifactRef,
        ] = {}
        for ref in refs:
            key = (ref.scope, ref.locator, ref.kind)
            existing = unique_refs.setdefault(key, ref)
            if existing != ref:
                raise ExperimentPromotionError("promotion artifact references are inconsistent")
        return (
            tuple(
                sorted(
                    unique_refs.values(),
                    key=lambda item: (item.scope.value, item.locator, item.kind.value),
                )
            ),
            tuple(checkpoints),
            tuple(sorted(checkpoint_errors)),
        )

    @staticmethod
    def _merge_evidence(
        merged: dict[
            tuple[ExperimentPromotionArtifactScope, str, ExperimentArtifactKind],
            tuple[ExperimentArtifactEvidence, set[str], set[ExperimentPromotionArtifactRole]],
        ],
        *,
        evidence: ExperimentArtifactEvidence,
        scope: ExperimentPromotionArtifactScope,
        locator: str,
        stage: str | None,
        role: ExperimentPromotionArtifactRole,
    ) -> None:
        key = (scope, locator, evidence.kind)
        existing = merged.get(key)
        if existing is None:
            stages = set() if stage is None else {stage}
            merged[key] = (evidence, stages, {role})
            return
        prior, stages, roles = existing
        if (
            prior.size_bytes != evidence.size_bytes
            or prior.sha256 != evidence.sha256
            or prior.content_id != evidence.content_id
        ):
            raise ExperimentPromotionError("promotion artifact evidence is inconsistent")
        if stage is not None:
            stages.add(stage)
        roles.add(role)

    def _evidence_locator(
        self,
        plan: ExperimentPlan,
        trial_id: str,
        evidence: ExperimentArtifactEvidence,
    ) -> tuple[ExperimentPromotionArtifactScope, str]:
        if evidence.path == "{config_path}":
            path = self.state_root / plan.plan_id / trial_id / "resolved-config.json"
            return ExperimentPromotionArtifactScope.STATE, path.relative_to(
                self.state_root
            ).as_posix()
        path = (self.root / evidence.path).resolve(strict=True)
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ExperimentPromotionError("promotion artifact escapes the project root") from error
        payload = path.read_bytes()
        if (
            len(payload) != evidence.size_bytes
            or hashlib.sha256(payload).hexdigest() != evidence.sha256
        ):
            raise ExperimentPromotionError("promotion artifact changed during preview")
        return ExperimentPromotionArtifactScope.PROJECT, relative

    def _checkpoint_payload_refs(
        self,
        refs: list[ExperimentPromotionArtifactRef],
    ) -> tuple[
        list[CheckpointManifest],
        list[ExperimentPromotionArtifactRef],
        list[str],
    ]:
        manifests = []
        payload_refs = []
        errors = []
        for ref in refs:
            if ref.kind is not ExperimentArtifactKind.CHECKPOINT_MANIFEST:
                continue
            manifest_path = self._reference_path(ref)
            try:
                manifest = CheckpointManifest.model_validate_json(manifest_path.read_bytes())
            except ValueError:
                errors.append(f"invalid_checkpoint_manifest:{ref.locator}")
                continue
            manifests.append(manifest)
            for artifact in manifest.artifacts:
                if artifact.location is ArtifactLocation.REMOTE:
                    try:
                        payload_refs.append(
                            ExperimentPromotionArtifactRef(
                                scope=ExperimentPromotionArtifactScope.REMOTE,
                                locator=artifact.uri,
                                kind=ExperimentArtifactKind.OTHER,
                                size_bytes=artifact.size_bytes,
                                sha256=artifact.sha256,
                                content_id=artifact.name,
                                roles=(ExperimentPromotionArtifactRole.CHECKPOINT_PAYLOAD,),
                            )
                        )
                    except ValueError:
                        errors.append(f"unsafe_remote_checkpoint:{artifact.name}")
                    errors.append(f"unverified_remote_checkpoint:{artifact.name}")
                    continue
                candidate = Path(artifact.uri)
                path = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
                try:
                    locator = path.relative_to(self.root).as_posix()
                except ValueError:
                    errors.append(f"checkpoint_outside_project:{artifact.name}")
                    continue
                if not path.is_file():
                    errors.append(f"missing_checkpoint_payload:{artifact.name}")
                    continue
                payload = path.read_bytes()
                if (
                    len(payload) != artifact.size_bytes
                    or hashlib.sha256(payload).hexdigest() != artifact.sha256
                ):
                    errors.append(f"mismatched_checkpoint_payload:{artifact.name}")
                    continue
                payload_refs.append(
                    ExperimentPromotionArtifactRef(
                        scope=ExperimentPromotionArtifactScope.PROJECT,
                        locator=locator,
                        kind=ExperimentArtifactKind.OTHER,
                        size_bytes=len(payload),
                        sha256=artifact.sha256,
                        content_id=artifact.name,
                        roles=(ExperimentPromotionArtifactRole.CHECKPOINT_PAYLOAD,),
                    )
                )
        return manifests, payload_refs, errors

    def _reference_path(self, ref: ExperimentPromotionArtifactRef) -> Path:
        if ref.scope is ExperimentPromotionArtifactScope.REMOTE:
            raise ExperimentPromotionError("remote promotion artifact cannot be opened locally")
        base = (
            self.root if ref.scope is ExperimentPromotionArtifactScope.PROJECT else self.state_root
        )
        path = (base / ref.locator).resolve(strict=True)
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ExperimentPromotionError(
                "promotion reference escapes its declared root"
            ) from error
        return path

    @staticmethod
    def _manifest(
        *,
        promotion_name: str,
        index: ExperimentOperationsIndex,
        analysis: ExperimentAnalysis,
        plan: ExperimentPlan,
        report: ExperimentReport,
        trial_id: str,
        refs: tuple[ExperimentPromotionArtifactRef, ...],
    ) -> ExperimentReproducibilityManifest:
        trial = next(item for item in plan.trials if item.trial_id == trial_id)
        resolved = trial.resolved_config
        for field in ("model", "policy_version", "dataset_name"):
            if not isinstance(resolved.get(field), str) or not resolved[field]:
                raise ExperimentPromotionError(f"resolved config is missing {field}")
        manifest_id = ExperimentReproducibilityManifest.expected_manifest_id(
            promotion_name=promotion_name,
            index_id=index.index_id,
            analysis_id=analysis.analysis_id,
            plan_id=plan.plan_id,
            report_id=report.report_id,
            trial_id=trial_id,
            experiment_name=plan.name,
            config_digest=trial.config_digest,
            model=str(resolved["model"]),
            policy_version=str(resolved["policy_version"]),
            dataset_name=str(resolved["dataset_name"]),
            parameters=trial.parameters,
            artifacts=refs,
        )
        return ExperimentReproducibilityManifest(
            manifest_id=manifest_id,
            promotion_name=promotion_name,
            index_id=index.index_id,
            analysis_id=analysis.analysis_id,
            plan_id=plan.plan_id,
            report_id=report.report_id,
            trial_id=trial_id,
            experiment_name=plan.name,
            config_digest=trial.config_digest,
            model=str(resolved["model"]),
            policy_version=str(resolved["policy_version"]),
            dataset_name=str(resolved["dataset_name"]),
            parameters=trial.parameters,
            artifacts=refs,
            artifact_count=len(refs),
            total_size_bytes=sum(item.size_bytes for item in refs),
            checkpoint_ids=tuple(
                sorted(
                    item.content_id
                    for item in refs
                    if item.kind is ExperimentArtifactKind.CHECKPOINT_MANIFEST
                    and item.content_id is not None
                )
            ),
        )

    @staticmethod
    def _checks(
        *,
        index: ExperimentOperationsIndex,
        current_index: ExperimentOperationsIndex,
        selected: ExperimentRankedTrial,
        report: ExperimentReport,
        report_trial_state: ExperimentTrialState,
        policy: ExperimentPromotionPolicy,
        refs: tuple[ExperimentPromotionArtifactRef, ...],
        checkpoint_manifests: tuple[CheckpointManifest, ...],
        checkpoint_errors: tuple[str, ...],
    ) -> list[ExperimentPromotionCheck]:
        checks = []

        def add(code: str, passed: bool, detail: str, evidence: tuple[str, ...] = ()) -> None:
            if _CHECK_CODE.fullmatch(code) is None:
                raise RuntimeError("invalid internal promotion check code")
            checks.append(
                ExperimentPromotionCheck(
                    code=code,
                    passed=passed,
                    detail=detail,
                    evidence=tuple(sorted(set(evidence))),
                )
            )

        add(
            "index.current",
            current_index.index_id == index.index_id,
            "saved experiment index matches current validated state",
            (current_index.index_id, index.index_id),
        )
        add(
            "index.clean",
            not policy.require_clean_index or not current_index.issues,
            "experiment index satisfies the discovery-issue policy",
            tuple(item.path for item in current_index.issues),
        )
        add(
            "analysis.eligible",
            selected.eligible,
            "selected trial is eligible in the saved analysis",
        )
        allowed_state = report_trial_state is ExperimentTrialState.COMPLETE or (
            policy.allow_regression and report_trial_state is ExperimentTrialState.REGRESSION
        )
        add(
            "trial.state",
            allowed_state,
            "current trial state satisfies the promotion policy",
            (report_trial_state.value,),
        )
        add(
            "report.current",
            report.report_id == selected.report_id,
            "selected report identity matches current experiment evidence",
            (report.report_id, selected.report_id),
        )
        add(
            "ranking.pareto",
            not policy.require_pareto_front or selected.pareto_front,
            "selected trial satisfies the Pareto-front policy",
        )
        add(
            "ranking.maximum",
            policy.maximum_rank is None
            or (selected.rank is not None and selected.rank <= policy.maximum_rank),
            "selected trial satisfies the maximum-rank policy",
            (() if selected.rank is None else (str(selected.rank),)),
        )
        present_kinds = {item.kind for item in refs}
        missing_kinds = tuple(
            item.value for item in policy.required_artifact_kinds if item not in present_kinds
        )
        add(
            "artifacts.required",
            not missing_kinds,
            "all required artifact kinds are present",
            missing_kinds,
        )
        checkpoint_count_valid = (
            not policy.require_single_checkpoint or len(checkpoint_manifests) == 1
        )
        add(
            "checkpoint.count",
            checkpoint_count_valid,
            "checkpoint manifest count satisfies the promotion policy",
            tuple(item.checkpoint_id for item in checkpoint_manifests),
        )
        add(
            "checkpoint.verified",
            not policy.require_fully_verified_checkpoint or not checkpoint_errors,
            "checkpoint manifests and payloads satisfy local verification policy",
            checkpoint_errors,
        )
        lineage_digests = {
            value
            for ref in refs
            if ref.kind
            in {
                ExperimentArtifactKind.DATASET,
                ExperimentArtifactKind.DATASET_MANIFEST,
                ExperimentArtifactKind.DATASET_COLLECTION_MANIFEST,
            }
            for value in (ref.sha256, ref.content_id)
            if value is not None and re.fullmatch(r"[0-9a-f]{64}", value)
        }
        checkpoint_lineage = tuple(
            item.dataset_manifest_digest
            for item in checkpoint_manifests
            if item.dataset_manifest_digest is not None
        )
        lineage_valid = bool(checkpoint_manifests) and all(
            digest in lineage_digests for digest in checkpoint_lineage
        )
        if any(item.dataset_manifest_digest is None for item in checkpoint_manifests):
            lineage_valid = False
        add(
            "dataset.lineage",
            not policy.require_dataset_lineage or lineage_valid,
            "checkpoint dataset lineage is present in the promoted artifact graph",
            checkpoint_lineage,
        )
        return checks

    def _publish_sidecars(
        self,
        store: LocalBlobStore,
        index: ExperimentOperationsIndex,
        analysis: ExperimentAnalysis,
        preview: ExperimentPromotionPreview,
        record: ExperimentPromotionRecord,
        model_card: str,
    ) -> None:
        plan = self._load_plan(preview.plan_id)
        report = ExperimentRunner(self.root, self.state_root).report(plan)
        prefix = preview.promotion_name
        payloads = {
            "record.json": record.canonical_bytes() + b"\n",
            "manifest.json": preview.manifest.canonical_bytes() + b"\n",
            "model-card.md": model_card.encode("utf-8"),
            "plan.json": plan.canonical_bytes() + b"\n",
            "report.json": report.canonical_bytes() + b"\n",
            "index.json": index.canonical_bytes() + b"\n",
            "analysis.json": analysis.canonical_bytes() + b"\n",
        }
        for name, payload in payloads.items():
            store.put_if_absent(f"{prefix}/{name}", payload)


def render_experiment_model_card(
    index: ExperimentOperationsIndex,
    analysis: ExperimentAnalysis,
    preview: ExperimentPromotionPreview,
    operator: str,
    reason: str,
    approvers: tuple[str, ...],
) -> str:
    selected = next(
        item
        for item in analysis.trials
        if item.plan_id == preview.plan_id and item.trial_id == preview.trial_id
    )
    definitions = {item.metric_id: item for item in index.metric_definitions}
    lines = [
        "---",
        "library_name: agentic-rl-forge",
        "tags:",
        "  - agentic-rl",
        "  - reinforcement-learning",
        "  - search-agent",
        "---",
        "",
        f"# {_model_card_text(preview.promotion_name)}",
        "",
        "## Model and experiment",
        "",
        f"- Base model: {_model_card_code(preview.manifest.model)}",
        f"- Policy version: {_model_card_code(preview.manifest.policy_version)}",
        f"- Dataset: {_model_card_code(preview.manifest.dataset_name)}",
        f"- Experiment: {_model_card_code(preview.manifest.experiment_name)}",
        f"- Plan: {_model_card_code(preview.plan_id)}",
        f"- Trial: {_model_card_code(preview.trial_id)}",
        f"- Report: {_model_card_code(preview.report_id)}",
        f"- Reproducibility manifest: {_model_card_code(preview.manifest.manifest_id)}",
        "",
        "## Selection",
        "",
        f"- Rank: {selected.rank}",
        f"- Pareto front: {'yes' if selected.pareto_front else 'no'}",
        f"- Trial state: {_model_card_code(preview.state.value)}",
        "",
        "| Objective | Direction | Weight | Value |",
        "| --- | --- | ---: | ---: |",
    ]
    for objective in analysis.spec.objectives:
        definition = definitions[objective.metric_id]
        value = selected.objective_values.get(objective.metric_id)
        rendered = "—" if value is None else format(value, ".8g")
        lines.append(
            f"| {_model_card_text(definition.label)} | {objective.direction.value} | "
            f"{format(objective.weight, '.8g')} | {rendered} |"
        )
    lines.extend(
        [
            "",
            "## Promotion decision",
            "",
            f"- Operator: {_model_card_code(operator)}",
            f"- Approvers: {', '.join(_model_card_code(item) for item in approvers)}",
            f"- Rationale: {_model_card_text(reason)}",
            "",
            "## Reproducibility",
            "",
            f"The manifest references {preview.manifest.artifact_count} verified artifacts "
            f"({preview.manifest.total_size_bytes} bytes) without copying mutable payloads.",
            "Use `manifest.json` with the recorded project and experiment-state roots to locate "
            "each digest-bound artifact. `plan.json`, `report.json`, `index.json`, and "
            "`analysis.json` freeze the exact selection context.",
            "",
            "## Limitations",
            "",
            "- Promotion records engineering eligibility and human approval identities; it is not "
            "a cryptographic approval signature.",
            "- Weighted rank and Pareto membership do not establish statistical significance.",
            "- Remote checkpoint payloads are not fully verified unless independently fetched and "
            "checked against their recorded SHA-256 values.",
        ]
    )
    return "\n".join(lines) + "\n"


def _model_card_text(value: str) -> str:
    return (
        html.escape(value, quote=False)
        .replace("`", "&#96;")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def _model_card_code(value: str) -> str:
    return "`" + _model_card_text(value) + "`"

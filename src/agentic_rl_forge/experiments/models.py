from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from enum import Enum
from typing import TypeAlias

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts import ContractModel, JsonObject

ExperimentScalar: TypeAlias = str | int | float | bool

_PARAMETER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_STAGE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_CONFIG_PATH_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z_][A-Za-z0-9_-]*)*$")


class ExperimentArtifactKind(str, Enum):
    COLLECTION_CONFIG = "collection_config"
    DATASET = "dataset"
    DATASET_MANIFEST = "dataset_manifest"
    DATASET_COLLECTION_MANIFEST = "dataset_collection_manifest"
    ROLLOUT_PLAN = "rollout_plan"
    CHECKPOINT_MANIFEST = "checkpoint_manifest"
    TRAINER_BATCH_MANIFEST = "trainer_batch_manifest"
    BENCHMARK_REPORT = "benchmark_report"
    COMPARISON_REPORT = "comparison_report"
    OTHER = "other"


class ExperimentMetricStatistic(str, Enum):
    VALUE = "value"
    BASELINE_MEAN = "baseline_mean"
    CANDIDATE_MEAN = "candidate_mean"
    ABSOLUTE_DELTA = "absolute_delta"
    RELATIVE_DELTA = "relative_delta"
    CONFIDENCE_LOW = "confidence_low"
    CONFIDENCE_HIGH = "confidence_high"


class ExperimentArtifactTemplate(ContractModel):
    kind: ExperimentArtifactKind
    path: str = Field(min_length=1)
    required: bool = True

    @model_validator(mode="after")
    def validate_template(self) -> ExperimentArtifactTemplate:
        if "\x00" in self.path:
            raise ValueError("experiment artifact template cannot contain NUL")
        return self


class ExperimentMetricGateTemplate(ContractModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    artifact_path: str = Field(min_length=1)
    metric: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    statistic: ExperimentMetricStatistic = ExperimentMetricStatistic.VALUE
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> ExperimentMetricGateTemplate:
        if self.minimum is None and self.maximum is None:
            raise ValueError("experiment gate requires a minimum or maximum")
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("experiment gate minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("experiment gate maximum must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("experiment gate minimum cannot exceed its maximum")
        return self


class ExperimentStageTemplate(ContractModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    command: tuple[str, ...] = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    inputs: tuple[ExperimentArtifactTemplate, ...] = ()
    outputs: tuple[ExperimentArtifactTemplate, ...] = ()
    gates: tuple[ExperimentMetricGateTemplate, ...] = ()

    @model_validator(mode="after")
    def validate_stage(self) -> ExperimentStageTemplate:
        if any(not argument or "\x00" in argument for argument in self.command):
            raise ValueError("experiment stage command arguments must be nonempty and NUL-free")
        if len(self.depends_on) != len(set(self.depends_on)) or self.name in self.depends_on:
            raise ValueError("experiment stage dependencies must be unique and cannot be self")
        output_paths = {item.path: item.kind for item in self.outputs}
        if len(output_paths) != len(self.outputs):
            raise ValueError("experiment stage output templates must be unique")
        gate_names = [item.name for item in self.gates]
        if len(gate_names) != len(set(gate_names)):
            raise ValueError("experiment stage gate names must be unique")
        for gate in self.gates:
            kind = output_paths.get(gate.artifact_path)
            if kind not in {
                ExperimentArtifactKind.BENCHMARK_REPORT,
                ExperimentArtifactKind.COMPARISON_REPORT,
            }:
                raise ValueError("experiment gate must reference a benchmark output")
            if (
                kind is ExperimentArtifactKind.BENCHMARK_REPORT
                and gate.statistic is not ExperimentMetricStatistic.VALUE
            ):
                raise ValueError("benchmark report gates support only the value statistic")
            if (
                kind is ExperimentArtifactKind.COMPARISON_REPORT
                and gate.statistic is ExperimentMetricStatistic.VALUE
            ):
                raise ValueError("comparison report gates require a comparison statistic")
        return self


class ExperimentMatrixSpec(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    base_config: str = Field(min_length=1)
    fixed_parameters: dict[str, ExperimentScalar] = Field(default_factory=dict)
    axes: dict[str, tuple[ExperimentScalar, ...]] = Field(min_length=1)
    exclude: tuple[dict[str, ExperimentScalar], ...] = ()
    config_bindings: dict[str, str] = Field(default_factory=dict)
    stages: tuple[ExperimentStageTemplate, ...] = Field(min_length=1)
    max_trials: int = Field(default=128, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_spec(self) -> ExperimentMatrixSpec:
        if "\x00" in self.base_config:
            raise ValueError("experiment base config path cannot contain NUL")
        parameter_names = set(self.fixed_parameters) | set(self.axes)
        if set(self.fixed_parameters) & set(self.axes):
            raise ValueError("fixed and matrix parameters must be disjoint")
        if any(_PARAMETER_PATTERN.fullmatch(name) is None for name in parameter_names):
            raise ValueError("experiment parameter names are invalid")
        for name, values in self.axes.items():
            if not values:
                raise ValueError(f"experiment axis {name!r} cannot be empty")
            identities = [self._scalar_identity(value) for value in values]
            if len(identities) != len(set(identities)):
                raise ValueError(f"experiment axis {name!r} contains duplicate values")
        axis_values = tuple(value for values in self.axes.values() for value in values)
        for value in (*self.fixed_parameters.values(), *axis_values):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("experiment parameter values must be finite")
        for selector in self.exclude:
            if not selector or not set(selector) <= parameter_names:
                raise ValueError("experiment exclusion selectors must use known parameters")
        for config_path, parameter_name in self.config_bindings.items():
            if _CONFIG_PATH_PATTERN.fullmatch(config_path) is None:
                raise ValueError("experiment config binding path is invalid")
            if parameter_name not in parameter_names:
                raise ValueError("experiment config binding uses an unknown parameter")
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("experiment stage names must be unique")
        completed: set[str] = set()
        for stage in self.stages:
            if not set(stage.depends_on) <= completed:
                raise ValueError("experiment stage dependencies must refer to earlier stages")
            completed.add(stage.name)
        return self

    @staticmethod
    def _scalar_identity(value: ExperimentScalar) -> bytes:
        return orjson.dumps(
            {"type": type(value).__name__, "value": value},
            option=orjson.OPT_SORT_KEYS,
        )


class ExperimentArtifactPlan(ContractModel):
    kind: ExperimentArtifactKind
    path: str = Field(min_length=1)
    required: bool = True

    @model_validator(mode="after")
    def validate_path(self) -> ExperimentArtifactPlan:
        if self.path != "{config_path}":
            parts = self.path.replace("\\", "/").split("/")
            if (
                self.path.startswith(("/", "\\"))
                or re.match(r"^[A-Za-z]:", self.path)
                or any(part in {"", ".", ".."} for part in parts)
                or "\x00" in self.path
            ):
                raise ValueError("experiment artifact paths must be safe relative paths")
        return self


class ExperimentMetricGate(ContractModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    artifact_path: str = Field(min_length=1)
    metric: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    statistic: ExperimentMetricStatistic
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_gate(self) -> ExperimentMetricGate:
        ExperimentMetricGateTemplate(
            name=self.name,
            artifact_path=self.artifact_path,
            metric=self.metric,
            statistic=self.statistic,
            minimum=self.minimum,
            maximum=self.maximum,
        )
        ExperimentArtifactPlan(kind=ExperimentArtifactKind.OTHER, path=self.artifact_path)
        return self


class ExperimentStagePlan(ContractModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    command: tuple[str, ...] = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    inputs: tuple[ExperimentArtifactPlan, ...] = ()
    outputs: tuple[ExperimentArtifactPlan, ...] = ()
    gates: tuple[ExperimentMetricGate, ...] = ()

    @model_validator(mode="after")
    def validate_plan(self) -> ExperimentStagePlan:
        if any(not argument or "\x00" in argument for argument in self.command):
            raise ValueError("experiment command arguments must be nonempty and NUL-free")
        if len(self.depends_on) != len(set(self.depends_on)) or self.name in self.depends_on:
            raise ValueError("experiment dependencies must be unique and cannot be self")
        output_paths = {item.path: item.kind for item in self.outputs}
        if len(output_paths) != len(self.outputs):
            raise ValueError("experiment output paths must be unique")
        for gate in self.gates:
            kind = output_paths.get(gate.artifact_path)
            if kind not in {
                ExperimentArtifactKind.BENCHMARK_REPORT,
                ExperimentArtifactKind.COMPARISON_REPORT,
            }:
                raise ValueError("experiment gate must reference a benchmark output")
            if (
                kind is ExperimentArtifactKind.BENCHMARK_REPORT
                and gate.statistic is not ExperimentMetricStatistic.VALUE
            ):
                raise ValueError("benchmark report gates support only the value statistic")
            if (
                kind is ExperimentArtifactKind.COMPARISON_REPORT
                and gate.statistic is ExperimentMetricStatistic.VALUE
            ):
                raise ValueError("comparison report gates require a comparison statistic")
        return self


class ExperimentTrialPlan(ContractModel):
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    parameters: dict[str, ExperimentScalar]
    resolved_config: JsonObject
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stages: tuple[ExperimentStagePlan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_trial(self) -> ExperimentTrialPlan:
        if tuple(self.parameters) != tuple(sorted(self.parameters)):
            raise ValueError("experiment trial parameters must be sorted")
        digest = hashlib.sha256(
            orjson.dumps(self.resolved_config, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()
        if self.config_digest != digest:
            raise ValueError("experiment resolved config digest does not match its contents")
        completed: set[str] = set()
        for stage in self.stages:
            if not set(stage.depends_on) <= completed:
                raise ValueError("experiment trial dependencies must refer to earlier stages")
            completed.add(stage.name)
        return self

    @staticmethod
    def expected_trial_id(
        *,
        experiment_name: str,
        parameters: dict[str, ExperimentScalar],
        config_digest: str,
    ) -> str:
        payload = orjson.dumps(
            {
                "name": experiment_name,
                "parameters": parameters,
                "config_digest": config_digest,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_trial_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPlan(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trials: tuple[ExperimentTrialPlan, ...] = Field(min_length=1)
    trial_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_experiment(self) -> ExperimentPlan:
        trial_ids = tuple(trial.trial_id for trial in self.trials)
        if trial_ids != tuple(sorted(set(trial_ids))):
            raise ValueError("experiment trial IDs must be sorted and unique")
        if self.trial_count != len(self.trials):
            raise ValueError("experiment trial count does not match its trials")
        output_owners: dict[str, str] = {}
        for trial in self.trials:
            expected_trial_id = ExperimentTrialPlan.expected_trial_id(
                experiment_name=self.name,
                parameters=trial.parameters,
                config_digest=trial.config_digest,
            )
            if trial.trial_id != expected_trial_id:
                raise ValueError("experiment trial ID does not match its contents")
            for stage in trial.stages:
                for artifact in stage.outputs:
                    owner = f"{trial.trial_id}:{stage.name}"
                    existing = output_owners.setdefault(artifact.path, owner)
                    if existing != owner:
                        raise ValueError("experiment trials cannot share an output path")
        expected = self.expected_plan_id(
            name=self.name,
            source_spec_sha256=self.source_spec_sha256,
            base_config_sha256=self.base_config_sha256,
            trials=self.trials,
        )
        if self.plan_id != expected:
            raise ValueError("experiment plan ID does not match its contents")
        return self

    @staticmethod
    def expected_plan_id(
        *,
        name: str,
        source_spec_sha256: str,
        base_config_sha256: str,
        trials: tuple[ExperimentTrialPlan, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "name": name,
                "source_spec_sha256": source_spec_sha256,
                "base_config_sha256": base_config_sha256,
                "trials": [trial.model_dump(mode="json") for trial in trials],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_plan_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentArtifactEvidence(ContractModel):
    kind: ExperimentArtifactKind
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_id: str | None = None
    summary: JsonObject = Field(default_factory=dict)


class ExperimentStageRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    record_id: str = Field(pattern=r"^experiment_stage_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    stage_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    command: tuple[str, ...] = Field(min_length=1)
    inputs: tuple[ExperimentArtifactEvidence, ...]
    outputs: tuple[ExperimentArtifactEvidence, ...]
    started_at: datetime
    completed_at: datetime
    stdout_path: str = Field(min_length=1)
    stderr_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_record(self) -> ExperimentStageRecord:
        if (
            self.started_at.tzinfo is None
            or self.started_at.utcoffset() is None
            or self.completed_at.tzinfo is None
            or self.completed_at.utcoffset() is None
        ):
            raise ValueError("experiment stage times must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("experiment stage completion cannot precede its start")
        expected = self.expected_record_id(
            plan_id=self.plan_id,
            trial_id=self.trial_id,
            stage_name=self.stage_name,
            command=self.command,
            inputs=self.inputs,
            outputs=self.outputs,
        )
        if self.record_id != expected:
            raise ValueError("experiment stage record ID does not match its contents")
        return self

    @staticmethod
    def expected_record_id(
        *,
        plan_id: str,
        trial_id: str,
        stage_name: str,
        command: tuple[str, ...],
        inputs: tuple[ExperimentArtifactEvidence, ...],
        outputs: tuple[ExperimentArtifactEvidence, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "plan_id": plan_id,
                "trial_id": trial_id,
                "stage_name": stage_name,
                "command": command,
                "inputs": [item.model_dump(mode="json") for item in inputs],
                "outputs": [item.model_dump(mode="json") for item in outputs],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_stage_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentStageAttempt(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    attempt_id: str = Field(pattern=r"^experiment_attempt_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    stage_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    command: tuple[str, ...] = Field(min_length=1)
    inputs: tuple[ExperimentArtifactEvidence, ...]
    return_code: int
    detail: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime
    stdout_path: str = Field(min_length=1)
    stderr_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_attempt(self) -> ExperimentStageAttempt:
        if (
            self.started_at.tzinfo is None
            or self.started_at.utcoffset() is None
            or self.completed_at.tzinfo is None
            or self.completed_at.utcoffset() is None
        ):
            raise ValueError("experiment attempt times must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("experiment attempt completion cannot precede its start")
        expected = self.expected_attempt_id(
            plan_id=self.plan_id,
            trial_id=self.trial_id,
            stage_name=self.stage_name,
            command=self.command,
            inputs=self.inputs,
            return_code=self.return_code,
            detail=self.detail,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )
        if self.attempt_id != expected:
            raise ValueError("experiment attempt ID does not match its contents")
        return self

    @staticmethod
    def expected_attempt_id(
        *,
        plan_id: str,
        trial_id: str,
        stage_name: str,
        command: tuple[str, ...],
        inputs: tuple[ExperimentArtifactEvidence, ...],
        return_code: int,
        detail: str,
        started_at: datetime,
        completed_at: datetime,
    ) -> str:
        payload = orjson.dumps(
            {
                "plan_id": plan_id,
                "trial_id": trial_id,
                "stage_name": stage_name,
                "command": command,
                "inputs": [item.model_dump(mode="json") for item in inputs],
                "return_code": return_code,
                "detail": detail,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_attempt_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentStageState(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"


class ExperimentGateResult(ContractModel):
    name: str
    artifact_path: str
    metric: str
    statistic: ExperimentMetricStatistic
    actual: float | None
    minimum: float | None = None
    maximum: float | None = None
    passed: bool
    detail: str

    @model_validator(mode="after")
    def validate_result(self) -> ExperimentGateResult:
        actual = self.actual
        if actual is not None and not math.isfinite(actual):
            raise ValueError("experiment gate actual value must be finite")
        expected = actual is not None
        if actual is not None and self.minimum is not None and actual < self.minimum:
            expected = False
        if actual is not None and self.maximum is not None and actual > self.maximum:
            expected = False
        if self.passed != expected:
            raise ValueError("experiment gate result is inconsistent")
        return self


class ExperimentStageSummary(ContractModel):
    name: str
    state: ExperimentStageState
    record_id: str | None = None
    latest_attempt_id: str | None = None
    detail: str
    inputs: tuple[ExperimentArtifactEvidence, ...] = ()
    outputs: tuple[ExperimentArtifactEvidence, ...] = ()
    gates: tuple[ExperimentGateResult, ...] = ()

    @model_validator(mode="after")
    def validate_summary(self) -> ExperimentStageSummary:
        if self.state is ExperimentStageState.SUCCEEDED:
            if self.record_id is None or self.latest_attempt_id is not None:
                raise ValueError("successful experiment stage summary is inconsistent")
        elif self.state is ExperimentStageState.FAILED:
            if self.latest_attempt_id is None or self.record_id is not None:
                raise ValueError("failed experiment stage summary is inconsistent")
        elif self.state in {ExperimentStageState.PENDING, ExperimentStageState.BLOCKED} and (
            self.record_id is not None
            or self.latest_attempt_id is not None
            or self.inputs
            or self.outputs
            or self.gates
        ):
            raise ValueError("inactive experiment stage summary contains evidence")
        if self.gates and self.state is not ExperimentStageState.SUCCEEDED:
            raise ValueError("only successful experiment stages can contain gate results")
        return self


class ExperimentTrialState(str, Enum):
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    REGRESSION = "regression"
    COMPLETE = "complete"


class ExperimentTrialSummary(ContractModel):
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    parameters: dict[str, ExperimentScalar]
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stages: tuple[ExperimentStageSummary, ...]
    state: ExperimentTrialState

    @model_validator(mode="after")
    def validate_summary(self) -> ExperimentTrialSummary:
        if tuple(self.parameters) != tuple(sorted(self.parameters)):
            raise ValueError("experiment report parameters must be sorted")
        failed = any(
            stage.state in {ExperimentStageState.FAILED, ExperimentStageState.INVALID}
            for stage in self.stages
        )
        complete = bool(self.stages) and all(
            stage.state is ExperimentStageState.SUCCEEDED for stage in self.stages
        )
        regressed = any(not gate.passed for stage in self.stages for gate in stage.gates)
        expected = (
            ExperimentTrialState.FAILED
            if failed
            else (
                ExperimentTrialState.REGRESSION
                if complete and regressed
                else (
                    ExperimentTrialState.COMPLETE if complete else ExperimentTrialState.INCOMPLETE
                )
            )
        )
        if self.state is not expected:
            raise ValueError("experiment trial report state is inconsistent")
        return self


class ExperimentReport(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    report_id: str = Field(pattern=r"^experiment_report_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    name: str
    trials: tuple[ExperimentTrialSummary, ...]
    trial_count: int = Field(ge=1)
    state_counts: dict[str, int]
    gate_count: int = Field(ge=0)
    failed_gate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_report(self) -> ExperimentReport:
        trial_ids = tuple(trial.trial_id for trial in self.trials)
        if trial_ids != tuple(sorted(set(trial_ids))):
            raise ValueError("experiment report trials must be sorted and unique")
        if self.trial_count != len(self.trials):
            raise ValueError("experiment report trial count is inconsistent")
        expected_counts = {
            state.value: sum(trial.state is state for trial in self.trials)
            for state in ExperimentTrialState
        }
        if self.state_counts != expected_counts:
            raise ValueError("experiment report state counts are inconsistent")
        gates = [gate for trial in self.trials for stage in trial.stages for gate in stage.gates]
        if self.gate_count != len(gates) or self.failed_gate_count != sum(
            not gate.passed for gate in gates
        ):
            raise ValueError("experiment report gate counts are inconsistent")
        expected_id = self.expected_report_id(
            plan_id=self.plan_id,
            name=self.name,
            trials=self.trials,
        )
        if self.report_id != expected_id:
            raise ValueError("experiment report ID does not match its contents")
        return self

    @staticmethod
    def expected_report_id(
        *,
        plan_id: str,
        name: str,
        trials: tuple[ExperimentTrialSummary, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "plan_id": plan_id,
                "name": name,
                "trials": [trial.model_dump(mode="json") for trial in trials],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_report_{hashlib.sha256(payload).hexdigest()[:24]}"

from __future__ import annotations

import csv
import hashlib
import html
import io
import math
import re
from collections.abc import Iterable
from enum import Enum
from itertools import pairwise
from pathlib import Path

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts import ContractModel
from agentic_rl_forge.evaluation import BenchmarkComparisonReport, BenchmarkReport
from agentic_rl_forge.experiments.models import (
    ExperimentArtifactKind,
    ExperimentMetricStatistic,
    ExperimentPlan,
    ExperimentReport,
    ExperimentScalar,
    ExperimentStageState,
    ExperimentTrialState,
)
from agentic_rl_forge.experiments.runner import ExperimentRunner

_PLAN_PATTERN = re.compile(r"^experiment_plan_[0-9a-f]{24}$")
_METRIC_PATTERN = re.compile(r"^experiment_metric_[0-9a-f]{24}$")


class ExperimentIndexIssue(ContractModel):
    path: str = Field(min_length=1)
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> ExperimentIndexIssue:
        parts = self.path.replace("\\", "/").split("/")
        if self.path.startswith(("/", "\\")) or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("experiment index issue paths must be safe relative paths")
        return self


class ExperimentReportReference(ContractModel):
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    report_id: str = Field(pattern=r"^experiment_report_[0-9a-f]{24}$")
    name: str = Field(min_length=1)
    trial_count: int = Field(ge=1)
    state_counts: dict[str, int]
    gate_count: int = Field(ge=0)
    failed_gate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> ExperimentReportReference:
        expected_keys = {state.value for state in ExperimentTrialState}
        if set(self.state_counts) != expected_keys or any(
            value < 0 for value in self.state_counts.values()
        ):
            raise ValueError("experiment report reference state counts are invalid")
        if sum(self.state_counts.values()) != self.trial_count:
            raise ValueError("experiment report reference trial count is inconsistent")
        if self.failed_gate_count > self.gate_count:
            raise ValueError("experiment report reference gate counts are inconsistent")
        return self


class ExperimentMetricDefinition(ContractModel):
    metric_id: str = Field(pattern=r"^experiment_metric_[0-9a-f]{24}$")
    experiment_name: str = Field(min_length=1)
    stage_name: str = Field(min_length=1)
    output_index: int = Field(ge=0)
    artifact_kind: ExperimentArtifactKind
    metric: str = Field(min_length=1)
    statistic: ExperimentMetricStatistic
    unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_definition(self) -> ExperimentMetricDefinition:
        if self.artifact_kind is ExperimentArtifactKind.BENCHMARK_REPORT:
            if self.statistic is not ExperimentMetricStatistic.VALUE:
                raise ValueError("benchmark metric definitions support only value statistics")
        elif self.artifact_kind is ExperimentArtifactKind.COMPARISON_REPORT:
            if self.statistic is ExperimentMetricStatistic.VALUE:
                raise ValueError("comparison metric definitions require comparison statistics")
        else:
            raise ValueError("experiment metric definitions require benchmark artifacts")
        expected = self.expected_metric_id(
            experiment_name=self.experiment_name,
            stage_name=self.stage_name,
            output_index=self.output_index,
            artifact_kind=self.artifact_kind,
            metric=self.metric,
            statistic=self.statistic,
            unit=self.unit,
        )
        if self.metric_id != expected:
            raise ValueError("experiment metric ID does not match its definition")
        return self

    @staticmethod
    def expected_metric_id(
        *,
        experiment_name: str,
        stage_name: str,
        output_index: int,
        artifact_kind: ExperimentArtifactKind,
        metric: str,
        statistic: ExperimentMetricStatistic,
        unit: str,
    ) -> str:
        payload = orjson.dumps(
            {
                "experiment_name": experiment_name,
                "stage_name": stage_name,
                "output_index": output_index,
                "artifact_kind": artifact_kind.value,
                "metric": metric,
                "statistic": statistic.value,
                "unit": unit,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_metric_{hashlib.sha256(payload).hexdigest()[:24]}"

    @property
    def label(self) -> str:
        return f"{self.stage_name}[{self.output_index}].{self.metric}.{self.statistic.value}"


class ExperimentIndexRow(ContractModel):
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    report_id: str = Field(pattern=r"^experiment_report_[0-9a-f]{24}$")
    experiment_name: str = Field(min_length=1)
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    state: ExperimentTrialState
    parameters: dict[str, ExperimentScalar]
    metric_values: dict[str, float]

    @model_validator(mode="after")
    def validate_row(self) -> ExperimentIndexRow:
        if tuple(self.parameters) != tuple(sorted(self.parameters)):
            raise ValueError("experiment index row parameters must be sorted")
        if tuple(self.metric_values) != tuple(sorted(self.metric_values)):
            raise ValueError("experiment index row metrics must be sorted")
        if any(_METRIC_PATTERN.fullmatch(key) is None for key in self.metric_values):
            raise ValueError("experiment index row contains an invalid metric ID")
        if any(not math.isfinite(value) for value in self.metric_values.values()):
            raise ValueError("experiment index metric values must be finite")
        return self


class ExperimentOperationsIndex(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    index_id: str = Field(pattern=r"^experiment_index_[0-9a-f]{24}$")
    reports: tuple[ExperimentReportReference, ...]
    rows: tuple[ExperimentIndexRow, ...]
    parameter_columns: tuple[str, ...]
    metric_definitions: tuple[ExperimentMetricDefinition, ...]
    state_counts: dict[str, int]
    issues: tuple[ExperimentIndexIssue, ...] = ()

    @model_validator(mode="after")
    def validate_index(self) -> ExperimentOperationsIndex:
        report_ids = tuple(item.plan_id for item in self.reports)
        if report_ids != tuple(sorted(set(report_ids))):
            raise ValueError("experiment index reports must be sorted and unique")
        row_keys = tuple((item.plan_id, item.trial_id) for item in self.rows)
        if row_keys != tuple(sorted(set(row_keys))):
            raise ValueError("experiment index rows must be sorted and unique")
        metric_ids = tuple(item.metric_id for item in self.metric_definitions)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("experiment index metric definitions must be sorted and unique")
        if self.parameter_columns != tuple(sorted(set(self.parameter_columns))):
            raise ValueError("experiment index parameter columns must be sorted and unique")
        issue_keys = tuple((item.path, item.detail) for item in self.issues)
        if issue_keys != tuple(sorted(set(issue_keys))):
            raise ValueError("experiment index issues must be sorted and unique")

        reports = {item.plan_id: item for item in self.reports}
        definitions = set(metric_ids)
        for row in self.rows:
            report = reports.get(row.plan_id)
            if report is None or report.report_id != row.report_id:
                raise ValueError("experiment index row does not reference its report")
            if report.name != row.experiment_name:
                raise ValueError("experiment index row name differs from its report")
            if not set(row.metric_values) <= definitions:
                raise ValueError("experiment index row references an unknown metric")
        for plan_id, report in reports.items():
            plan_rows = [row for row in self.rows if row.plan_id == plan_id]
            if len(plan_rows) != report.trial_count:
                raise ValueError("experiment index report trial count differs from its rows")
            plan_counts = {
                state.value: sum(row.state is state for row in plan_rows)
                for state in ExperimentTrialState
            }
            if plan_counts != report.state_counts:
                raise ValueError("experiment index report states differ from its rows")
        expected_parameters = tuple(sorted({key for row in self.rows for key in row.parameters}))
        if self.parameter_columns != expected_parameters:
            raise ValueError("experiment index parameter columns are inconsistent")
        expected_counts = {
            state.value: sum(row.state is state for row in self.rows)
            for state in ExperimentTrialState
        }
        if self.state_counts != expected_counts:
            raise ValueError("experiment index state counts are inconsistent")
        expected_id = self.expected_index_id(
            reports=self.reports,
            rows=self.rows,
            parameter_columns=self.parameter_columns,
            metric_definitions=self.metric_definitions,
            issues=self.issues,
        )
        if self.index_id != expected_id:
            raise ValueError("experiment index ID does not match its contents")
        return self

    @staticmethod
    def expected_index_id(
        *,
        reports: tuple[ExperimentReportReference, ...],
        rows: tuple[ExperimentIndexRow, ...],
        parameter_columns: tuple[str, ...],
        metric_definitions: tuple[ExperimentMetricDefinition, ...],
        issues: tuple[ExperimentIndexIssue, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "reports": [item.model_dump(mode="json") for item in reports],
                "rows": [item.model_dump(mode="json") for item in rows],
                "parameter_columns": parameter_columns,
                "metric_definitions": [item.model_dump(mode="json") for item in metric_definitions],
                "issues": [item.model_dump(mode="json") for item in issues],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_index_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentObjectiveDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ExperimentObjective(ContractModel):
    metric_id: str = Field(pattern=r"^experiment_metric_[0-9a-f]{24}$")
    direction: ExperimentObjectiveDirection
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_weight(self) -> ExperimentObjective:
        if not math.isfinite(self.weight):
            raise ValueError("experiment objective weight must be finite")
        return self


class ExperimentRankingSpec(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    objectives: tuple[ExperimentObjective, ...] = Field(min_length=1)
    eligible_states: tuple[ExperimentTrialState, ...] = (ExperimentTrialState.COMPLETE,)
    experiment_names: tuple[str, ...] = ()
    plan_ids: tuple[str, ...] = ()
    baseline_plan_id: str | None = Field(
        default=None,
        pattern=r"^experiment_plan_[0-9a-f]{24}$",
    )
    baseline_trial_id: str | None = Field(
        default=None,
        pattern=r"^experiment_trial_[0-9a-f]{24}$",
    )
    max_candidates: int = Field(default=10_000, ge=1, le=50_000)

    @model_validator(mode="after")
    def validate_spec(self) -> ExperimentRankingSpec:
        objective_ids = tuple(item.metric_id for item in self.objectives)
        if objective_ids != tuple(sorted(set(objective_ids))):
            raise ValueError("experiment ranking objectives must be sorted and unique")
        if self.eligible_states != tuple(
            sorted(set(self.eligible_states), key=lambda item: item.value)
        ):
            raise ValueError("experiment ranking states must be sorted and unique")
        allowed = {ExperimentTrialState.COMPLETE, ExperimentTrialState.REGRESSION}
        if not set(self.eligible_states) <= allowed:
            raise ValueError("experiment ranking can include only completed trial states")
        if self.experiment_names != tuple(sorted(set(self.experiment_names))):
            raise ValueError("experiment ranking names must be sorted and unique")
        if self.plan_ids != tuple(sorted(set(self.plan_ids))):
            raise ValueError("experiment ranking plan IDs must be sorted and unique")
        if (self.baseline_plan_id is None) != (self.baseline_trial_id is None):
            raise ValueError("experiment ranking baseline requires both plan and trial IDs")
        return self


class ExperimentBaselineDelta(ContractModel):
    metric_id: str = Field(pattern=r"^experiment_metric_[0-9a-f]{24}$")
    baseline_value: float
    candidate_value: float
    raw_delta: float
    improvement: float

    @model_validator(mode="after")
    def validate_delta(self) -> ExperimentBaselineDelta:
        values = (
            self.baseline_value,
            self.candidate_value,
            self.raw_delta,
            self.improvement,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("experiment baseline deltas must be finite")
        if not math.isclose(
            self.raw_delta,
            self.candidate_value - self.baseline_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("experiment baseline raw delta is inconsistent")
        return self


class ExperimentRankedTrial(ContractModel):
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    report_id: str = Field(pattern=r"^experiment_report_[0-9a-f]{24}$")
    experiment_name: str = Field(min_length=1)
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")
    state: ExperimentTrialState
    parameters: dict[str, ExperimentScalar]
    objective_values: dict[str, float]
    eligible: bool
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None
    pareto_front: bool = False
    baseline_deltas: tuple[ExperimentBaselineDelta, ...] = ()
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ranked_trial(self) -> ExperimentRankedTrial:
        if tuple(self.parameters) != tuple(sorted(self.parameters)):
            raise ValueError("ranked experiment parameters must be sorted")
        if tuple(self.objective_values) != tuple(sorted(self.objective_values)):
            raise ValueError("ranked experiment objective values must be sorted")
        if any(not math.isfinite(value) for value in self.objective_values.values()):
            raise ValueError("ranked experiment objective values must be finite")
        delta_ids = tuple(item.metric_id for item in self.baseline_deltas)
        if delta_ids != tuple(sorted(set(delta_ids))):
            raise ValueError("experiment baseline deltas must be sorted and unique")
        if self.eligible:
            if self.rank is None or self.score is None or not math.isfinite(self.score):
                raise ValueError("eligible experiment trial requires a finite rank and score")
        elif (
            self.rank is not None
            or self.score is not None
            or self.pareto_front
            or self.baseline_deltas
        ):
            raise ValueError("ineligible experiment trial contains ranking evidence")
        return self


class ExperimentTrialReference(ContractModel):
    plan_id: str = Field(pattern=r"^experiment_plan_[0-9a-f]{24}$")
    trial_id: str = Field(pattern=r"^experiment_trial_[0-9a-f]{24}$")


class ExperimentAnalysis(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    analysis_id: str = Field(pattern=r"^experiment_analysis_[0-9a-f]{24}$")
    index_id: str = Field(pattern=r"^experiment_index_[0-9a-f]{24}$")
    spec: ExperimentRankingSpec
    trials: tuple[ExperimentRankedTrial, ...]
    eligible_count: int = Field(ge=1)
    ineligible_count: int = Field(ge=0)
    pareto_front: tuple[ExperimentTrialReference, ...]

    @model_validator(mode="after")
    def validate_analysis(self) -> ExperimentAnalysis:
        keys = tuple((item.plan_id, item.trial_id) for item in self.trials)
        if len(keys) != len(set(keys)):
            raise ValueError("experiment analysis trials must be unique")
        eligible = [item for item in self.trials if item.eligible]
        ineligible = [item for item in self.trials if not item.eligible]
        if self.eligible_count != len(eligible) or self.ineligible_count != len(ineligible):
            raise ValueError("experiment analysis counts are inconsistent")
        if [item.rank for item in eligible] != list(range(1, len(eligible) + 1)):
            raise ValueError("experiment analysis ranks must be consecutive")
        if len(eligible) > self.spec.max_candidates:
            raise ValueError("experiment analysis exceeds its maximum candidate count")
        objective_ids = tuple(item.metric_id for item in self.spec.objectives)
        objective_set = set(objective_ids)
        for item in self.trials:
            if (
                self.spec.experiment_names
                and item.experiment_name not in self.spec.experiment_names
            ):
                raise ValueError("experiment analysis contains an unselected experiment")
            if self.spec.plan_ids and item.plan_id not in self.spec.plan_ids:
                raise ValueError("experiment analysis contains an unselected plan")
            if not set(item.objective_values) <= objective_set:
                raise ValueError("experiment analysis contains an unknown objective value")
            expected_eligible = (
                item.state in self.spec.eligible_states
                and set(item.objective_values) == objective_set
            )
            if item.eligible != expected_eligible:
                raise ValueError("experiment analysis eligibility is inconsistent")
        self._validate_scores(eligible)
        self._validate_baseline_deltas(eligible)
        self._validate_pareto_front(eligible)
        scores = [item.score for item in eligible]
        if any(
            first is not None and second is not None and first < second
            for first, second in pairwise(scores)
        ):
            raise ValueError("experiment analysis scores must be descending")
        expected_front = tuple(
            ExperimentTrialReference(plan_id=item.plan_id, trial_id=item.trial_id)
            for item in sorted(
                (item for item in eligible if item.pareto_front),
                key=lambda item: (item.plan_id, item.trial_id),
            )
        )
        if self.pareto_front != expected_front:
            raise ValueError("experiment analysis Pareto front is inconsistent")
        if self.trials != tuple(
            eligible + sorted(ineligible, key=lambda item: (item.plan_id, item.trial_id))
        ):
            raise ValueError("experiment analysis trials are not canonically ordered")
        expected_id = self.expected_analysis_id(
            index_id=self.index_id,
            spec=self.spec,
            trials=self.trials,
        )
        if self.analysis_id != expected_id:
            raise ValueError("experiment analysis ID does not match its contents")
        return self

    def _validate_scores(self, eligible: list[ExperimentRankedTrial]) -> None:
        total_weight = sum(item.weight for item in self.spec.objectives)
        ranges: dict[str, tuple[float, float]] = {}
        for objective in self.spec.objectives:
            aligned_values = [
                item.objective_values[objective.metric_id]
                if objective.direction is ExperimentObjectiveDirection.MAXIMIZE
                else -item.objective_values[objective.metric_id]
                for item in eligible
            ]
            ranges[objective.metric_id] = (min(aligned_values), max(aligned_values))
        for item in eligible:
            expected_score = 0.0
            for objective in self.spec.objectives:
                value = item.objective_values[objective.metric_id]
                aligned_value = (
                    value
                    if objective.direction is ExperimentObjectiveDirection.MAXIMIZE
                    else -value
                )
                low, high = ranges[objective.metric_id]
                normalized = (
                    1.0 if math.isclose(low, high) else (aligned_value - low) / (high - low)
                )
                expected_score += objective.weight * normalized
            expected_score /= total_weight
            if item.score is None or not math.isclose(
                item.score,
                expected_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("experiment analysis score is inconsistent")

    def _validate_baseline_deltas(self, eligible: list[ExperimentRankedTrial]) -> None:
        if self.spec.baseline_plan_id is None or self.spec.baseline_trial_id is None:
            if any(item.baseline_deltas for item in eligible):
                raise ValueError("experiment analysis has deltas without a baseline")
            return
        baseline = next(
            (
                item
                for item in eligible
                if item.plan_id == self.spec.baseline_plan_id
                and item.trial_id == self.spec.baseline_trial_id
            ),
            None,
        )
        if baseline is None:
            raise ValueError("experiment analysis baseline is not eligible")
        objectives = {item.metric_id: item for item in self.spec.objectives}
        expected_ids = tuple(sorted(objectives))
        for item in eligible:
            if tuple(delta.metric_id for delta in item.baseline_deltas) != expected_ids:
                raise ValueError("experiment analysis baseline deltas are incomplete")
            for delta in item.baseline_deltas:
                objective = objectives[delta.metric_id]
                baseline_value = baseline.objective_values[delta.metric_id]
                candidate_value = item.objective_values[delta.metric_id]
                raw_delta = candidate_value - baseline_value
                improvement = (
                    raw_delta
                    if objective.direction is ExperimentObjectiveDirection.MAXIMIZE
                    else -raw_delta
                )
                if not all(
                    (
                        math.isclose(delta.baseline_value, baseline_value),
                        math.isclose(delta.candidate_value, candidate_value),
                        math.isclose(delta.raw_delta, raw_delta),
                        math.isclose(delta.improvement, improvement),
                    )
                ):
                    raise ValueError("experiment analysis baseline delta is inconsistent")

    def _validate_pareto_front(self, eligible: list[ExperimentRankedTrial]) -> None:
        def aligned(item: ExperimentRankedTrial, objective: ExperimentObjective) -> float:
            value = item.objective_values[objective.metric_id]
            return value if objective.direction is ExperimentObjectiveDirection.MAXIMIZE else -value

        def dominates(candidate: ExperimentRankedTrial, other: ExperimentRankedTrial) -> bool:
            comparisons = [
                (aligned(candidate, objective), aligned(other, objective))
                for objective in self.spec.objectives
            ]
            return all(left >= right for left, right in comparisons) and any(
                left > right for left, right in comparisons
            )

        for item in eligible:
            expected = not any(other is not item and dominates(other, item) for other in eligible)
            if item.pareto_front != expected:
                raise ValueError("experiment analysis Pareto classification is inconsistent")

    @staticmethod
    def expected_analysis_id(
        *,
        index_id: str,
        spec: ExperimentRankingSpec,
        trials: tuple[ExperimentRankedTrial, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "index_id": index_id,
                "spec": spec.model_dump(mode="json"),
                "trials": [item.model_dump(mode="json") for item in trials],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"experiment_analysis_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentIndexBuilder:
    def __init__(self, root: Path | str, state_root: Path | str) -> None:
        self.root = Path(root).resolve(strict=True)
        self.state_root = Path(state_root).resolve(strict=True)
        if not self.root.is_dir() or not self.state_root.is_dir():
            raise ValueError("experiment root and state directory must be directories")

    def build(self) -> ExperimentOperationsIndex:
        runner = ExperimentRunner(self.root, self.state_root)
        reports: list[ExperimentReportReference] = []
        rows: list[ExperimentIndexRow] = []
        definitions: dict[str, ExperimentMetricDefinition] = {}
        issues: list[ExperimentIndexIssue] = []

        for entry in sorted(self.state_root.iterdir(), key=lambda item: item.name):
            relative = entry.relative_to(self.state_root).as_posix()
            if (
                entry.is_symlink()
                or not entry.is_dir()
                or _PLAN_PATTERN.fullmatch(entry.name) is None
            ):
                issues.append(
                    ExperimentIndexIssue(
                        path=relative,
                        detail="unrecognized experiment state entry",
                    )
                )
                continue
            try:
                plan = self._load_plan(entry)
                report = runner.report(plan)
                report_reference = self._report_reference(report)
                plan_rows, plan_definitions = self._report_rows(plan, report)
                for definition in plan_definitions:
                    existing = definitions.setdefault(definition.metric_id, definition)
                    if existing != definition:
                        raise ValueError("experiment metric ID collision")
            except (OSError, RuntimeError, ValueError) as error:
                issues.append(
                    ExperimentIndexIssue(
                        path=relative,
                        detail=f"invalid experiment state: {error}",
                    )
                )
                continue
            reports.append(report_reference)
            rows.extend(plan_rows)

        report_tuple = tuple(sorted(reports, key=lambda item: item.plan_id))
        row_tuple = tuple(sorted(rows, key=lambda item: (item.plan_id, item.trial_id)))
        parameter_columns = tuple(sorted({key for row in row_tuple for key in row.parameters}))
        metric_definitions = tuple(sorted(definitions.values(), key=lambda item: item.metric_id))
        issue_tuple = tuple(sorted(set(issues), key=lambda item: (item.path, item.detail)))
        state_counts = {
            state.value: sum(row.state is state for row in row_tuple)
            for state in ExperimentTrialState
        }
        index_id = ExperimentOperationsIndex.expected_index_id(
            reports=report_tuple,
            rows=row_tuple,
            parameter_columns=parameter_columns,
            metric_definitions=metric_definitions,
            issues=issue_tuple,
        )
        return ExperimentOperationsIndex(
            index_id=index_id,
            reports=report_tuple,
            rows=row_tuple,
            parameter_columns=parameter_columns,
            metric_definitions=metric_definitions,
            state_counts=state_counts,
            issues=issue_tuple,
        )

    def _load_plan(self, directory: Path) -> ExperimentPlan:
        plan_path = directory / "plan.json"
        payload = plan_path.read_bytes()
        plan = ExperimentPlan.model_validate_json(payload)
        if payload != plan.canonical_bytes() + b"\n":
            raise ValueError("experiment plan is not canonical")
        if directory.name != plan.plan_id:
            raise ValueError("experiment plan directory differs from its plan ID")
        return plan

    @staticmethod
    def _report_reference(report: ExperimentReport) -> ExperimentReportReference:
        return ExperimentReportReference(
            plan_id=report.plan_id,
            report_id=report.report_id,
            name=report.name,
            trial_count=report.trial_count,
            state_counts=report.state_counts,
            gate_count=report.gate_count,
            failed_gate_count=report.failed_gate_count,
        )

    def _report_rows(
        self,
        plan: ExperimentPlan,
        report: ExperimentReport,
    ) -> tuple[list[ExperimentIndexRow], list[ExperimentMetricDefinition]]:
        report_trials = {item.trial_id: item for item in report.trials}
        rows: list[ExperimentIndexRow] = []
        definitions: dict[str, ExperimentMetricDefinition] = {}
        for trial in plan.trials:
            summary = report_trials[trial.trial_id]
            metric_values: dict[str, float] = {}
            for stage_plan, stage_summary in zip(trial.stages, summary.stages, strict=True):
                if stage_summary.state is not ExperimentStageState.SUCCEEDED:
                    continue
                for output_index, evidence in enumerate(stage_summary.outputs):
                    extracted = self._artifact_metrics(
                        experiment_name=plan.name,
                        stage_name=stage_plan.name,
                        output_index=output_index,
                        kind=evidence.kind,
                        path=evidence.path,
                        expected_sha256=evidence.sha256,
                    )
                    for definition, value in extracted:
                        existing = definitions.setdefault(definition.metric_id, definition)
                        if existing != definition:
                            raise ValueError("experiment metric ID collision")
                        metric_values[definition.metric_id] = value
            rows.append(
                ExperimentIndexRow(
                    plan_id=plan.plan_id,
                    report_id=report.report_id,
                    experiment_name=plan.name,
                    trial_id=trial.trial_id,
                    state=summary.state,
                    parameters=trial.parameters,
                    metric_values=dict(sorted(metric_values.items())),
                )
            )
        return rows, list(definitions.values())

    def _artifact_metrics(
        self,
        *,
        experiment_name: str,
        stage_name: str,
        output_index: int,
        kind: ExperimentArtifactKind,
        path: str,
        expected_sha256: str,
    ) -> list[tuple[ExperimentMetricDefinition, float]]:
        if kind not in {
            ExperimentArtifactKind.BENCHMARK_REPORT,
            ExperimentArtifactKind.COMPARISON_REPORT,
        }:
            return []
        resolved = (self.root / path).resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("experiment metric artifact escapes the project root") from error
        payload = resolved.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("experiment metric artifact digest changed during indexing")

        extracted: list[tuple[ExperimentMetricDefinition, float]] = []
        if kind is ExperimentArtifactKind.BENCHMARK_REPORT:
            report = BenchmarkReport.model_validate_json(payload)
            for metric_name, benchmark_metric in sorted(report.metrics.items()):
                definition = self._definition(
                    experiment_name=experiment_name,
                    stage_name=stage_name,
                    output_index=output_index,
                    kind=kind,
                    metric=metric_name,
                    statistic=ExperimentMetricStatistic.VALUE,
                    unit=benchmark_metric.unit,
                )
                extracted.append((definition, benchmark_metric.value))
            return extracted

        comparison = BenchmarkComparisonReport.model_validate_json(payload)
        statistics = (
            ExperimentMetricStatistic.BASELINE_MEAN,
            ExperimentMetricStatistic.CANDIDATE_MEAN,
            ExperimentMetricStatistic.ABSOLUTE_DELTA,
            ExperimentMetricStatistic.RELATIVE_DELTA,
            ExperimentMetricStatistic.CONFIDENCE_LOW,
            ExperimentMetricStatistic.CONFIDENCE_HIGH,
        )
        for metric_name, comparison_metric in sorted(comparison.metrics.items()):
            for statistic in statistics:
                value = getattr(comparison_metric, statistic.value)
                if value is None:
                    continue
                definition = self._definition(
                    experiment_name=experiment_name,
                    stage_name=stage_name,
                    output_index=output_index,
                    kind=kind,
                    metric=metric_name,
                    statistic=statistic,
                    unit=comparison_metric.unit,
                )
                extracted.append((definition, value))
        return extracted

    @staticmethod
    def _definition(
        *,
        experiment_name: str,
        stage_name: str,
        output_index: int,
        kind: ExperimentArtifactKind,
        metric: str,
        statistic: ExperimentMetricStatistic,
        unit: str,
    ) -> ExperimentMetricDefinition:
        metric_id = ExperimentMetricDefinition.expected_metric_id(
            experiment_name=experiment_name,
            stage_name=stage_name,
            output_index=output_index,
            artifact_kind=kind,
            metric=metric,
            statistic=statistic,
            unit=unit,
        )
        return ExperimentMetricDefinition(
            metric_id=metric_id,
            experiment_name=experiment_name,
            stage_name=stage_name,
            output_index=output_index,
            artifact_kind=kind,
            metric=metric,
            statistic=statistic,
            unit=unit,
        )


class ExperimentAnalyzer:
    def analyze(
        self,
        index: ExperimentOperationsIndex,
        spec: ExperimentRankingSpec,
    ) -> ExperimentAnalysis:
        definitions = {item.metric_id for item in index.metric_definitions}
        unknown = [item.metric_id for item in spec.objectives if item.metric_id not in definitions]
        if unknown:
            raise ValueError("unknown experiment objective metrics: " + ", ".join(unknown))

        selected = [
            row
            for row in index.rows
            if (not spec.experiment_names or row.experiment_name in spec.experiment_names)
            and (not spec.plan_ids or row.plan_id in spec.plan_ids)
        ]
        if not selected:
            raise ValueError("experiment ranking selection contains no trials")

        objective_ids = tuple(item.metric_id for item in spec.objectives)
        eligible_rows = [
            row
            for row in selected
            if row.state in spec.eligible_states
            and all(metric_id in row.metric_values for metric_id in objective_ids)
        ]
        if not eligible_rows:
            raise ValueError("experiment ranking has no eligible trials with complete objectives")
        if len(eligible_rows) > spec.max_candidates:
            raise ValueError("experiment ranking exceeds its maximum candidate count")

        baseline = self._baseline(spec, eligible_rows)
        normalized = self._normalized_values(spec.objectives, eligible_rows)
        pareto = self._pareto_front(spec.objectives, eligible_rows)
        weights = sum(item.weight for item in spec.objectives)
        scored = [
            (
                sum(
                    objective.weight * normalized[(row.plan_id, row.trial_id, objective.metric_id)]
                    for objective in spec.objectives
                )
                / weights,
                row,
            )
            for row in eligible_rows
        ]
        scored.sort(key=lambda item: (-item[0], item[1].plan_id, item[1].trial_id))

        ranked: list[ExperimentRankedTrial] = []
        for rank, (score, row) in enumerate(scored, start=1):
            key = (row.plan_id, row.trial_id)
            ranked.append(
                ExperimentRankedTrial(
                    plan_id=row.plan_id,
                    report_id=row.report_id,
                    experiment_name=row.experiment_name,
                    trial_id=row.trial_id,
                    state=row.state,
                    parameters=row.parameters,
                    objective_values={
                        metric_id: row.metric_values[metric_id]
                        for metric_id in sorted(objective_ids)
                    },
                    eligible=True,
                    rank=rank,
                    score=score,
                    pareto_front=key in pareto,
                    baseline_deltas=self._baseline_deltas(spec.objectives, baseline, row),
                    detail="eligible completed trial with all objective metrics",
                )
            )

        eligible_keys = {(item.plan_id, item.trial_id) for item in eligible_rows}
        ineligible = [
            ExperimentRankedTrial(
                plan_id=row.plan_id,
                report_id=row.report_id,
                experiment_name=row.experiment_name,
                trial_id=row.trial_id,
                state=row.state,
                parameters=row.parameters,
                objective_values={
                    metric_id: row.metric_values[metric_id]
                    for metric_id in sorted(objective_ids)
                    if metric_id in row.metric_values
                },
                eligible=False,
                detail=self._ineligible_detail(row, spec, objective_ids),
            )
            for row in selected
            if (row.plan_id, row.trial_id) not in eligible_keys
        ]
        trial_tuple = tuple(
            ranked + sorted(ineligible, key=lambda item: (item.plan_id, item.trial_id))
        )
        front = tuple(
            ExperimentTrialReference(plan_id=item.plan_id, trial_id=item.trial_id)
            for item in sorted(
                (item for item in ranked if item.pareto_front),
                key=lambda item: (item.plan_id, item.trial_id),
            )
        )
        analysis_id = ExperimentAnalysis.expected_analysis_id(
            index_id=index.index_id,
            spec=spec,
            trials=trial_tuple,
        )
        return ExperimentAnalysis(
            analysis_id=analysis_id,
            index_id=index.index_id,
            spec=spec,
            trials=trial_tuple,
            eligible_count=len(ranked),
            ineligible_count=len(ineligible),
            pareto_front=front,
        )

    @staticmethod
    def _baseline(
        spec: ExperimentRankingSpec,
        rows: list[ExperimentIndexRow],
    ) -> ExperimentIndexRow | None:
        if spec.baseline_plan_id is None or spec.baseline_trial_id is None:
            return None
        for row in rows:
            if row.plan_id == spec.baseline_plan_id and row.trial_id == spec.baseline_trial_id:
                return row
        raise ValueError("experiment ranking baseline is not an eligible selected trial")

    @staticmethod
    def _normalized_values(
        objectives: tuple[ExperimentObjective, ...],
        rows: list[ExperimentIndexRow],
    ) -> dict[tuple[str, str, str], float]:
        normalized: dict[tuple[str, str, str], float] = {}
        for objective in objectives:
            aligned = {
                (row.plan_id, row.trial_id): (
                    row.metric_values[objective.metric_id]
                    if objective.direction is ExperimentObjectiveDirection.MAXIMIZE
                    else -row.metric_values[objective.metric_id]
                )
                for row in rows
            }
            low = min(aligned.values())
            high = max(aligned.values())
            for (plan_id, trial_id), value in aligned.items():
                normalized[(plan_id, trial_id, objective.metric_id)] = (
                    1.0 if math.isclose(low, high) else (value - low) / (high - low)
                )
        return normalized

    @staticmethod
    def _pareto_front(
        objectives: tuple[ExperimentObjective, ...],
        rows: list[ExperimentIndexRow],
    ) -> set[tuple[str, str]]:
        def aligned(row: ExperimentIndexRow, objective: ExperimentObjective) -> float:
            value = row.metric_values[objective.metric_id]
            return value if objective.direction is ExperimentObjectiveDirection.MAXIMIZE else -value

        def dominates(candidate: ExperimentIndexRow, other: ExperimentIndexRow) -> bool:
            comparisons = [
                (aligned(candidate, objective), aligned(other, objective))
                for objective in objectives
            ]
            return all(left >= right for left, right in comparisons) and any(
                left > right for left, right in comparisons
            )

        front: set[tuple[str, str]] = set()
        for row in rows:
            if not any(other is not row and dominates(other, row) for other in rows):
                front.add((row.plan_id, row.trial_id))
        return front

    @staticmethod
    def _baseline_deltas(
        objectives: tuple[ExperimentObjective, ...],
        baseline: ExperimentIndexRow | None,
        row: ExperimentIndexRow,
    ) -> tuple[ExperimentBaselineDelta, ...]:
        if baseline is None:
            return ()
        deltas = []
        for objective in sorted(objectives, key=lambda item: item.metric_id):
            baseline_value = baseline.metric_values[objective.metric_id]
            candidate_value = row.metric_values[objective.metric_id]
            raw_delta = candidate_value - baseline_value
            improvement = (
                raw_delta
                if objective.direction is ExperimentObjectiveDirection.MAXIMIZE
                else -raw_delta
            )
            deltas.append(
                ExperimentBaselineDelta(
                    metric_id=objective.metric_id,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    raw_delta=raw_delta,
                    improvement=improvement,
                )
            )
        return tuple(deltas)

    @staticmethod
    def _ineligible_detail(
        row: ExperimentIndexRow,
        spec: ExperimentRankingSpec,
        objective_ids: tuple[str, ...],
    ) -> str:
        if row.state not in spec.eligible_states:
            return f"trial state {row.state.value!r} is excluded by the ranking policy"
        missing = [metric_id for metric_id in objective_ids if metric_id not in row.metric_values]
        return "missing objective metrics: " + ", ".join(missing)


def render_experiment_markdown(
    index: ExperimentOperationsIndex,
    analysis: ExperimentAnalysis,
) -> str:
    _validate_render_inputs(index, analysis)
    definitions = {item.metric_id: item for item in index.metric_definitions}
    objectives = tuple(item.metric_id for item in analysis.spec.objectives)
    lines = [
        "# Experiment operations report",
        "",
        f"- Index: `{index.index_id}`",
        f"- Analysis: `{analysis.analysis_id}`",
        f"- Reports: {len(index.reports)}",
        f"- Selected trials: {len(analysis.trials)}",
        f"- Eligible trials: {analysis.eligible_count}",
        f"- Pareto-front trials: {len(analysis.pareto_front)}",
        f"- Discovery issues: {len(index.issues)}",
        "",
        "## Ranking",
        "",
    ]
    headers = ["Rank", "Pareto", "State", "Score", "Trial", *index.parameter_columns]
    headers.extend(definitions[metric_id].label for metric_id in objectives)
    if analysis.spec.baseline_plan_id is not None:
        headers.extend(f"Improvement: {definitions[metric_id].label}" for metric_id in objectives)
    lines.extend(_markdown_table(headers, _ranking_rows(index, analysis, objectives)))
    lines.extend(["", "## Metric catalog", ""])
    catalog_rows = [
        [
            item.metric_id,
            item.experiment_name,
            item.label,
            item.artifact_kind.value,
            item.unit,
        ]
        for item in index.metric_definitions
    ]
    lines.extend(
        _markdown_table(
            ["Metric ID", "Experiment", "Label", "Artifact", "Unit"],
            catalog_rows,
        )
    )
    lines.extend(["", "## Report inventory", ""])
    report_rows = [
        [
            item.plan_id,
            item.report_id,
            item.name,
            str(item.trial_count),
            str(item.failed_gate_count),
        ]
        for item in index.reports
    ]
    lines.extend(
        _markdown_table(
            ["Plan", "Report", "Experiment", "Trials", "Failed gates"],
            report_rows,
        )
    )
    if index.issues:
        lines.extend(["", "## Discovery issues", ""])
        lines.extend(
            _markdown_table(["Path", "Detail"], [[i.path, i.detail] for i in index.issues])
        )
    return "\n".join(lines) + "\n"


def render_experiment_csv(index: ExperimentOperationsIndex) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    metric_ids = tuple(item.metric_id for item in index.metric_definitions)
    writer.writerow(
        [
            "plan_id",
            "report_id",
            "experiment_name",
            "trial_id",
            "state",
            *(f"parameter:{name}" for name in index.parameter_columns),
            *metric_ids,
        ]
    )
    for row in index.rows:
        writer.writerow(
            [
                row.plan_id,
                row.report_id,
                row.experiment_name,
                row.trial_id,
                row.state.value,
                *(_format_csv_scalar(row.parameters.get(name)) for name in index.parameter_columns),
                *(
                    format(row.metric_values[metric_id], ".17g")
                    if metric_id in row.metric_values
                    else ""
                    for metric_id in metric_ids
                ),
            ]
        )
    return output.getvalue()


def render_experiment_html(
    index: ExperimentOperationsIndex,
    analysis: ExperimentAnalysis,
) -> str:
    _validate_render_inputs(index, analysis)
    definitions = {item.metric_id: item for item in index.metric_definitions}
    objectives = tuple(item.metric_id for item in analysis.spec.objectives)
    headers = ["Rank", "Pareto", "State", "Score", "Trial", *index.parameter_columns]
    headers.extend(definitions[metric_id].label for metric_id in objectives)
    if analysis.spec.baseline_plan_id is not None:
        headers.extend(f"Improvement: {definitions[metric_id].label}" for metric_id in objectives)
    ranking = _html_table(headers, _ranking_rows(index, analysis, objectives))
    catalog = _html_table(
        ["Metric ID", "Experiment", "Label", "Artifact", "Unit"],
        [
            [
                item.metric_id,
                item.experiment_name,
                item.label,
                item.artifact_kind.value,
                item.unit,
            ]
            for item in index.metric_definitions
        ],
    )
    reports = _html_table(
        ["Plan", "Report", "Experiment", "Trials", "Failed gates"],
        [
            [
                item.plan_id,
                item.report_id,
                item.name,
                str(item.trial_count),
                str(item.failed_gate_count),
            ]
            for item in index.reports
        ],
    )
    issues = (
        "<h2>Discovery issues</h2>"
        + _html_table(["Path", "Detail"], [[item.path, item.detail] for item in index.issues])
        if index.issues
        else ""
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Experiment operations report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1440px;margin:2rem auto;"
        "padding:0 1rem;color:#18212f}code{font-size:.86em}table{border-collapse:collapse;"
        "width:100%;font-size:.9rem}th,td{border:1px solid #d7dde5;padding:.45rem;"
        "text-align:left;vertical-align:top}th{background:#f3f6f9;position:sticky;top:0}"
        ".table{overflow:auto;margin-bottom:2rem}.summary{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem}.card{"
        "background:#f7f9fb;border:1px solid #d7dde5;border-radius:.5rem;padding:.75rem}"
        "</style></head><body>"
        "<h1>Experiment operations report</h1>"
        '<div class="summary">'
        '<div class="card"><strong>Index</strong><br><code>'
        f"{html.escape(index.index_id)}</code></div>"
        '<div class="card"><strong>Analysis</strong><br><code>'
        f"{html.escape(analysis.analysis_id)}</code></div>"
        f'<div class="card"><strong>Reports</strong><br>{len(index.reports)}</div>'
        f'<div class="card"><strong>Eligible</strong><br>{analysis.eligible_count}</div>'
        f'<div class="card"><strong>Pareto front</strong><br>{len(analysis.pareto_front)}</div>'
        f'<div class="card"><strong>Issues</strong><br>{len(index.issues)}</div>'
        "</div><h2>Ranking</h2>"
        f"{ranking}<h2>Metric catalog</h2>{catalog}<h2>Report inventory</h2>{reports}{issues}"
        "</body></html>\n"
    )


def _validate_render_inputs(
    index: ExperimentOperationsIndex,
    analysis: ExperimentAnalysis,
) -> None:
    if analysis.index_id != index.index_id:
        raise ValueError("experiment analysis was produced from another index")


def _ranking_rows(
    index: ExperimentOperationsIndex,
    analysis: ExperimentAnalysis,
    objectives: tuple[str, ...],
) -> list[list[str]]:
    rows = []
    for item in analysis.trials:
        row = [
            str(item.rank) if item.rank is not None else "—",
            "yes" if item.pareto_front else "",
            item.state.value,
            _format_number(item.score) if item.score is not None else "—",
            item.trial_id,
        ]
        row.extend(_format_scalar(item.parameters.get(name)) for name in index.parameter_columns)
        row.extend(
            _format_number(item.objective_values[metric_id])
            if metric_id in item.objective_values
            else "—"
            for metric_id in objectives
        )
        if analysis.spec.baseline_plan_id is not None:
            deltas = {delta.metric_id: delta for delta in item.baseline_deltas}
            row.extend(
                _format_number(deltas[metric_id].improvement) if metric_id in deltas else "—"
                for metric_id in objectives
            )
        rows.append(row)
    return rows


def _format_scalar(value: ExperimentScalar | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_csv_scalar(value: ExperimentScalar | None) -> str:
    rendered = _format_scalar(value)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + rendered
    return rendered


def _format_number(value: float) -> str:
    return format(value, ".8g")


def _markdown_table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    escaped_headers = [_markdown_cell(item) for item in headers]
    output = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in escaped_headers) + " |",
    ]
    output.extend("| " + " | ".join(_markdown_cell(item) for item in row) + " |" for row in rows)
    return output


def _markdown_cell(value: str) -> str:
    escaped = html.escape(value, quote=False)
    return escaped.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _html_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    header = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(item)}</td>" for item in row) + "</tr>" for row in rows
    )
    return (
        f'<div class="table"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )

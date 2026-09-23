from __future__ import annotations

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import orjson

from agentic_rl_forge.contracts import (
    CheckpointManifest,
    DatasetCollectionManifest,
    DatasetManifest,
    RolloutPlan,
    TrainerBatchManifest,
)
from agentic_rl_forge.evaluation import BenchmarkComparisonReport, BenchmarkReport
from agentic_rl_forge.experiments.models import (
    ExperimentArtifactEvidence,
    ExperimentArtifactKind,
    ExperimentArtifactPlan,
    ExperimentGateResult,
    ExperimentMetricGate,
    ExperimentMetricStatistic,
    ExperimentPlan,
    ExperimentReport,
    ExperimentStageAttempt,
    ExperimentStagePlan,
    ExperimentStageRecord,
    ExperimentStageState,
    ExperimentStageSummary,
    ExperimentTrialPlan,
    ExperimentTrialState,
    ExperimentTrialSummary,
)
from agentic_rl_forge.pipelines import load_search_r1_collection_config
from agentic_rl_forge.storage import BlobConflictError, LocalBlobStore


class ExperimentExecutionError(ValueError):
    pass


class ExperimentRunner:
    MAX_WORKERS = 64

    def __init__(self, root: Path | str, state_root: Path | str) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("experiment root must be a directory")
        self.state_root = Path(state_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.store = LocalBlobStore(self.state_root)

    def run(
        self,
        plan: ExperimentPlan,
        *,
        confirm_plan_id: str,
        max_workers: int = 1,
    ) -> ExperimentReport:
        if confirm_plan_id.strip() != plan.plan_id:
            raise ExperimentExecutionError("experiment confirmation does not match the plan ID")
        if max_workers < 1 or max_workers > self.MAX_WORKERS:
            raise ValueError("experiment worker count must be between 1 and 64")
        self._persist_plan(plan)
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="experiment-trial",
        ) as executor:
            futures = {
                trial.trial_id: executor.submit(self._run_trial, plan, trial)
                for trial in plan.trials
            }
            for trial in plan.trials:
                futures[trial.trial_id].result()
        return self.report(plan)

    def report(self, plan: ExperimentPlan) -> ExperimentReport:
        self._validate_persisted_plan(plan)
        trial_summaries = tuple(self._trial_summary(plan, trial) for trial in plan.trials)
        state_counts = {
            state.value: sum(trial.state is state for trial in trial_summaries)
            for state in ExperimentTrialState
        }
        gates = [
            gate for trial in trial_summaries for stage in trial.stages for gate in stage.gates
        ]
        report_id = ExperimentReport.expected_report_id(
            plan_id=plan.plan_id,
            name=plan.name,
            trials=trial_summaries,
        )
        return ExperimentReport(
            report_id=report_id,
            plan_id=plan.plan_id,
            name=plan.name,
            trials=trial_summaries,
            trial_count=len(trial_summaries),
            state_counts=state_counts,
            gate_count=len(gates),
            failed_gate_count=sum(not gate.passed for gate in gates),
        )

    def _run_trial(self, plan: ExperimentPlan, trial: ExperimentTrialPlan) -> None:
        config_path = self._materialize_config(plan, trial)
        succeeded: set[str] = set()
        for stage in trial.stages:
            if not set(stage.depends_on) <= succeeded:
                continue
            if self._execute_stage(plan, trial, stage, config_path):
                succeeded.add(stage.name)

    def _execute_stage(
        self,
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
        stage: ExperimentStagePlan,
        config_path: Path,
    ) -> bool:
        command = self._execution_command(stage, config_path)
        existing = self._load_stage_record(plan, trial, stage)
        try:
            inputs = self._collect_evidence(stage.inputs, config_path=config_path)
        except ExperimentExecutionError as error:
            self._record_failed_attempt(
                plan,
                trial,
                stage,
                command=command,
                inputs=(),
                return_code=2,
                detail=str(error),
            )
            return False
        if existing is not None:
            if existing.command != command or existing.inputs != inputs:
                return False
            try:
                outputs = self._collect_evidence(stage.outputs, config_path=config_path)
            except ExperimentExecutionError:
                return False
            return existing.outputs == outputs

        log_token = uuid4().hex
        stdout_key = self._stage_key(
            plan,
            trial,
            stage,
            f"logs/{log_token}.stdout.log",
        )
        stderr_key = self._stage_key(
            plan,
            trial,
            stage,
            f"logs/{log_token}.stderr.log",
        )
        stdout_path = self.store._filesystem_path(self.state_root / stdout_key)
        stderr_path = self.store._filesystem_path(self.state_root / stderr_key)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc)
        return_code = 127
        detail = "experiment stage command was not started"
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=self.root,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        shell=False,
                        env=self._stage_environment(),
                    )
                except OSError as error:
                    detail = f"experiment stage could not start: {error}"
                else:
                    return_code = completed.returncode
                    detail = (
                        "experiment stage command failed"
                        if return_code
                        else "experiment stage outputs are invalid"
                    )
        except OSError as error:
            detail = f"experiment stage log creation failed: {error}"
        completed_at = datetime.now(timezone.utc)
        if return_code != 0:
            self._write_attempt(
                ExperimentStageAttempt(
                    attempt_id=ExperimentStageAttempt.expected_attempt_id(
                        plan_id=plan.plan_id,
                        trial_id=trial.trial_id,
                        stage_name=stage.name,
                        command=command,
                        inputs=inputs,
                        return_code=return_code,
                        detail=detail,
                        started_at=started_at,
                        completed_at=completed_at,
                    ),
                    plan_id=plan.plan_id,
                    trial_id=trial.trial_id,
                    stage_name=stage.name,
                    command=command,
                    inputs=inputs,
                    return_code=return_code,
                    detail=detail,
                    started_at=started_at,
                    completed_at=completed_at,
                    stdout_path=stdout_key,
                    stderr_path=stderr_key,
                )
            )
            return False
        try:
            outputs = self._collect_evidence(stage.outputs, config_path=config_path)
        except ExperimentExecutionError as error:
            self._write_attempt(
                ExperimentStageAttempt(
                    attempt_id=ExperimentStageAttempt.expected_attempt_id(
                        plan_id=plan.plan_id,
                        trial_id=trial.trial_id,
                        stage_name=stage.name,
                        command=command,
                        inputs=inputs,
                        return_code=0,
                        detail=str(error),
                        started_at=started_at,
                        completed_at=completed_at,
                    ),
                    plan_id=plan.plan_id,
                    trial_id=trial.trial_id,
                    stage_name=stage.name,
                    command=command,
                    inputs=inputs,
                    return_code=0,
                    detail=str(error),
                    started_at=started_at,
                    completed_at=completed_at,
                    stdout_path=stdout_key,
                    stderr_path=stderr_key,
                )
            )
            return False
        record_id = ExperimentStageRecord.expected_record_id(
            plan_id=plan.plan_id,
            trial_id=trial.trial_id,
            stage_name=stage.name,
            command=command,
            inputs=inputs,
            outputs=outputs,
        )
        record = ExperimentStageRecord(
            record_id=record_id,
            plan_id=plan.plan_id,
            trial_id=trial.trial_id,
            stage_name=stage.name,
            command=command,
            inputs=inputs,
            outputs=outputs,
            started_at=started_at,
            completed_at=completed_at,
            stdout_path=stdout_key,
            stderr_path=stderr_key,
        )
        key = self._stage_key(plan, trial, stage, "record.json")
        try:
            self.store.put_if_absent(key, record.canonical_bytes() + b"\n")
        except BlobConflictError:
            loaded = self._load_stage_record(plan, trial, stage)
            return loaded is not None and loaded.record_id == record.record_id
        return True

    def _trial_summary(
        self,
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
    ) -> ExperimentTrialSummary:
        config_path = self._config_path(plan, trial)
        summaries: list[ExperimentStageSummary] = []
        states: dict[str, ExperimentStageState] = {}
        for stage in trial.stages:
            blocked = any(
                states.get(dependency) is not ExperimentStageState.SUCCEEDED
                for dependency in stage.depends_on
            )
            if blocked:
                summary = ExperimentStageSummary(
                    name=stage.name,
                    state=ExperimentStageState.BLOCKED,
                    detail="one or more stage dependencies are incomplete",
                )
            else:
                summary = self._stage_summary(plan, trial, stage, config_path)
            summaries.append(summary)
            states[stage.name] = summary.state
        stage_tuple = tuple(summaries)
        failed = any(
            stage.state in {ExperimentStageState.FAILED, ExperimentStageState.INVALID}
            for stage in stage_tuple
        )
        regressed = any(not gate.passed for stage in stage_tuple for gate in stage.gates)
        complete = all(stage.state is ExperimentStageState.SUCCEEDED for stage in stage_tuple)
        state = (
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
        return ExperimentTrialSummary(
            trial_id=trial.trial_id,
            parameters=trial.parameters,
            config_digest=trial.config_digest,
            stages=stage_tuple,
            state=state,
        )

    def _stage_summary(
        self,
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
        stage: ExperimentStagePlan,
        config_path: Path,
    ) -> ExperimentStageSummary:
        try:
            record = self._load_stage_record(plan, trial, stage)
        except ExperimentExecutionError as error:
            return ExperimentStageSummary(
                name=stage.name,
                state=ExperimentStageState.INVALID,
                detail=str(error),
            )
        if record is None:
            try:
                attempt = self._latest_attempt(plan, trial, stage)
            except ExperimentExecutionError as error:
                return ExperimentStageSummary(
                    name=stage.name,
                    state=ExperimentStageState.INVALID,
                    detail=str(error),
                )
            if attempt is None:
                return ExperimentStageSummary(
                    name=stage.name,
                    state=ExperimentStageState.PENDING,
                    detail="stage has no execution evidence",
                )
            return ExperimentStageSummary(
                name=stage.name,
                state=ExperimentStageState.FAILED,
                latest_attempt_id=attempt.attempt_id,
                detail=attempt.detail,
                inputs=attempt.inputs,
            )
        command = self._execution_command(stage, config_path)
        if record.command != command:
            return ExperimentStageSummary(
                name=stage.name,
                state=ExperimentStageState.INVALID,
                record_id=record.record_id,
                detail="stage command differs from immutable success evidence",
            )
        try:
            inputs = self._collect_evidence(stage.inputs, config_path=config_path)
            outputs = self._collect_evidence(stage.outputs, config_path=config_path)
        except ExperimentExecutionError as error:
            return ExperimentStageSummary(
                name=stage.name,
                state=ExperimentStageState.INVALID,
                record_id=record.record_id,
                detail=str(error),
            )
        if inputs != record.inputs or outputs != record.outputs:
            return ExperimentStageSummary(
                name=stage.name,
                state=ExperimentStageState.INVALID,
                record_id=record.record_id,
                detail="stage artifacts differ from immutable success evidence",
                inputs=inputs,
                outputs=outputs,
            )
        gates = tuple(self._evaluate_gate(gate) for gate in stage.gates)
        return ExperimentStageSummary(
            name=stage.name,
            state=ExperimentStageState.SUCCEEDED,
            record_id=record.record_id,
            detail="stage command and artifact evidence are valid",
            inputs=inputs,
            outputs=outputs,
            gates=gates,
        )

    def _evaluate_gate(self, gate: ExperimentMetricGate) -> ExperimentGateResult:
        path = self._artifact_path(gate.artifact_path, config_path=None)
        actual: float | None = None
        detail = "metric gate passed"
        try:
            payload = path.read_bytes()
            if gate.statistic is ExperimentMetricStatistic.VALUE:
                benchmark_report = BenchmarkReport.model_validate_json(payload)
                benchmark_metric = benchmark_report.metrics.get(gate.metric)
                if benchmark_metric is None:
                    raise ValueError("benchmark metric is missing")
                actual = benchmark_metric.value
            else:
                comparison_report = BenchmarkComparisonReport.model_validate_json(payload)
                comparison_metric = comparison_report.metrics.get(gate.metric)
                if comparison_metric is None:
                    raise ValueError("comparison metric is missing")
                value = getattr(comparison_metric, gate.statistic.value)
                if value is None:
                    raise ValueError("comparison statistic is unavailable")
                actual = float(value)
        except (OSError, ValueError) as error:
            detail = str(error) or error.__class__.__name__
        passed = actual is not None
        if actual is not None and gate.minimum is not None and actual < gate.minimum:
            passed = False
            detail = "metric is below the required minimum"
        if actual is not None and gate.maximum is not None and actual > gate.maximum:
            passed = False
            detail = "metric exceeds the allowed maximum"
        if not passed and detail == "metric gate passed":
            detail = "metric gate failed"
        return ExperimentGateResult(
            name=gate.name,
            artifact_path=gate.artifact_path,
            metric=gate.metric,
            statistic=gate.statistic,
            actual=actual,
            minimum=gate.minimum,
            maximum=gate.maximum,
            passed=passed,
            detail=detail,
        )

    def _collect_evidence(
        self,
        artifacts: tuple[ExperimentArtifactPlan, ...],
        *,
        config_path: Path,
    ) -> tuple[ExperimentArtifactEvidence, ...]:
        evidence = []
        for artifact in artifacts:
            path = self._artifact_path(artifact.path, config_path=config_path)
            if not path.is_file():
                if artifact.required:
                    raise ExperimentExecutionError(
                        f"required experiment artifact {artifact.path!r} is missing"
                    )
                continue
            evidence.append(self._inspect_artifact(artifact, path))
        return tuple(evidence)

    def _inspect_artifact(
        self,
        artifact: ExperimentArtifactPlan,
        path: Path,
    ) -> ExperimentArtifactEvidence:
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ExperimentExecutionError(
                f"experiment artifact {artifact.path!r} could not be read"
            ) from error
        digest = hashlib.sha256(payload).hexdigest()
        content_id: str | None = None
        summary: dict[str, object] = {}
        try:
            if artifact.kind is ExperimentArtifactKind.COLLECTION_CONFIG:
                config = load_search_r1_collection_config(path)
                content_id = config.plan_config_digest
                summary = {
                    "name": config.name,
                    "dataset_name": config.dataset_name,
                    "model": config.model,
                    "policy_version": config.policy_version,
                    "seed": config.seed,
                }
            elif artifact.kind is ExperimentArtifactKind.DATASET:
                content_id = digest
            elif artifact.kind is ExperimentArtifactKind.DATASET_MANIFEST:
                manifest = DatasetManifest.model_validate_json(payload)
                content_id = manifest.manifest_id
                summary = {
                    "name": manifest.name,
                    "split": manifest.split,
                    "record_count": manifest.record_count,
                    "content_digest": manifest.content_digest,
                }
            elif artifact.kind is ExperimentArtifactKind.DATASET_COLLECTION_MANIFEST:
                collection = DatasetCollectionManifest.model_validate_json(payload)
                content_id = collection.collection_id
                summary = {
                    "name": collection.name,
                    "splits": sorted(collection.splits),
                    "total_record_count": collection.total_record_count,
                    "record_ids_digest": collection.record_ids_digest,
                }
            elif artifact.kind is ExperimentArtifactKind.ROLLOUT_PLAN:
                rollout = RolloutPlan.model_validate_json(payload)
                content_id = rollout.plan_id
                summary = {
                    "policy_version": rollout.policy_version,
                    "source_sha256": rollout.source_sha256,
                    "config_digest": rollout.config_digest,
                    "task_count": len(rollout.task_ids),
                    "slot_count": len(rollout.slots),
                }
            elif artifact.kind is ExperimentArtifactKind.CHECKPOINT_MANIFEST:
                checkpoint = CheckpointManifest.model_validate_json(payload)
                content_id = checkpoint.checkpoint_id
                summary = {
                    "run_id": checkpoint.run_id,
                    "step": checkpoint.step,
                    "policy_version": checkpoint.policy_version,
                    "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                    "dataset_manifest_digest": checkpoint.dataset_manifest_digest,
                }
            elif artifact.kind is ExperimentArtifactKind.TRAINER_BATCH_MANIFEST:
                batch = TrainerBatchManifest.model_validate_json(payload)
                content_id = batch.batch_id
                summary = {
                    "policy_version": batch.policy_version,
                    "source_run_id": batch.source_run_id,
                    "group_count": batch.group_count,
                    "trajectory_count": batch.trajectory_count,
                    "payload_sha256": batch.payload_sha256,
                }
            elif artifact.kind is ExperimentArtifactKind.BENCHMARK_REPORT:
                benchmark = BenchmarkReport.model_validate_json(payload)
                content_id = benchmark.report_id
                summary = {
                    "benchmark": benchmark.benchmark,
                    "run_id": benchmark.run_id,
                    "task_count": benchmark.task_count,
                    "attempt_count": benchmark.attempt_count,
                    "metrics": sorted(benchmark.metrics),
                }
            elif artifact.kind is ExperimentArtifactKind.COMPARISON_REPORT:
                comparison = BenchmarkComparisonReport.model_validate_json(payload)
                content_id = comparison.comparison_id
                summary = {
                    "benchmark": comparison.benchmark,
                    "baseline_name": comparison.baseline_name,
                    "candidate_name": comparison.candidate_name,
                    "matched_task_count": comparison.matched_task_count,
                    "metrics": sorted(comparison.metrics),
                }
        except ValueError as error:
            raise ExperimentExecutionError(
                f"experiment artifact {artifact.path!r} has an invalid "
                f"{artifact.kind.value} contract"
            ) from error
        return ExperimentArtifactEvidence(
            kind=artifact.kind,
            path=artifact.path,
            size_bytes=len(payload),
            sha256=digest,
            content_id=content_id,
            summary=summary,
        )

    def _materialize_config(
        self,
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
    ) -> Path:
        key = self._config_key(plan, trial)
        payload = orjson.dumps(
            trial.resolved_config,
            option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE,
        )
        self.store.put_if_absent(key, payload)
        path = self.state_root / key
        if hashlib.sha256(path.read_bytes().rstrip(b"\n")).hexdigest() != trial.config_digest:
            raise ExperimentExecutionError("materialized experiment config has an invalid digest")
        return path

    def _persist_plan(self, plan: ExperimentPlan) -> None:
        key = f"{plan.plan_id}/plan.json"
        try:
            self.store.put_if_absent(key, plan.canonical_bytes() + b"\n")
        except BlobConflictError as error:
            raise ExperimentExecutionError(
                "experiment state directory contains another plan with the same ID"
            ) from error

    def _validate_persisted_plan(self, plan: ExperimentPlan) -> None:
        key = f"{plan.plan_id}/plan.json"
        info = self.store.head(key)
        if info is None:
            return
        payload = self.store.get(key)
        if payload != plan.canonical_bytes() + b"\n":
            raise ExperimentExecutionError("persisted experiment plan differs from this plan")

    def _load_stage_record(
        self,
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
        stage: ExperimentStagePlan,
    ) -> ExperimentStageRecord | None:
        key = self._stage_key(plan, trial, stage, "record.json")
        if self.store.head(key) is None:
            return None
        try:
            payload = self.store.get(key)
            record = ExperimentStageRecord.model_validate_json(payload)
        except (KeyError, ValueError) as error:
            raise ExperimentExecutionError("experiment stage record is invalid") from error
        if payload != record.canonical_bytes() + b"\n":
            raise ExperimentExecutionError("experiment stage record is not canonical")
        if (
            record.plan_id != plan.plan_id
            or record.trial_id != trial.trial_id
            or record.stage_name != stage.name
        ):
            raise ExperimentExecutionError("experiment stage record identity is inconsistent")
        return record

    def _latest_attempt(
        self,
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
        stage: ExperimentStagePlan,
    ) -> ExperimentStageAttempt | None:
        prefix = self._stage_key(plan, trial, stage, "attempts/")
        attempts = []
        for key in self.store.list(prefix):
            try:
                payload = self.store.get(key)
                attempt = ExperimentStageAttempt.model_validate_json(payload)
            except (KeyError, ValueError) as error:
                raise ExperimentExecutionError("experiment stage attempt is invalid") from error
            if payload != attempt.canonical_bytes() + b"\n":
                raise ExperimentExecutionError("experiment stage attempt is not canonical")
            if (
                attempt.plan_id != plan.plan_id
                or attempt.trial_id != trial.trial_id
                or attempt.stage_name != stage.name
            ):
                raise ExperimentExecutionError("experiment stage attempt identity is inconsistent")
            attempts.append(attempt)
        return max(attempts, key=lambda item: (item.completed_at, item.attempt_id), default=None)

    def _record_failed_attempt(
        self,
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
        stage: ExperimentStagePlan,
        *,
        command: tuple[str, ...],
        inputs: tuple[ExperimentArtifactEvidence, ...],
        return_code: int,
        detail: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        attempt_id = ExperimentStageAttempt.expected_attempt_id(
            plan_id=plan.plan_id,
            trial_id=trial.trial_id,
            stage_name=stage.name,
            command=command,
            inputs=inputs,
            return_code=return_code,
            detail=detail,
            started_at=now,
            completed_at=now,
        )
        self._write_attempt(
            ExperimentStageAttempt(
                attempt_id=attempt_id,
                plan_id=plan.plan_id,
                trial_id=trial.trial_id,
                stage_name=stage.name,
                command=command,
                inputs=inputs,
                return_code=return_code,
                detail=detail,
                started_at=now,
                completed_at=now,
                stdout_path="not-created",
                stderr_path="not-created",
            )
        )

    def _write_attempt(self, attempt: ExperimentStageAttempt) -> None:
        key = (
            f"{attempt.plan_id}/{attempt.trial_id}/stages/{attempt.stage_name}/"
            f"attempts/{attempt.attempt_id}.json"
        )
        self.store.put_if_absent(key, attempt.canonical_bytes() + b"\n")

    def _execution_command(
        self,
        stage: ExperimentStagePlan,
        config_path: Path,
    ) -> tuple[str, ...]:
        return tuple(
            argument.replace("{config_path}", str(config_path)) for argument in stage.command
        )

    @staticmethod
    def _stage_environment() -> dict[str, str]:
        environment = dict(os.environ)
        for name in tuple(environment):
            if name.startswith("COV_CORE_") or name == "COVERAGE_PROCESS_START":
                environment.pop(name)
        return environment

    def _artifact_path(self, value: str, *, config_path: Path | None) -> Path:
        if value == "{config_path}":
            if config_path is None:
                raise ExperimentExecutionError("resolved config is unavailable for this artifact")
            return config_path
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ExperimentExecutionError(
                f"experiment artifact path {value!r} escapes the execution root"
            ) from error
        return candidate

    def _config_path(self, plan: ExperimentPlan, trial: ExperimentTrialPlan) -> Path:
        return self.state_root / self._config_key(plan, trial)

    @staticmethod
    def _config_key(plan: ExperimentPlan, trial: ExperimentTrialPlan) -> str:
        return f"{plan.plan_id}/{trial.trial_id}/resolved-config.json"

    @staticmethod
    def _stage_key(
        plan: ExperimentPlan,
        trial: ExperimentTrialPlan,
        stage: ExperimentStagePlan,
        suffix: str,
    ) -> str:
        return f"{plan.plan_id}/{trial.trial_id}/stages/{stage.name}/{suffix}"

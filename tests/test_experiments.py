from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    ArtifactLocation,
    CheckpointArtifact,
    CheckpointManifest,
    DataOrigin,
    DatasetCollectionManifest,
    DatasetFileManifest,
    DatasetManifest,
    Message,
    MessageRole,
    TaskSpec,
    TrainerBatchManifest,
    TrainerTrajectoryRef,
    VerifierSpec,
)
from agentic_rl_forge.evaluation import BenchmarkComparisonReport, PairedMetricDelta
from agentic_rl_forge.experiments import (
    ExperimentExecutionError,
    ExperimentPlan,
    ExperimentPlanError,
    ExperimentReport,
    ExperimentRunner,
    ExperimentStageRecord,
    ExperimentStageState,
    ExperimentTrialState,
    build_experiment_plan,
)
from agentic_rl_forge.rollout import RolloutPlanBuilder


def write_base_config(root: Path) -> Path:
    path = root / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "experiment-search-r1",
                "dataset_name": "fixture",
                "model": "fixture-model",
                "policy_version": "fixture-policy",
                "model_base_url": "http://127.0.0.1:8001",
                "retrieval_endpoint": "http://127.0.0.1:8000/retrieve",
                "rollouts_per_task": 2,
                "seed": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def benchmark_command(output: str, score: str) -> list[str]:
    code = (
        "from datetime import datetime,timezone;"
        "from pathlib import Path;"
        "from agentic_rl_forge.evaluation import BenchmarkReport,MetricValue;"
        f"p=Path({output!r});p.parent.mkdir(parents=True,exist_ok=True);"
        f"v=float({score!r});"
        "r=BenchmarkReport(report_id='report_fixture',benchmark='fixture',"
        "created_at=datetime(2026,9,18,tzinfo=timezone.utc),task_count=1,group_count=1,"
        "attempt_count=1,policy_versions=('fixture-policy',),"
        "environment_versions=('fixture-env',),status_counts=dict(succeeded=1),"
        "metrics=dict(score=MetricValue(value=v,numerator=v,denominator=1,unit='ratio')),"
        "group_diagnostics=());"
        "p.write_bytes(r.canonical_bytes()+b'\\n')"
    )
    return [sys.executable, "-c", code]


def write_matrix(
    root: Path,
    *,
    axes: dict[str, list[Any]] | None = None,
    stages: list[dict[str, Any]] | None = None,
    exclude: list[dict[str, Any]] | None = None,
) -> Path:
    output = "outputs/{trial_id}/benchmark.json"
    path = root / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "search-r1-ablation",
                "base_config": "base.yaml",
                "fixed_parameters": {"rollouts": 3},
                "axes": axes or {"quality": [0.8, 0.9], "seed": [11, 12]},
                "exclude": exclude or [],
                "config_bindings": {
                    "rollouts_per_task": "rollouts",
                    "seed": "seed",
                },
                "stages": stages
                or [
                    {
                        "name": "evaluate",
                        "command": benchmark_command(output, "{quality}"),
                        "inputs": [{"kind": "collection_config", "path": "{config_path}"}],
                        "outputs": [{"kind": "benchmark_report", "path": output}],
                        "gates": [
                            {
                                "name": "quality-floor",
                                "artifact_path": output,
                                "metric": "score",
                                "statistic": "value",
                                "minimum": 0.75,
                            }
                        ],
                    }
                ],
                "max_trials": 16,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_matrix_expansion_is_deterministic_bounded_and_config_bound(tmp_path: Path) -> None:
    write_base_config(tmp_path)
    matrix_path = write_matrix(
        tmp_path,
        exclude=[{"quality": 0.8, "seed": 12}],
    )

    plan = build_experiment_plan(matrix_path)
    repeated = build_experiment_plan(matrix_path)

    assert plan == repeated
    assert plan.canonical_bytes() == repeated.canonical_bytes()
    assert plan.trial_count == 3
    assert tuple(item.trial_id for item in plan.trials) == tuple(
        sorted(item.trial_id for item in plan.trials)
    )
    assert {item.resolved_config["rollouts_per_task"] for item in plan.trials} == {3}
    assert {item.resolved_config["seed"] for item in plan.trials} == {11, 12}
    assert all(item.stages[0].inputs[0].path == "{config_path}" for item in plan.trials)
    assert all(item.trial_id in item.stages[0].outputs[0].path for item in plan.trials)

    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    payload["max_trials"] = 2
    matrix_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentPlanError, match="maximum trial count"):
        build_experiment_plan(matrix_path)

    payload["max_trials"] = 16
    payload["stages"][0]["outputs"][0]["path"] = "shared-benchmark.json"
    payload["stages"][0]["gates"][0]["artifact_path"] = "shared-benchmark.json"
    matrix_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentPlanError, match="expanded experiment plan"):
        build_experiment_plan(matrix_path)


def test_bundled_search_r1_matrix_expands_to_four_reviewable_trials() -> None:
    project_root = Path(__file__).resolve().parents[1]
    plan = build_experiment_plan(project_root / "configs" / "experiments" / "search_r1_matrix.yaml")

    assert plan.trial_count == 4
    assert {tuple(trial.parameters) for trial in plan.trials} == {
        ("rollouts_per_task", "seed", "temperature")
    }
    assert all(len(trial.stages) == 3 for trial in plan.trials)
    assert all(trial.stages[-1].gates for trial in plan.trials)


def test_stage_environment_keeps_runtime_values_but_isolates_coverage_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPERIMENT_RUNTIME_VALUE", "preserved")
    monkeypatch.setenv("COV_CORE_SOURCE", "agentic_rl_forge")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "pyproject.toml")

    environment = ExperimentRunner._stage_environment()

    assert environment["EXPERIMENT_RUNTIME_VALUE"] == "preserved"
    assert "COV_CORE_SOURCE" not in environment
    assert "COVERAGE_PROCESS_START" not in environment


def test_runner_executes_resumes_and_detects_artifact_drift(tmp_path: Path) -> None:
    write_base_config(tmp_path)
    matrix_path = write_matrix(
        tmp_path,
        axes={"quality": [0.8, 0.9], "seed": [11]},
    )
    plan = build_experiment_plan(matrix_path)
    runner = ExperimentRunner(tmp_path, tmp_path / "state")

    preview = runner.report(plan)
    assert preview.state_counts[ExperimentTrialState.INCOMPLETE.value] == 2
    with pytest.raises(ExperimentExecutionError, match="confirmation"):
        runner.run(plan, confirm_plan_id="experiment_plan_" + "0" * 24)

    report = runner.run(plan, confirm_plan_id=plan.plan_id, max_workers=2)
    repeated = runner.run(plan, confirm_plan_id=plan.plan_id, max_workers=1)

    assert report == repeated
    assert report.state_counts[ExperimentTrialState.COMPLETE.value] == 2
    assert report.gate_count == 2
    assert report.failed_gate_count == 0
    for trial in report.trials:
        stage = trial.stages[0]
        assert stage.state is ExperimentStageState.SUCCEEDED
        assert stage.record_id is not None
        assert stage.inputs[0].content_id is not None
        assert stage.outputs[0].content_id == "report_fixture"

    drifted = tmp_path / plan.trials[0].stages[0].outputs[0].path
    drifted.write_bytes(b"{}\n")
    drift_report = runner.report(plan)
    assert drift_report.state_counts[ExperimentTrialState.FAILED.value] == 1
    assert drift_report.trials[0].stages[0].state is ExperimentStageState.INVALID
    rerun = runner.run(plan, confirm_plan_id=plan.plan_id)
    assert rerun.state_counts[ExperimentTrialState.FAILED.value] == 1
    assert drifted.read_bytes() == b"{}\n"


def test_failed_stage_is_retained_and_later_resumes_dependents(tmp_path: Path) -> None:
    write_base_config(tmp_path)
    output = "outputs/{trial_id}/benchmark.json"
    marker = "allow-experiment"
    guarded_code = (
        "from pathlib import Path;import sys;"
        f"sys.exit(7) if not Path({marker!r}).is_file() else None;"
        + benchmark_command(output, "0.9")[2]
    )
    summary_path = "outputs/{trial_id}/summary.txt"
    dependent_code = (
        "from pathlib import Path;"
        f"p=Path({summary_path!r});p.parent.mkdir(parents=True,exist_ok=True);"
        "p.write_text('complete',encoding='utf-8')"
    )
    stages = [
        {
            "name": "evaluate",
            "command": [sys.executable, "-c", guarded_code],
            "inputs": [{"kind": "collection_config", "path": "{config_path}"}],
            "outputs": [{"kind": "benchmark_report", "path": output}],
        },
        {
            "name": "summarize",
            "depends_on": ["evaluate"],
            "command": [sys.executable, "-c", dependent_code],
            "inputs": [{"kind": "benchmark_report", "path": output}],
            "outputs": [{"kind": "other", "path": summary_path}],
        },
    ]
    matrix_path = write_matrix(
        tmp_path,
        axes={"quality": [0.9], "seed": [1]},
        stages=stages,
    )
    plan = build_experiment_plan(matrix_path)
    runner = ExperimentRunner(tmp_path, tmp_path / "state")

    failed = runner.run(plan, confirm_plan_id=plan.plan_id)
    assert failed.trials[0].state is ExperimentTrialState.FAILED
    assert failed.trials[0].stages[0].state is ExperimentStageState.FAILED
    assert failed.trials[0].stages[1].state is ExperimentStageState.BLOCKED

    (tmp_path / marker).write_text("ready", encoding="utf-8")
    completed = runner.run(plan, confirm_plan_id=plan.plan_id)
    assert completed.trials[0].state is ExperimentTrialState.COMPLETE
    assert all(
        stage.state is ExperimentStageState.SUCCEEDED for stage in completed.trials[0].stages
    )
    attempt_keys = runner.store.list(
        f"{plan.plan_id}/{plan.trials[0].trial_id}/stages/evaluate/attempts/"
    )
    assert len(attempt_keys) == 1


def test_regression_gate_and_cli_preview_confirmation_and_outputs(tmp_path: Path) -> None:
    write_base_config(tmp_path)
    matrix_path = write_matrix(
        tmp_path,
        axes={"quality": [0.5], "seed": [1]},
    )
    plan_path = tmp_path / "plan.json"
    state_dir = tmp_path / "state"
    report_path = tmp_path / "report.json"
    runner = CliRunner()

    planned = runner.invoke(
        app,
        ["experiment-plan", str(matrix_path), "--output", str(plan_path)],
    )
    assert planned.exit_code == 0
    plan = ExperimentPlan.model_validate_json(plan_path.read_bytes())

    preview = runner.invoke(
        app,
        [
            "experiment-run",
            str(plan_path),
            "--root",
            str(tmp_path),
            "--state-dir",
            str(state_dir),
            "--fail-on-incomplete",
        ],
    )
    missing_confirmation = runner.invoke(
        app,
        [
            "experiment-run",
            str(plan_path),
            "--root",
            str(tmp_path),
            "--state-dir",
            str(state_dir),
            "--execute",
        ],
    )
    executed = runner.invoke(
        app,
        [
            "experiment-run",
            str(plan_path),
            "--root",
            str(tmp_path),
            "--state-dir",
            str(state_dir),
            "--execute",
            "--confirm-plan-id",
            plan.plan_id,
            "--fail-on-regression",
            "--output",
            str(report_path),
        ],
    )

    assert preview.exit_code == 1
    assert missing_confirmation.exit_code == 2
    assert executed.exit_code == 1
    report = ExperimentReport.model_validate_json(report_path.read_bytes())
    assert report.state_counts[ExperimentTrialState.REGRESSION.value] == 1
    assert report.failed_gate_count == 1
    assert report.canonical_bytes() + b"\n" == report_path.read_bytes()


def test_artifact_evidence_links_dataset_rollout_checkpoint_batch_and_comparison(
    tmp_path: Path,
) -> None:
    write_base_config(tmp_path)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"task_id":"task-1"}\n', encoding="utf-8")
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    file_manifest = DatasetFileManifest(
        relative_path="dataset.jsonl",
        size_bytes=dataset_path.stat().st_size,
        sha256=dataset_digest,
        record_count=1,
        unique_id_count=1,
        duplicate_id_count=0,
        record_ids_digest="1" * 64,
    )
    split_manifest = DatasetManifest(
        manifest_id="dataset_" + "1" * 24,
        name="fixture",
        split="train",
        format="jsonl",
        id_field="task_id",
        files=(file_manifest,),
        record_count=1,
        unique_id_count=1,
        duplicate_id_count=0,
        record_ids_digest="1" * 64,
        record_ids=("task-1",),
        content_digest=dataset_digest,
    )
    collection = DatasetCollectionManifest(
        collection_id="collection_" + "2" * 24,
        name="fixture",
        splits={"train": split_manifest},
        total_record_count=1,
        total_unique_id_count=1,
        record_ids_digest="1" * 64,
    )
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest_path.write_bytes(collection.canonical_bytes() + b"\n")

    rollout = RolloutPlanBuilder().build(
        (
            TaskSpec(
                task_id="task-1",
                messages=(Message(role=MessageRole.USER, content="question"),),
                verifier=VerifierSpec(kind="exact_match"),
            ),
        ),
        policy_version="fixture-policy",
        source_sha256=dataset_digest,
        config_digest="3" * 64,
        rollouts_per_task=2,
        seed=1,
    )
    rollout_path = tmp_path / "rollout-plan.json"
    rollout_path.write_bytes(rollout.canonical_bytes() + b"\n")

    checkpoint_payload = tmp_path / "model.bin"
    checkpoint_payload.write_bytes(b"checkpoint")
    checkpoint_digest = hashlib.sha256(checkpoint_payload.read_bytes()).hexdigest()
    checkpoint = CheckpointManifest(
        checkpoint_id="checkpoint-fixture",
        run_id="run-fixture",
        step=10,
        policy_version="fixture-policy",
        config_digest="3" * 64,
        artifacts=(
            CheckpointArtifact(
                name="model",
                uri=str(checkpoint_payload),
                location=ArtifactLocation.LOCAL,
                sha256=checkpoint_digest,
                size_bytes=checkpoint_payload.stat().st_size,
            ),
        ),
        created_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        dataset_manifest_digest=dataset_digest,
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_bytes(checkpoint.canonical_bytes() + b"\n")

    refs = tuple(
        TrainerTrajectoryRef(
            trajectory_id=f"traj-{index}",
            task_id="task-1",
            group_id="group-1",
            content_digest=str(index) * 64,
            origin=DataOrigin.ON_POLICY,
            reward=float(index),
        )
        for index in (1, 2)
    )
    trainer_batch = TrainerBatchManifest(
        batch_id="batch_" + "4" * 24,
        created_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        policy_version="fixture-policy",
        environment_versions=("fixture-env",),
        source_run_id="run-fixture",
        group_size=2,
        group_count=1,
        trajectory_count=2,
        payload_key="trainer/payload.jsonl",
        payload_sha256="5" * 64,
        payload_size_bytes=10,
        trajectories=refs,
    )
    batch_path = tmp_path / "trainer-batch.json"
    batch_path.write_bytes(trainer_batch.canonical_bytes() + b"\n")

    comparison = BenchmarkComparisonReport(
        comparison_id="comparison-fixture",
        benchmark="fixture",
        baseline_name="baseline",
        candidate_name="candidate",
        created_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        matched_task_count=1,
        baseline_policy_versions=("baseline",),
        candidate_policy_versions=("candidate",),
        metrics={
            "task_pass_rate": PairedMetricDelta(
                baseline_mean=0.5,
                candidate_mean=0.6,
                absolute_delta=0.1,
                relative_delta=0.2,
                confidence_low=0.01,
                confidence_high=0.19,
                confidence_level=0.95,
                sample_count=1,
                unit="ratio",
            )
        },
    )
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_bytes(comparison.canonical_bytes() + b"\n")

    stages = [
        {
            "name": "lineage",
            "command": [sys.executable, "-c", "pass"],
            "inputs": [
                {"kind": "dataset", "path": "dataset.jsonl"},
                {
                    "kind": "dataset_collection_manifest",
                    "path": "dataset-manifest.json",
                },
                {"kind": "rollout_plan", "path": "rollout-plan.json"},
                {"kind": "checkpoint_manifest", "path": "checkpoint.json"},
                {"kind": "trainer_batch_manifest", "path": "trainer-batch.json"},
            ],
            "outputs": [{"kind": "comparison_report", "path": "comparison.json"}],
            "gates": [
                {
                    "name": "non-regression",
                    "artifact_path": "comparison.json",
                    "metric": "task_pass_rate",
                    "statistic": "confidence_low",
                    "minimum": 0.0,
                }
            ],
        }
    ]
    matrix_path = write_matrix(
        tmp_path,
        axes={"quality": [0.9], "seed": [1]},
        stages=stages,
    )
    plan = build_experiment_plan(matrix_path)
    report = ExperimentRunner(tmp_path, tmp_path / "state").run(
        plan,
        confirm_plan_id=plan.plan_id,
    )

    stage = report.trials[0].stages[0]
    assert report.trials[0].state is ExperimentTrialState.COMPLETE
    assert tuple(item.content_id for item in stage.inputs) == (
        dataset_digest,
        collection.collection_id,
        rollout.plan_id,
        checkpoint.checkpoint_id,
        trainer_batch.batch_id,
    )
    assert stage.outputs[0].content_id == comparison.comparison_id
    assert stage.gates[0].passed


def test_experiment_contracts_reject_mutated_content_identities(tmp_path: Path) -> None:
    write_base_config(tmp_path)
    matrix_path = write_matrix(
        tmp_path,
        axes={"quality": [0.9], "seed": [1]},
    )
    plan = build_experiment_plan(matrix_path)
    runner = ExperimentRunner(tmp_path, tmp_path / "state")
    report = runner.run(plan, confirm_plan_id=plan.plan_id)
    trial = plan.trials[0]
    stage = trial.stages[0]
    record_path = (
        tmp_path / "state" / plan.plan_id / trial.trial_id / "stages" / stage.name / "record.json"
    )
    record = ExperimentStageRecord.model_validate_json(record_path.read_bytes())

    plan_payload = plan.model_dump(mode="python")
    with pytest.raises(ValueError, match="plan ID"):
        ExperimentPlan.model_validate({**plan_payload, "plan_id": "experiment_plan_" + "0" * 24})
    trial_payload = trial.model_dump(mode="python")
    mutated_trial = {**trial_payload, "config_digest": "0" * 64}
    with pytest.raises(ValueError, match="config digest"):
        type(trial).model_validate(mutated_trial)
    record_payload = record.model_dump(mode="python")
    with pytest.raises(ValueError, match="record ID"):
        ExperimentStageRecord.model_validate(
            {**record_payload, "record_id": "experiment_stage_" + "0" * 24}
        )
    report_payload = report.model_dump(mode="python")
    for update, message in (
        ({"state_counts": {}}, "state counts"),
        ({"failed_gate_count": 2}, "gate counts"),
        ({"report_id": "experiment_report_" + "0" * 24}, "report ID"),
    ):
        with pytest.raises(ValueError, match=message):
            ExperimentReport.model_validate({**report_payload, **update})

    parsed = json.loads(report.canonical_bytes())
    assert parsed["report_id"] == report.report_id
    assert record.completed_at.tzinfo is not None
    assert record.completed_at.utcoffset() is not None

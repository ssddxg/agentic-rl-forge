from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import CheckpointArtifact, CheckpointManifest
from agentic_rl_forge.experiments import (
    ExperimentAnalysis,
    ExperimentAnalyzer,
    ExperimentArtifactKind,
    ExperimentBaselineDelta,
    ExperimentIndexBuilder,
    ExperimentIndexIssue,
    ExperimentIndexRow,
    ExperimentMetricDefinition,
    ExperimentMetricStatistic,
    ExperimentObjective,
    ExperimentObjectiveDirection,
    ExperimentOperationsIndex,
    ExperimentPromoter,
    ExperimentPromotionArtifactRef,
    ExperimentPromotionArtifactRole,
    ExperimentPromotionArtifactScope,
    ExperimentPromotionCheck,
    ExperimentPromotionError,
    ExperimentPromotionPolicy,
    ExperimentPromotionPreview,
    ExperimentPromotionRecord,
    ExperimentRankedTrial,
    ExperimentRankingSpec,
    ExperimentReportReference,
    ExperimentReproducibilityManifest,
    ExperimentRunner,
    ExperimentTrialState,
    build_experiment_plan,
    render_experiment_csv,
    render_experiment_html,
    render_experiment_markdown,
)


def write_base_config(root: Path) -> None:
    (root / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "operations-fixture",
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


def report_command(benchmark_output: str, comparison_output: str) -> list[str]:
    code = (
        "from datetime import datetime,timezone;"
        "from pathlib import Path;"
        "from agentic_rl_forge.evaluation import "
        "BenchmarkComparisonReport,BenchmarkReport,MetricValue,PairedMetricDelta;"
        f"b=Path({benchmark_output!r});b.parent.mkdir(parents=True,exist_ok=True);"
        "quality=float('{quality}');cost=float('{cost}');"
        "r=BenchmarkReport(report_id='report_fixture',benchmark='fixture',"
        "created_at=datetime(2026,9,18,tzinfo=timezone.utc),task_count=1,group_count=1,"
        "attempt_count=1,policy_versions=('fixture-policy',),"
        "environment_versions=('fixture-env',),status_counts=dict(succeeded=1),"
        "metrics=dict(score=MetricValue(value=quality,numerator=quality,denominator=1,"
        "unit='ratio'),cost=MetricValue(value=cost,numerator=cost,denominator=1,"
        "unit='seconds')),group_diagnostics=());b.write_bytes(r.canonical_bytes()+b'\\n');"
        f"c=Path({comparison_output!r});c.parent.mkdir(parents=True,exist_ok=True);"
        "d=PairedMetricDelta(baseline_mean=0.5,candidate_mean=quality,"
        "absolute_delta=quality-0.5,relative_delta=(quality-0.5)/0.5,"
        "confidence_low=quality-0.55,confidence_high=quality-0.45,"
        "confidence_level=0.95,sample_count=1,unit='ratio');"
        "x=BenchmarkComparisonReport(comparison_id='comparison_fixture',benchmark='fixture',"
        "baseline_name='baseline',candidate_name='candidate',"
        "created_at=datetime(2026,9,18,tzinfo=timezone.utc),matched_task_count=1,"
        "baseline_policy_versions=('baseline',),candidate_policy_versions=('candidate',),"
        "metrics=dict(score=d));c.write_bytes(x.canonical_bytes()+b'\\n')"
    )
    return [sys.executable, "-c", code]


def write_matrix(root: Path) -> Path:
    benchmark_output = "outputs/{trial_id}/benchmark.json"
    comparison_output = "outputs/{trial_id}/comparison.json"
    selected = {(0.9, 3), (0.8, 1), (0.7, 2)}
    exclude = [
        {"quality": quality, "cost": cost}
        for quality in (0.9, 0.8, 0.7)
        for cost in (3, 1, 2)
        if (quality, cost) not in selected
    ]
    path = root / "matrix.yaml"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": "operations-ranking",
        "base_config": "base.yaml",
        "fixed_parameters": {"label": "=1+1 <unsafe>|value"},
        "axes": {"quality": [0.9, 0.8, 0.7], "cost": [3, 1, 2]},
        "exclude": exclude,
        "stages": [
            {
                "name": "evaluate",
                "command": report_command(benchmark_output, comparison_output),
                "inputs": [
                    {"kind": "collection_config", "path": "{config_path}"},
                    {"kind": "dataset", "path": "dataset.jsonl"},
                    {"kind": "checkpoint_manifest", "path": "checkpoint.json"},
                ],
                "outputs": [
                    {"kind": "benchmark_report", "path": benchmark_output},
                    {"kind": "comparison_report", "path": comparison_output},
                ],
            }
        ],
        "max_trials": 9,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def completed_index(
    tmp_path: Path,
    *,
    include_issue: bool = True,
) -> tuple[ExperimentOperationsIndex, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    write_base_config(tmp_path)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"task_id":"task-1"}\n', encoding="utf-8")
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    checkpoint_payload = tmp_path / "checkpoint.bin"
    checkpoint_payload.write_bytes(b"verified-checkpoint")
    checkpoint = CheckpointManifest(
        checkpoint_id="checkpoint-operations-fixture",
        run_id="run-operations-fixture",
        step=10,
        policy_version="fixture-policy",
        config_digest="1" * 64,
        artifacts=(CheckpointArtifact.from_file("model", checkpoint_payload),),
        created_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        dataset_manifest_digest=dataset_digest,
    )
    (tmp_path / "checkpoint.json").write_bytes(checkpoint.canonical_bytes() + b"\n")
    plan = build_experiment_plan(write_matrix(tmp_path))
    state = tmp_path / "state"
    report = ExperimentRunner(tmp_path, state).run(
        plan,
        confirm_plan_id=plan.plan_id,
        max_workers=3,
    )
    assert report.state_counts[ExperimentTrialState.COMPLETE.value] == 3
    if include_issue:
        (state / "notes.txt").write_text("operator note", encoding="utf-8")
    return ExperimentIndexBuilder(tmp_path, state).build(), state


def objective(
    index: ExperimentOperationsIndex,
    metric: str,
    direction: ExperimentObjectiveDirection,
) -> ExperimentObjective:
    definition = next(
        item
        for item in index.metric_definitions
        if item.metric == metric and item.statistic.value == "value"
    )
    return ExperimentObjective(metric_id=definition.metric_id, direction=direction)


def objectives(*items: ExperimentObjective) -> tuple[ExperimentObjective, ...]:
    return tuple(sorted(items, key=lambda item: item.metric_id))


def test_index_discovers_reports_parameters_metrics_and_issues(tmp_path: Path) -> None:
    index, state = completed_index(tmp_path)
    repeated = ExperimentIndexBuilder(tmp_path, state).build()

    assert index == repeated
    assert index.canonical_bytes() == repeated.canonical_bytes()
    assert len(index.reports) == 1
    assert len(index.rows) == 3
    assert index.parameter_columns == ("cost", "label", "quality")
    assert index.state_counts[ExperimentTrialState.COMPLETE.value] == 3
    assert [(item.path, item.detail) for item in index.issues] == [
        ("notes.txt", "unrecognized experiment state entry")
    ]
    definitions = {(item.metric, item.statistic.value) for item in index.metric_definitions}
    assert ("score", "value") in definitions
    assert ("cost", "value") in definitions
    assert {
        "baseline_mean",
        "candidate_mean",
        "absolute_delta",
        "relative_delta",
        "confidence_low",
        "confidence_high",
    } <= {statistic for metric, statistic in definitions if metric == "score"}
    assert all(len(row.metric_values) == 8 for row in index.rows)

    plan_directory = state / index.reports[0].plan_id
    alias = state / ("experiment_plan_" + "0" * 24)
    try:
        alias.symlink_to(plan_directory, target_is_directory=True)
    except OSError as error:
        # Creating symlinks requires SeCreateSymbolicLinkPrivilege (or
        # Developer Mode) on Windows. The production scanner still rejects
        # symlink entries when the platform can create them.
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or elevated privilege")
        raise
    with_alias = ExperimentIndexBuilder(tmp_path, state).build()
    assert any(item.path == alias.name for item in with_alias.issues)
    assert len(with_alias.reports) == 1


def test_analysis_ranks_compares_baseline_and_builds_pareto_dashboard(tmp_path: Path) -> None:
    index, _ = completed_index(tmp_path)
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    cost = objective(index, "cost", ExperimentObjectiveDirection.MINIMIZE)
    baseline = next(row for row in index.rows if row.parameters["quality"] == 0.9)
    spec = ExperimentRankingSpec(
        objectives=objectives(score, cost),
        baseline_plan_id=baseline.plan_id,
        baseline_trial_id=baseline.trial_id,
    )

    analysis = ExperimentAnalyzer().analyze(index, spec)
    repeated = ExperimentAnalyzer().analyze(index, spec)

    assert analysis == repeated
    assert [item.parameters["quality"] for item in analysis.trials] == [0.8, 0.9, 0.7]
    assert [item.rank for item in analysis.trials] == [1, 2, 3]
    assert {item.parameters["quality"] for item in analysis.trials if item.pareto_front} == {
        0.8,
        0.9,
    }
    candidate = analysis.trials[0]
    improvements = {item.metric_id: item.improvement for item in candidate.baseline_deltas}
    assert improvements[score.metric_id] == pytest.approx(-0.1)
    assert improvements[cost.metric_id] == pytest.approx(2.0)

    markdown = render_experiment_markdown(index, analysis)
    rendered_html = render_experiment_html(index, analysis)
    rendered_csv = render_experiment_csv(index)
    assert markdown == render_experiment_markdown(index, analysis)
    assert rendered_html == render_experiment_html(index, analysis)
    assert "Improvement: evaluate[0].score.value" in markdown
    assert "=1+1 &lt;unsafe&gt;\\|value" in markdown
    assert "=1+1 &lt;unsafe&gt;|value" in rendered_html
    assert "<script" not in rendered_html
    assert rendered_csv == render_experiment_csv(index)
    assert "parameter:quality" in rendered_csv.splitlines()[0]
    assert "'=1+1 <unsafe>|value" in rendered_csv


def test_operations_cli_writes_canonical_index_analysis_and_static_exports(
    tmp_path: Path,
) -> None:
    index, state = completed_index(tmp_path)
    runner = CliRunner()
    index_path = tmp_path / "index.json"
    csv_path = tmp_path / "index.csv"

    indexed = runner.invoke(
        app,
        [
            "experiment-index",
            "--root",
            str(tmp_path),
            "--state-dir",
            str(state),
            "--output",
            str(index_path),
            "--csv-output",
            str(csv_path),
            "--fail-on-issues",
        ],
    )
    assert indexed.exit_code == 1
    loaded_index = ExperimentOperationsIndex.model_validate_json(index_path.read_bytes())
    assert loaded_index == index
    assert index_path.read_bytes() == loaded_index.canonical_bytes() + b"\n"
    assert csv_path.read_text(encoding="utf-8") == render_experiment_csv(index)

    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    cost = objective(index, "cost", ExperimentObjectiveDirection.MINIMIZE)
    baseline = next(row for row in index.rows if row.parameters["quality"] == 0.9)
    analysis_path = tmp_path / "analysis.json"
    markdown_path = tmp_path / "dashboard.md"
    html_path = tmp_path / "dashboard.html"
    analyzed = runner.invoke(
        app,
        [
            "experiment-analyze",
            str(index_path),
            "--objective",
            f"{score.metric_id}:maximize",
            "--objective",
            f"{cost.metric_id}:minimize:2",
            "--baseline-plan-id",
            baseline.plan_id,
            "--baseline-trial-id",
            baseline.trial_id,
            "--output",
            str(analysis_path),
            "--markdown-output",
            str(markdown_path),
            "--html-output",
            str(html_path),
        ],
    )

    assert analyzed.exit_code == 0, analyzed.output
    analysis = ExperimentAnalysis.model_validate_json(analysis_path.read_bytes())
    assert analysis.index_id == index.index_id
    assert analysis_path.read_bytes() == analysis.canonical_bytes() + b"\n"
    assert markdown_path.read_text(encoding="utf-8").startswith("# Experiment operations report")
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_operations_contracts_reject_mutated_identities_and_invalid_policy(
    tmp_path: Path,
) -> None:
    index, _ = completed_index(tmp_path)
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    analysis = ExperimentAnalyzer().analyze(index, ExperimentRankingSpec(objectives=(score,)))

    index_payload = index.model_dump(mode="python")
    with pytest.raises(ValueError, match="index ID"):
        ExperimentOperationsIndex.model_validate(
            {**index_payload, "index_id": "experiment_index_" + "0" * 24}
        )
    analysis_payload = analysis.model_dump(mode="python")
    with pytest.raises(ValueError, match="analysis ID"):
        ExperimentAnalysis.model_validate(
            {**analysis_payload, "analysis_id": "experiment_analysis_" + "0" * 24}
        )
    with pytest.raises(ValueError, match="completed trial states"):
        ExperimentRankingSpec(
            objectives=(score,),
            eligible_states=(ExperimentTrialState.FAILED,),
        )
    with pytest.raises(ValueError, match="unknown experiment objective"):
        ExperimentAnalyzer().analyze(
            index,
            ExperimentRankingSpec(
                objectives=(
                    ExperimentObjective(
                        metric_id="experiment_metric_" + "0" * 24,
                        direction=ExperimentObjectiveDirection.MAXIMIZE,
                    ),
                )
            ),
        )


def test_operations_contracts_fail_closed_on_malformed_tables_and_rankings(
    tmp_path: Path,
) -> None:
    index, _ = completed_index(tmp_path)
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    cost = objective(index, "cost", ExperimentObjectiveDirection.MINIMIZE)
    baseline = index.rows[0]
    analysis = ExperimentAnalyzer().analyze(
        index,
        ExperimentRankingSpec(
            objectives=objectives(score, cost),
            baseline_plan_id=baseline.plan_id,
            baseline_trial_id=baseline.trial_id,
        ),
    )

    with pytest.raises(ValueError, match="safe relative"):
        ExperimentIndexIssue(path="../escape", detail="invalid")

    report_payload = index.reports[0].model_dump(mode="python")
    for update, message in (
        ({"state_counts": {}}, "state counts"),
        ({"trial_count": 4}, "trial count"),
        ({"gate_count": 0, "failed_gate_count": 1}, "gate counts"),
    ):
        with pytest.raises(ValueError, match=message):
            ExperimentReportReference.model_validate({**report_payload, **update})

    benchmark_definition = next(
        item
        for item in index.metric_definitions
        if item.artifact_kind is ExperimentArtifactKind.BENCHMARK_REPORT
    )
    comparison_definition = next(
        item
        for item in index.metric_definitions
        if item.artifact_kind is ExperimentArtifactKind.COMPARISON_REPORT
    )
    definition_payload = benchmark_definition.model_dump(mode="python")
    invalid_definitions = (
        ({**definition_payload, "statistic": ExperimentMetricStatistic.ABSOLUTE_DELTA}, "value"),
        (
            {
                **comparison_definition.model_dump(mode="python"),
                "statistic": ExperimentMetricStatistic.VALUE,
            },
            "comparison statistics",
        ),
        ({**definition_payload, "artifact_kind": ExperimentArtifactKind.OTHER}, "benchmark"),
        ({**definition_payload, "metric_id": "experiment_metric_" + "0" * 24}, "metric ID"),
    )
    for payload, message in invalid_definitions:
        with pytest.raises(ValueError, match=message):
            ExperimentMetricDefinition.model_validate(payload)

    row_payload = index.rows[0].model_dump(mode="python")
    metric_items = list(row_payload["metric_values"].items())
    invalid_rows = (
        ({**row_payload, "parameters": {"z": 1, "a": 2}}, "parameters"),
        ({**row_payload, "metric_values": dict(reversed(metric_items))}, "metrics must be sorted"),
        ({**row_payload, "metric_values": {"invalid": 1.0}}, "invalid metric ID"),
        (
            {**row_payload, "metric_values": {benchmark_definition.metric_id: float("inf")}},
            "finite",
        ),
    )
    for payload, message in invalid_rows:
        with pytest.raises(ValueError, match=message):
            ExperimentIndexRow.model_validate(payload)

    index_payload = index.model_dump(mode="python")
    first_row = index.rows[0]
    wrong_report_row = first_row.model_copy(update={"report_id": "experiment_report_" + "0" * 24})
    wrong_name_row = first_row.model_copy(update={"experiment_name": "different"})
    unknown_metric_row = first_row.model_copy(
        update={
            "metric_values": dict(
                sorted({**first_row.metric_values, "experiment_metric_" + "0" * 24: 1.0}.items())
            )
        }
    )
    regressed_row = first_row.model_copy(update={"state": ExperimentTrialState.REGRESSION})
    invalid_indexes = (
        ({**index_payload, "reports": (index.reports[0], index.reports[0])}, "reports"),
        ({**index_payload, "rows": (index.rows[0], index.rows[0])}, "rows"),
        (
            {
                **index_payload,
                "metric_definitions": (
                    index.metric_definitions[0],
                    index.metric_definitions[0],
                ),
            },
            "metric definitions",
        ),
        (
            {**index_payload, "parameter_columns": tuple(reversed(index.parameter_columns))},
            "columns",
        ),
        ({**index_payload, "issues": (index.issues[0], index.issues[0])}, "issues"),
        ({**index_payload, "rows": (wrong_report_row, *index.rows[1:])}, "reference its report"),
        ({**index_payload, "rows": (wrong_name_row, *index.rows[1:])}, "name differs"),
        ({**index_payload, "rows": (unknown_metric_row, *index.rows[1:])}, "unknown metric"),
        ({**index_payload, "rows": index.rows[:-1]}, "trial count differs"),
        ({**index_payload, "rows": (regressed_row, *index.rows[1:])}, "states differ"),
        ({**index_payload, "parameter_columns": index.parameter_columns[:-1]}, "columns"),
        ({**index_payload, "state_counts": {}}, "state counts"),
    )
    for payload, message in invalid_indexes:
        with pytest.raises(ValueError, match=message):
            ExperimentOperationsIndex.model_validate(payload)

    with pytest.raises(ValueError, match="finite"):
        ExperimentObjective(
            metric_id=score.metric_id,
            direction=ExperimentObjectiveDirection.MAXIMIZE,
            weight=float("inf"),
        )
    ranking_errors = (
        (
            {
                "objectives": tuple(reversed(objectives(score, cost))),
            },
            "sorted and unique",
        ),
        (
            {
                "objectives": (score,),
                "eligible_states": (
                    ExperimentTrialState.REGRESSION,
                    ExperimentTrialState.COMPLETE,
                ),
            },
            "states must be sorted",
        ),
        ({"objectives": (score,), "experiment_names": ("z", "a")}, "names"),
        (
            {
                "objectives": (score,),
                "plan_ids": (
                    "experiment_plan_" + "f" * 24,
                    "experiment_plan_" + "0" * 24,
                ),
            },
            "plan IDs",
        ),
        ({"objectives": (score,), "baseline_plan_id": baseline.plan_id}, "both plan and trial"),
    )
    for payload, message in ranking_errors:
        with pytest.raises(ValueError, match=message):
            ExperimentRankingSpec.model_validate(payload)

    with pytest.raises(ValueError, match="raw delta"):
        ExperimentBaselineDelta(
            metric_id=score.metric_id,
            baseline_value=0.5,
            candidate_value=0.8,
            raw_delta=0.1,
            improvement=0.3,
        )

    ranked_payload = analysis.trials[0].model_dump(mode="python")
    invalid_ranked = (
        ({**ranked_payload, "parameters": {"z": 1, "a": 2}}, "parameters"),
        (
            {
                **ranked_payload,
                "objective_values": dict(
                    reversed(list(ranked_payload["objective_values"].items()))
                ),
            },
            "objective values must be sorted",
        ),
        ({**ranked_payload, "score": None}, "finite rank and score"),
        ({**ranked_payload, "eligible": False}, "ineligible"),
    )
    for payload, message in invalid_ranked:
        with pytest.raises(ValueError, match=message):
            ExperimentRankedTrial.model_validate(payload)

    analysis_payload = analysis.model_dump(mode="python")
    altered_score = analysis.trials[0].model_copy(update={"score": 0.123})
    altered_pareto = analysis.trials[0].model_copy(
        update={"pareto_front": not analysis.trials[0].pareto_front}
    )
    delta = analysis.trials[0].baseline_deltas[0]
    altered_delta = delta.model_copy(update={"improvement": delta.improvement + 1.0})
    altered_baseline = analysis.trials[0].model_copy(
        update={"baseline_deltas": (altered_delta, *analysis.trials[0].baseline_deltas[1:])}
    )
    invalid_analyses = (
        ({**analysis_payload, "eligible_count": 99}, "counts"),
        ({**analysis_payload, "trials": (altered_score, *analysis.trials[1:])}, "score"),
        ({**analysis_payload, "trials": (altered_pareto, *analysis.trials[1:])}, "Pareto"),
        (
            {**analysis_payload, "trials": (altered_baseline, *analysis.trials[1:])},
            "baseline delta",
        ),
    )
    for payload, message in invalid_analyses:
        with pytest.raises(ValueError, match=message):
            ExperimentAnalysis.model_validate(payload)


def test_promotion_revalidates_lineage_requires_confirmation_and_recovers_sidecars(
    tmp_path: Path,
) -> None:
    index, state = completed_index(tmp_path, include_issue=False)
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    analysis = ExperimentAnalyzer().analyze(index, ExperimentRankingSpec(objectives=(score,)))
    selected = analysis.trials[0]
    promotion_root = tmp_path / "promotions"
    promoter = ExperimentPromoter(tmp_path, state, promotion_root)
    policy = ExperimentPromotionPolicy(maximum_rank=1, require_pareto_front=True)

    preview = promoter.preview(
        index,
        analysis,
        promotion_name="search-r1-candidate",
        plan_id=selected.plan_id,
        trial_id=selected.trial_id,
        policy=policy,
    )
    repeated_preview = promoter.preview(
        index,
        analysis,
        promotion_name="search-r1-candidate",
        plan_id=selected.plan_id,
        trial_id=selected.trial_id,
        policy=policy,
    )

    assert preview == repeated_preview
    assert preview.eligible
    assert not promotion_root.exists()
    assert preview.failed_check_count == 0
    assert preview.manifest.checkpoint_ids == ("checkpoint-operations-fixture",)
    roles = {role.value for artifact in preview.manifest.artifacts for role in artifact.roles}
    assert {"config", "input", "output", "checkpoint_payload"} <= roles
    assert any(item.locator == "checkpoint.bin" for item in preview.manifest.artifacts)
    assert all(item.passed for item in preview.checks)

    with pytest.raises(ExperimentPromotionError, match="confirmation"):
        promoter.promote(
            index,
            analysis,
            preview,
            confirm_preview_id="experiment_promotion_preview_" + "0" * 24,
            operator="release-operator",
            reason="approve the verified experiment candidate",
            approvers=("reviewer@example.com",),
        )

    checkpoint_payload = tmp_path / "checkpoint.bin"
    checkpoint_payload.write_bytes(b"drifted-checkpoint")
    with pytest.raises(ExperimentPromotionError, match="state changed"):
        promoter.promote(
            index,
            analysis,
            preview,
            confirm_preview_id=preview.preview_id,
            operator="release-operator",
            reason="approve the verified experiment candidate",
            approvers=("reviewer@example.com",),
        )
    checkpoint_payload.write_bytes(b"verified-checkpoint")

    record = promoter.promote(
        index,
        analysis,
        preview,
        confirm_preview_id=preview.preview_id,
        operator="release-operator",
        reason="approve <script> and `document` after independent review",
        approvers=("reviewer@example.com",),
    )
    repeated_record = promoter.promote(
        index,
        analysis,
        preview,
        confirm_preview_id=preview.preview_id,
        operator="release-operator",
        reason="approve <script> and `document` after independent review",
        approvers=("reviewer@example.com",),
    )

    assert record == repeated_record
    promotion_path = promotion_root / preview.promotion_name
    assert (promotion_path / "decision.json").read_bytes() == record.canonical_bytes() + b"\n"
    assert (promotion_path / "record.json").read_bytes() == record.canonical_bytes() + b"\n"
    assert (promotion_path / "manifest.json").read_bytes() == (
        preview.manifest.canonical_bytes() + b"\n"
    )
    model_card = (promotion_path / "model-card.md").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in model_card
    assert "&#96;document&#96;" in model_card
    assert "<script>" not in model_card
    assert record.model_card_sha256 == hashlib.sha256(model_card.encode("utf-8")).hexdigest()
    assert {
        "analysis.json",
        "decision.json",
        "index.json",
        "manifest.json",
        "model-card.md",
        "plan.json",
        "record.json",
        "report.json",
    } == {item.name for item in promotion_path.iterdir()}

    (promotion_path / "record.json").unlink()
    recovered = promoter.promote(
        index,
        analysis,
        preview,
        confirm_preview_id=preview.preview_id,
        operator="release-operator",
        reason="approve <script> and `document` after independent review",
        approvers=("reviewer@example.com",),
    )
    assert recovered == record
    assert (promotion_path / "record.json").is_file()

    with pytest.raises(ExperimentPromotionError, match="different immutable decision"):
        promoter.promote(
            index,
            analysis,
            preview,
            confirm_preview_id=preview.preview_id,
            operator="another-operator",
            reason="approve the same candidate under another decision",
            approvers=("reviewer@example.com",),
        )


def test_promotion_approval_policy_and_cli_preview_execute(tmp_path: Path) -> None:
    index, state = completed_index(tmp_path, include_issue=False)
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    analysis = ExperimentAnalyzer().analyze(index, ExperimentRankingSpec(objectives=(score,)))
    selected = analysis.trials[0]
    promoter = ExperimentPromoter(tmp_path, state, tmp_path / "api-promotions")
    policy = ExperimentPromotionPolicy(minimum_approvals=2)
    preview = promoter.preview(
        index,
        analysis,
        promotion_name="approval-policy",
        plan_id=selected.plan_id,
        trial_id=selected.trial_id,
        policy=policy,
    )

    with pytest.raises(ValueError, match="minimum approval"):
        promoter.promote(
            index,
            analysis,
            preview,
            confirm_preview_id=preview.preview_id,
            operator="release-operator",
            reason="require two independent approvals",
            approvers=("reviewer-one",),
        )
    with pytest.raises(ValueError, match="independent approval"):
        promoter.promote(
            index,
            analysis,
            preview,
            confirm_preview_id=preview.preview_id,
            operator="release-operator",
            reason="require two independent approvals",
            approvers=("release-operator", "reviewer-one"),
        )

    index_path = tmp_path / "promotion-index.json"
    analysis_path = tmp_path / "promotion-analysis.json"
    preview_path = tmp_path / "promotion-preview.json"
    record_path = tmp_path / "promotion-record.json"
    index_path.write_bytes(index.canonical_bytes() + b"\n")
    analysis_path.write_bytes(analysis.canonical_bytes() + b"\n")
    runner = CliRunner()
    common = [
        "experiment-promote",
        str(index_path),
        str(analysis_path),
        "--name",
        "cli-candidate",
        "--plan-id",
        selected.plan_id,
        "--trial-id",
        selected.trial_id,
        "--root",
        str(tmp_path),
        "--state-dir",
        str(state),
        "--promotion-dir",
        str(tmp_path / "cli-promotions"),
    ]
    previewed = runner.invoke(app, [*common, "--preview-output", str(preview_path)])

    assert previewed.exit_code == 0, previewed.output
    cli_preview = ExperimentPromotionPreview.model_validate_json(preview_path.read_bytes())
    assert cli_preview.eligible
    assert preview_path.read_bytes() == cli_preview.canonical_bytes() + b"\n"

    missing_confirmation = runner.invoke(app, [*common, "--execute"])
    assert missing_confirmation.exit_code == 2

    executed = runner.invoke(
        app,
        [
            *common,
            "--execute",
            "--confirm-preview-id",
            cli_preview.preview_id,
            "--operator",
            "cli-operator",
            "--reason",
            "approve the CLI candidate after review",
            "--approver",
            "cli-reviewer",
            "--record-output",
            str(record_path),
        ],
    )

    assert executed.exit_code == 0, executed.output
    cli_record = ExperimentPromotionRecord.model_validate_json(record_path.read_bytes())
    assert cli_record.preview.preview_id == cli_preview.preview_id
    assert record_path.read_bytes() == cli_record.canonical_bytes() + b"\n"

    dominated = analysis.trials[-1]
    ineligible_path = tmp_path / "ineligible-preview.json"
    ineligible = runner.invoke(
        app,
        [
            "experiment-promote",
            str(index_path),
            str(analysis_path),
            "--name",
            "dominated-candidate",
            "--plan-id",
            dominated.plan_id,
            "--trial-id",
            dominated.trial_id,
            "--root",
            str(tmp_path),
            "--state-dir",
            str(state),
            "--promotion-dir",
            str(tmp_path / "cli-promotions"),
            "--require-pareto-front",
            "--preview-output",
            str(ineligible_path),
            "--fail-on-ineligible",
        ],
    )
    assert ineligible.exit_code == 1
    assert not ExperimentPromotionPreview.model_validate_json(ineligible_path.read_bytes()).eligible


def test_promotion_contracts_and_policy_relaxations_fail_closed(tmp_path: Path) -> None:
    index, state = completed_index(tmp_path, include_issue=False)
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    analysis = ExperimentAnalyzer().analyze(index, ExperimentRankingSpec(objectives=(score,)))
    selected = analysis.trials[0]
    promoter = ExperimentPromoter(tmp_path, state, tmp_path / "contract-promotions")
    preview = promoter.preview(
        index,
        analysis,
        promotion_name="contract-candidate",
        plan_id=selected.plan_id,
        trial_id=selected.trial_id,
        policy=ExperimentPromotionPolicy(),
    )
    record = promoter.promote(
        index,
        analysis,
        preview,
        confirm_preview_id=preview.preview_id,
        operator="contract-operator",
        reason="validate promotion contract mutation handling",
        approvers=("contract-reviewer",),
    )

    reference = preview.manifest.artifacts[0]
    reference_payload = reference.model_dump(mode="python")
    invalid_references = (
        (
            {
                **reference_payload,
                "scope": ExperimentPromotionArtifactScope.REMOTE,
                "locator": "https://user:secret@example.com/model?token=secret",
            },
            "secret-free URI",
        ),
        ({**reference_payload, "locator": "../escape"}, "safe and relative"),
        ({**reference_payload, "stages": ("z", "a")}, "stages"),
        (
            {
                **reference_payload,
                "roles": (
                    ExperimentPromotionArtifactRole.OUTPUT,
                    ExperimentPromotionArtifactRole.INPUT,
                ),
            },
            "roles",
        ),
    )
    for payload, message in invalid_references:
        with pytest.raises(ValueError, match=message):
            ExperimentPromotionArtifactRef.model_validate(payload)

    manifest_payload = preview.manifest.model_dump(mode="python")
    invalid_manifests = (
        ({**manifest_payload, "parameters": {"z": 1, "a": 2}}, "parameters"),
        (
            {**manifest_payload, "artifacts": tuple(reversed(preview.manifest.artifacts))},
            "artifacts",
        ),
        ({**manifest_payload, "artifact_count": 999}, "artifact count"),
        ({**manifest_payload, "total_size_bytes": 0}, "byte count"),
        ({**manifest_payload, "checkpoint_ids": ()}, "checkpoint IDs"),
        (
            {**manifest_payload, "manifest_id": "experiment_repro_" + "0" * 24},
            "manifest ID",
        ),
    )
    for payload, message in invalid_manifests:
        with pytest.raises(ValueError, match=message):
            ExperimentReproducibilityManifest.model_validate(payload)

    with pytest.raises(ValueError, match="sorted and unique"):
        ExperimentPromotionPolicy(
            required_artifact_kinds=(
                ExperimentArtifactKind.CHECKPOINT_MANIFEST,
                ExperimentArtifactKind.BENCHMARK_REPORT,
            )
        )
    with pytest.raises(ValueError, match="evidence"):
        ExperimentPromotionCheck(code="check", passed=False, detail="invalid", evidence=("z", "a"))

    preview_payload = preview.model_dump(mode="python")
    invalid_previews = (
        ({**preview_payload, "checks": (preview.checks[0], preview.checks[0])}, "checks"),
        ({**preview_payload, "eligible": False}, "eligibility"),
        ({**preview_payload, "promotion_name": "different-name"}, "manifest identity"),
        (
            {
                **preview_payload,
                "preview_id": "experiment_promotion_preview_" + "0" * 24,
            },
            "preview ID",
        ),
    )
    for payload, message in invalid_previews:
        with pytest.raises(ValueError, match=message):
            ExperimentPromotionPreview.model_validate(payload)

    record_payload = record.model_dump(mode="python")
    invalid_records = (
        ({**record_payload, "operator": " bad "}, "printable identity"),
        ({**record_payload, "reason": "invalid\x00reason"}, "NUL"),
        ({**record_payload, "approvers": ("a", "a")}, "sorted and unique"),
        ({**record_payload, "approvers": ("bad\nidentity",)}, "printable identities"),
        (
            {**record_payload, "promotion_id": "experiment_promotion_" + "0" * 24},
            "promotion ID",
        ),
    )
    for payload, message in invalid_records:
        with pytest.raises(ValueError, match=message):
            ExperimentPromotionRecord.model_validate(payload)

    with pytest.raises(ValueError, match="outside experiment state"):
        ExperimentPromoter(tmp_path, state, state / "promotions")
    with pytest.raises(ValueError, match="promotion name"):
        promoter.preview(
            index,
            analysis,
            promotion_name="invalid/name",
            plan_id=selected.plan_id,
            trial_id=selected.trial_id,
            policy=ExperimentPromotionPolicy(),
        )
    mismatched_analysis = analysis.model_copy(update={"index_id": "experiment_index_" + "0" * 24})
    with pytest.raises(ValueError, match="another index"):
        promoter.preview(
            index,
            mismatched_analysis,
            promotion_name="index-mismatch",
            plan_id=selected.plan_id,
            trial_id=selected.trial_id,
            policy=ExperimentPromotionPolicy(),
        )
    with pytest.raises(ExperimentPromotionError, match="not present in the analysis"):
        promoter.preview(
            index,
            analysis,
            promotion_name="missing-trial",
            plan_id=selected.plan_id,
            trial_id="experiment_trial_" + "0" * 24,
            policy=ExperimentPromotionPolicy(),
        )

    dominated = analysis.trials[-1]
    ineligible = promoter.preview(
        index,
        analysis,
        promotion_name="ineligible-candidate",
        plan_id=dominated.plan_id,
        trial_id=dominated.trial_id,
        policy=ExperimentPromotionPolicy(
            require_pareto_front=True,
            required_artifact_kinds=(ExperimentArtifactKind.TRAINER_BATCH_MANIFEST,),
        ),
    )
    failed_codes = {item.code for item in ineligible.checks if not item.passed}
    assert {"artifacts.required", "ranking.pareto"} <= failed_codes
    with pytest.raises(ValueError, match="ineligible"):
        ExperimentPromotionRecord.model_validate({**record_payload, "preview": ineligible})
    with pytest.raises(ExperimentPromotionError, match="not eligible"):
        promoter.promote(
            index,
            analysis,
            ineligible,
            confirm_preview_id=ineligible.preview_id,
            operator="contract-operator",
            reason="reject an ineligible candidate promotion",
            approvers=("contract-reviewer",),
        )

    issue_index, issue_state = completed_index(tmp_path / "with-issue")
    issue_score = objective(issue_index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    issue_analysis = ExperimentAnalyzer().analyze(
        issue_index,
        ExperimentRankingSpec(objectives=(issue_score,)),
    )
    issue_selected = issue_analysis.trials[0]
    issue_promoter = ExperimentPromoter(
        tmp_path / "with-issue",
        issue_state,
        tmp_path / "issue-promotions",
    )
    strict_preview = issue_promoter.preview(
        issue_index,
        issue_analysis,
        promotion_name="strict-index",
        plan_id=issue_selected.plan_id,
        trial_id=issue_selected.trial_id,
        policy=ExperimentPromotionPolicy(),
    )
    relaxed_preview = issue_promoter.preview(
        issue_index,
        issue_analysis,
        promotion_name="relaxed-index",
        plan_id=issue_selected.plan_id,
        trial_id=issue_selected.trial_id,
        policy=ExperimentPromotionPolicy(require_clean_index=False),
    )
    assert not strict_preview.eligible
    assert relaxed_preview.eligible
    assert strict_preview.preview_id != relaxed_preview.preview_id

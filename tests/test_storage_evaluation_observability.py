import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    CheckpointArtifact,
    CheckpointManifest,
    DataOrigin,
    Message,
    MessageRole,
    Provenance,
    RewardSignal,
    RewardSource,
    RewardSummary,
    RunKind,
    RunLivenessState,
    RunManifest,
    RunStatus,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
)
from agentic_rl_forge.data import PRMDatasetBuilder
from agentic_rl_forge.evaluation import BenchmarkAggregator, BenchmarkComparator
from agentic_rl_forge.rollout import SignalAwareRolloutFilter
from agentic_rl_forge.search import HeuristicProcessRewardModel, ProcessScore
from agentic_rl_forge.services import MetricsRegistry, create_prm_app
from agentic_rl_forge.storage import (
    CheckpointRegistry,
    RunLeaseConflictError,
    SQLiteTrajectoryStore,
    TrajectoryQuery,
)


def make_trajectory(
    trajectory_id: str,
    *,
    reward: float,
    answer: str,
    status: TrajectoryStatus,
    task_id: str = "task-1",
    group_id: str = "group-1",
    policy_version: str = "policy-1",
) -> Trajectory:
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    completed_at = started_at + timedelta(seconds=2)
    steps = (
        TrajectoryStep(
            index=0,
            input_messages=(Message(role=MessageRole.USER, content="Return yes."),),
            action=AgentAction(
                kind=ActionKind.FINAL,
                reasoning=f"answer {answer}",
                final_answer=answer,
            ),
            generated_token_count=2,
            generated_token_mask=(1, 1),
            rewards=RewardSummary(
                signals=(
                    RewardSignal(
                        name="format",
                        source=RewardSource.FORMAT,
                        value=0.1,
                        confidence=0.8,
                    ),
                )
            ),
            started_at=started_at,
            completed_at=completed_at,
        ),
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        group_id=group_id,
        policy_version=policy_version,
        environment_version="env-1",
        provenance=Provenance(
            origin=DataOrigin.ON_POLICY,
            producer="test",
            producer_version="1",
        ),
        status=status,
        steps=steps,
        final_reward=RewardSummary(
            signals=(
                RewardSignal(
                    name="outcome",
                    source=RewardSource.OUTCOME,
                    value=reward,
                    terminal=True,
                    confidence=0.9,
                ),
            )
        ),
        started_at=started_at,
        completed_at=completed_at,
    )


def test_sqlite_store_preserves_runs_content_and_filters(tmp_path: Path) -> None:
    database = tmp_path / "trajectories.db"
    success = make_trajectory(
        "trajectory-1",
        reward=1.0,
        answer="yes",
        status=TrajectoryStatus.SUCCEEDED,
    )
    failure = make_trajectory(
        "trajectory-2",
        reward=0.0,
        answer="no",
        status=TrajectoryStatus.FAILED,
    )
    manifest = RunManifest(name="unit rollout", kind=RunKind.ROLLOUT)

    with SQLiteTrajectoryStore(database) as store:
        assert store.create_run(manifest)
        assert store.put_many((success, failure), run_id=manifest.run_id) == 2
        assert store.put(success) is False
        finished = store.finish_run(manifest.run_id, metadata={"worker_count": 1})

        assert finished.status is RunStatus.COMPLETED
        assert store.get("trajectory-1") == success
        assert store.query(
            TrajectoryQuery(
                run_id=manifest.run_id,
                statuses=(TrajectoryStatus.SUCCEEDED,),
            )
        ) == (success,)
        assert store.summary().trajectory_count == 2
        assert store.summary().task_count == 1

        conflicting = success.model_copy(update={"metadata": {"changed": True}})
        with pytest.raises(ValueError, match="other content"):
            store.put(conflicting)


def test_sqlite_run_leases_fence_stale_owners_and_report_liveness(tmp_path: Path) -> None:
    database = tmp_path / "leased-runs.db"
    started_at = datetime.now(timezone.utc) - timedelta(seconds=20)
    run = RunManifest(name="leased rollout", kind=RunKind.ROLLOUT, started_at=started_at)
    missing_heartbeat_run = RunManifest(
        name="missing heartbeat",
        kind=RunKind.ROLLOUT,
        started_at=started_at,
    )
    released_run = RunManifest(
        name="released lease",
        kind=RunKind.ROLLOUT,
        started_at=started_at,
    )
    trajectory = make_trajectory(
        "leased-trajectory",
        reward=1.0,
        answer="yes",
        status=TrajectoryStatus.SUCCEEDED,
    )

    with SQLiteTrajectoryStore(database) as store:
        store.create_run(run)
        store.create_run(missing_heartbeat_run)
        store.create_run(released_run)
        lease = store.acquire_run_lease(
            run.run_id,
            owner_id="worker-a",
            ttl_s=30,
            now=started_at,
        )
        assert lease.epoch == 1
        assert store.get_run_heartbeat(run.run_id) == lease
        with pytest.raises(RunLeaseConflictError, match="worker-a"):
            store.acquire_run_lease(
                run.run_id,
                owner_id="worker-b",
                ttl_s=30,
                now=started_at + timedelta(seconds=1),
            )
        renewed = store.renew_run_lease(
            run.run_id,
            lease.token,
            ttl_s=30,
            now=started_at + timedelta(seconds=10),
        )
        assert renewed.lease_expires_at == started_at + timedelta(seconds=40)
        active = {
            item.run_id: item
            for item in store.list_run_liveness(
                stale_after_s=60,
                now=started_at + timedelta(seconds=20),
            )
        }
        assert active[run.run_id].state is RunLivenessState.ACTIVE
        assert active[run.run_id].detail == "lease_active"
        assert store.put(trajectory, run_id=run.run_id, lease=lease.token)
        with pytest.raises(RunLeaseConflictError, match="requires a lease token"):
            store.put(trajectory, run_id=run.run_id)

        takeover = store.acquire_run_lease(
            run.run_id,
            owner_id="worker-b",
            ttl_s=30,
            now=started_at + timedelta(seconds=41),
        )
        assert takeover.epoch == 2
        with pytest.raises(RunLeaseConflictError, match="does not match"):
            store.finish_run(
                run.run_id,
                completed_at=started_at + timedelta(seconds=42),
                lease=lease.token,
            )
        finished = store.finish_run(
            run.run_id,
            completed_at=started_at + timedelta(seconds=42),
            lease=takeover.token,
        )
        assert finished.status is RunStatus.COMPLETED
        assert store.get_run_heartbeat(run.run_id).released_at == started_at + timedelta(seconds=42)

        released_lease = store.acquire_run_lease(
            released_run.run_id,
            owner_id="worker-c",
            ttl_s=30,
            now=started_at,
        )
        store.release_run_lease(
            released_run.run_id,
            released_lease.token,
            now=started_at + timedelta(seconds=5),
        )
        liveness = {
            item.run_id: item
            for item in store.list_run_liveness(
                stale_after_s=60,
                now=started_at + timedelta(seconds=120),
            )
        }

    assert liveness[run.run_id].state is RunLivenessState.TERMINAL
    assert liveness[missing_heartbeat_run.run_id].state is RunLivenessState.STALE
    assert liveness[missing_heartbeat_run.run_id].detail == "heartbeat_missing"
    assert liveness[released_run.run_id].state is RunLivenessState.STALE
    assert liveness[released_run.run_id].detail == "lease_released_while_running"


@pytest.mark.parametrize("legacy_version", (1, 2))
def test_sqlite_store_migrates_legacy_metadata_for_run_safety_tables(
    tmp_path: Path,
    legacy_version: int,
) -> None:
    database = tmp_path / f"v{legacy_version}.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(legacy_version),),
    )
    connection.commit()
    connection.close()

    with SQLiteTrajectoryStore(database):
        pass

    connection = sqlite3.connect(database)
    version = connection.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()[0]
    heartbeat_table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'run_heartbeats'"
    ).fetchone()
    reconciliation_table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'run_reconciliations'"
    ).fetchone()
    connection.close()
    assert version == "3"
    assert heartbeat_table == ("run_heartbeats",)
    assert reconciliation_table == ("run_reconciliations",)


def test_benchmark_report_exposes_denominators_and_group_signal() -> None:
    trajectories = (
        make_trajectory(
            "trajectory-1",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
        ),
        make_trajectory(
            "trajectory-2",
            reward=0.0,
            answer="no",
            status=TrajectoryStatus.FAILED,
        ),
    )

    report = BenchmarkAggregator().aggregate(trajectories, benchmark="tiny")

    assert report.metrics["attempt_success_rate"].value == pytest.approx(0.5)
    assert report.metrics["attempt_success_rate"].denominator == 2
    assert report.metrics["group_pass_rate"].value == pytest.approx(1.0)
    assert report.metrics["learning_signal_group_rate"].value == pytest.approx(1.0)
    assert report.group_diagnostics[0].reward_stddev == pytest.approx(0.45)
    assert report.group_diagnostics[0].unique_trajectory_ratio == pytest.approx(1.0)


def test_prm_dataset_is_reproducible_and_preserves_lineage() -> None:
    trajectories = (
        make_trajectory(
            "trajectory-1",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
        ),
        make_trajectory(
            "trajectory-2",
            reward=0.0,
            answer="no",
            status=TrajectoryStatus.FAILED,
        ),
    )
    builder = PRMDatasetBuilder(gamma=0.9, skip_zero_variance_groups=True)

    first = builder.build(trajectories)
    second = builder.build(trajectories)
    summary = builder.summarize(first)

    assert first == second
    assert len(first) == 2
    assert first[0].provenance.origin is DataOrigin.PRM_DATASET
    assert first[0].provenance.parent_ids == (first[0].trajectory_id,)
    assert first[0].metadata["group_advantage"] > 0
    assert first[1].metadata["group_advantage"] < 0
    assert len({example.split for example in first}) == 1
    assert summary.example_count == 2
    assert summary.task_count == 1


@pytest.mark.asyncio
async def test_metrics_registry_and_prm_service_export_prometheus_metrics() -> None:
    registry = MetricsRegistry()
    trajectory = make_trajectory(
        "trajectory-1",
        reward=1.0,
        answer="yes",
        status=TrajectoryStatus.SUCCEEDED,
    )
    registry.record_trajectory(trajectory)
    model = HeuristicProcessRewardModel(
        lambda item: ProcessScore(value=float(item.state["value"])),
        version="heuristic-test",
    )
    transport = httpx.ASGITransport(app=create_prm_app(model, metrics=registry))
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    response = await client.post("/score", json={"inputs": [{"state": {"value": 0.5}}]})
    metrics = await client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["scores"][0]["value"] == pytest.approx(0.5)
    assert metrics.status_code == 200
    assert "arf_trajectories_total" in metrics.text
    assert "arf_prm_examples_total" in metrics.text
    assert "arf_http_requests_total" in metrics.text
    await client.aclose()


def test_store_and_metrics_reject_invalid_state(tmp_path: Path) -> None:
    database = tmp_path / "invalid-state.db"
    manifest = RunManifest(name="failed run", kind=RunKind.ROLLOUT)
    trajectory = make_trajectory(
        "trajectory-1",
        reward=0.0,
        answer="no",
        status=TrajectoryStatus.FAILED,
    )

    with pytest.raises(ValueError, match="positive"):
        TrajectoryQuery(limit=0)
    with pytest.raises(ValueError, match="negative"):
        TrajectoryQuery(offset=-1)

    with SQLiteTrajectoryStore(database) as store:
        assert store.create_run(manifest)
        assert store.create_run(manifest) is False
        store.finish_run(manifest.run_id, status=RunStatus.FAILED)
        assert store.get("missing") is None
        assert store.get_run("missing") is None
        with pytest.raises(ValueError, match="finished run"):
            store.put(trajectory, run_id=manifest.run_id)
        with pytest.raises(KeyError):
            store.finish_run("missing")

    registry = MetricsRegistry()
    registry.set_gauge("arf_workers", 2.0, labels={"pool": "rollout"})
    registry.observe("arf_queue_seconds", 0.1, buckets=(0.1, 1.0))
    assert registry.snapshot().gauges['arf_workers{pool="rollout"}'] == 2.0
    assert "arf_queue_seconds_bucket" in registry.render_prometheus()
    with pytest.raises(ValueError, match="negative"):
        registry.increment("arf_invalid_total", -1.0)
    with pytest.raises(ValueError, match="already declared"):
        registry.increment("arf_workers")


def test_prm_builder_can_drop_groups_without_reward_signal() -> None:
    trajectories = (
        make_trajectory(
            "trajectory-1",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
        ),
        make_trajectory(
            "trajectory-2",
            reward=1.0,
            answer="also yes",
            status=TrajectoryStatus.SUCCEEDED,
        ),
    )

    examples = PRMDatasetBuilder(skip_zero_variance_groups=True).build(trajectories)

    assert examples == ()
    with pytest.raises(ValueError, match="sum to one"):
        PRMDatasetBuilder(train_ratio=0.5, validation_ratio=0.1, test_ratio=0.1)


def test_signal_filter_keeps_informative_on_policy_groups() -> None:
    trajectories = (
        make_trajectory(
            "signal-1",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
            group_id="signal-group",
        ),
        make_trajectory(
            "signal-2",
            reward=0.0,
            answer="no",
            status=TrajectoryStatus.FAILED,
            group_id="signal-group",
        ),
        make_trajectory(
            "flat-1",
            reward=0.0,
            answer="same",
            status=TrajectoryStatus.FAILED,
            task_id="task-2",
            group_id="flat-group",
        ),
        make_trajectory(
            "flat-2",
            reward=0.0,
            answer="same",
            status=TrajectoryStatus.FAILED,
            task_id="task-2",
            group_id="flat-group",
        ),
    )

    result = SignalAwareRolloutFilter(
        expected_group_size=2,
        min_unique_trajectory_ratio=0.5,
    ).filter(trajectories, expected_policy_version="policy-1")

    assert {item.group_id for item in result.accepted} == {"signal-group"}
    assert result.accepted_group_count == 1
    assert result.rejected_groups[0].group_id == "flat-group"
    assert "reward_variance_below_threshold" in result.rejected_groups[0].reasons


def test_checkpoint_registry_detects_artifact_drift(tmp_path: Path) -> None:
    artifact_path = tmp_path / "actor.safetensors"
    artifact_path.write_bytes(b"policy weights")
    artifact = CheckpointArtifact.from_file("actor", artifact_path)
    manifest = CheckpointManifest(
        checkpoint_id="checkpoint-10",
        run_id="run-1",
        step=10,
        policy_version="policy-10",
        config_digest="a" * 64,
        artifacts=(artifact,),
    )
    registry = CheckpointRegistry(tmp_path / "registry")

    assert registry.save(manifest)
    assert registry.save(manifest) is False
    assert registry.get("checkpoint-10") == manifest
    assert registry.latest(run_id="run-1") == manifest
    assert registry.verify(manifest).fully_verified

    artifact_path.write_bytes(b"changed weights")
    verification = registry.verify(manifest)

    assert not verification.valid
    assert verification.mismatched_artifacts == ("actor",)


def test_benchmark_comparison_uses_matched_task_bootstrap() -> None:
    baseline = (
        make_trajectory(
            "baseline-1",
            reward=0.0,
            answer="no",
            status=TrajectoryStatus.FAILED,
            task_id="task-1",
            group_id="baseline-1",
            policy_version="baseline",
        ),
        make_trajectory(
            "baseline-2",
            reward=0.0,
            answer="no",
            status=TrajectoryStatus.FAILED,
            task_id="task-2",
            group_id="baseline-2",
            policy_version="baseline",
        ),
    )
    candidate = (
        make_trajectory(
            "candidate-1",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
            task_id="task-1",
            group_id="candidate-1",
            policy_version="candidate",
        ),
        make_trajectory(
            "candidate-2",
            reward=0.0,
            answer="no",
            status=TrajectoryStatus.FAILED,
            task_id="task-2",
            group_id="candidate-2",
            policy_version="candidate",
        ),
    )

    report = BenchmarkComparator(bootstrap_samples=200, seed=7).compare(
        baseline,
        candidate,
        benchmark="paired",
        baseline_name="baseline",
        candidate_name="candidate",
    )

    pass_delta = report.metrics["task_pass_rate"]
    assert report.matched_task_count == 2
    assert pass_delta.absolute_delta == pytest.approx(0.5)
    assert pass_delta.confidence_low <= pass_delta.absolute_delta
    assert pass_delta.confidence_high >= pass_delta.absolute_delta


def test_new_training_cli_workflows(tmp_path: Path) -> None:
    database = tmp_path / "training.db"
    filtered = tmp_path / "filtered.jsonl"
    filter_report = tmp_path / "filter-report.json"
    comparison = tmp_path / "comparison.json"
    config = tmp_path / "train.yaml"
    actor = tmp_path / "actor.safetensors"
    registry = tmp_path / "checkpoints"
    config.write_text("trainer:\n  seed: 7\n", encoding="utf-8")
    actor.write_bytes(b"weights")
    trajectories = (
        make_trajectory(
            "base-1",
            reward=0.0,
            answer="no",
            status=TrajectoryStatus.FAILED,
            group_id="base-group",
            policy_version="baseline",
        ),
        make_trajectory(
            "base-2",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
            group_id="base-group",
            policy_version="baseline",
        ),
        make_trajectory(
            "candidate-1",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
            group_id="candidate-group",
            policy_version="candidate",
        ),
        make_trajectory(
            "candidate-2",
            reward=1.0,
            answer="yes",
            status=TrajectoryStatus.SUCCEEDED,
            group_id="candidate-group",
            policy_version="candidate",
        ),
    )
    with SQLiteTrajectoryStore(database) as store:
        store.put_many(trajectories)
    runner = CliRunner()

    filtered_result = runner.invoke(
        app,
        [
            "filter-rollouts",
            str(database),
            str(filtered),
            "--policy-version",
            "baseline",
            "--expected-group-size",
            "2",
            "--report",
            str(filter_report),
        ],
    )
    compared = runner.invoke(
        app,
        [
            "compare",
            str(database),
            "--baseline-policy",
            "baseline",
            "--candidate-policy",
            "candidate",
            "--bootstrap-samples",
            "100",
            "--output",
            str(comparison),
        ],
    )
    registered = runner.invoke(
        app,
        [
            "checkpoint-register",
            str(registry),
            "--run-id",
            "run-1",
            "--step",
            "1",
            "--policy-version",
            "candidate",
            "--config",
            str(config),
            "--artifact",
            f"actor={actor}",
        ],
    )
    listed = runner.invoke(app, ["checkpoint-list", str(registry)])

    assert filtered_result.exit_code == 0
    assert len(filtered.read_text(encoding="utf-8").splitlines()) == 2
    assert filter_report.exists()
    assert compared.exit_code == 0
    assert comparison.exists()
    assert registered.exit_code == 0
    assert '"fully_verified": true' in registered.stdout
    assert listed.exit_code == 0
    assert '"policy_version": "candidate"' in listed.stdout


def test_cli_persistent_workflow(tmp_path: Path) -> None:
    runner = CliRunner()
    source = tmp_path / "demo.jsonl"
    database = tmp_path / "trajectories.db"
    report = tmp_path / "report.json"
    prm = tmp_path / "prm.jsonl"

    demo = runner.invoke(app, ["demo", "--output", str(source)])
    imported = runner.invoke(
        app,
        ["trajectory-import", str(source), str(database), "--run-name", "demo-run"],
    )
    summary = runner.invoke(app, ["trajectory-summary", str(database)])
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            str(database),
            "--benchmark",
            "local-demo",
            "--output",
            str(report),
        ],
    )
    built = runner.invoke(app, ["build-prm-dataset", str(database), str(prm)])

    assert demo.exit_code == 0
    assert imported.exit_code == 0
    assert summary.exit_code == 0
    assert '"trajectory_count": 1' in summary.stdout
    assert evaluated.exit_code == 0
    assert built.exit_code == 0
    assert report.exists()
    assert len(prm.read_text(encoding="utf-8").splitlines()) == 2

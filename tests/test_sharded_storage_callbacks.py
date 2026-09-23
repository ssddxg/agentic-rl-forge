from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    DataOrigin,
    Message,
    MessageRole,
    Provenance,
    RewardSignal,
    RewardSource,
    RewardSummary,
    RunKind,
    RunManifest,
    TaskSpec,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    VerifierSpec,
)
from agentic_rl_forge.environments import LocalToolEnvironment
from agentic_rl_forge.rewards import ExactMatchOutcome, RewardEngine
from agentic_rl_forge.rollout import (
    AgentLoop,
    MetricsRolloutCallback,
    PolicyOutput,
    RolloutScheduler,
    ScriptedPolicy,
    ShardedRolloutCallback,
    SQLiteRolloutCallback,
)
from agentic_rl_forge.services import MetricsRegistry
from agentic_rl_forge.storage import (
    ShardedTrajectoryStore,
    SQLiteTrajectoryStore,
    export_trajectories_jsonl,
)


def make_trajectory(
    trajectory_id: str,
    *,
    task_id: str = "task-1",
    group_id: str = "group-1",
    answer: str = "yes",
    reward: float = 1.0,
) -> Trajectory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        group_id=group_id,
        policy_version="policy-1",
        environment_version="env-1",
        provenance=Provenance(
            origin=DataOrigin.ON_POLICY,
            producer="test",
            producer_version="1",
        ),
        status=(TrajectoryStatus.SUCCEEDED if reward > 0 else TrajectoryStatus.FAILED),
        steps=(
            TrajectoryStep(
                index=0,
                input_messages=(Message(role=MessageRole.USER, content="Return yes."),),
                action=AgentAction(kind=ActionKind.FINAL, final_answer=answer),
                generated_token_count=1,
                generated_token_mask=(1,),
                started_at=now,
                completed_at=now,
            ),
        ),
        final_reward=RewardSummary(
            signals=(
                RewardSignal(
                    name="outcome",
                    source=RewardSource.OUTCOME,
                    value=reward,
                    terminal=True,
                ),
            )
        ),
        started_at=now,
        completed_at=now,
    )


def make_task(task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        messages=(Message(role=MessageRole.USER, content="Return yes."),),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "yes"}),
    )


def test_sharded_store_finalizes_verifies_exports_and_recovers(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    store = ShardedTrajectoryStore(root, run_id="run-1")
    first = make_trajectory("trajectory-1")
    second = make_trajectory(
        "trajectory/with/path-characters",
        task_id="task-2",
        group_id="group-2",
    )

    assert store.put(first)
    assert store.put(second)
    assert store.put(first) is False
    with pytest.raises(ValueError, match="other content"):
        store.put(first.model_copy(update={"metadata": {"changed": True}}))
    manifest = store.finalize(expected_policy_version="policy-1")
    annotated_manifest = store.finalize(metadata={"phase": "evaluation"})
    exported = tmp_path / "trajectories.jsonl"

    assert manifest.trajectory_count == 2
    assert annotated_manifest.manifest_id != manifest.manifest_id
    assert store.verify(manifest).complete
    assert store.export_jsonl(manifest, exported) == 2
    assert len(exported.read_text(encoding="utf-8").splitlines()) == 2

    recovered = ShardedTrajectoryStore(root, run_id="run-1")
    assert recovered.get(first.trajectory_id) == first
    assert recovered.get_manifest(manifest.manifest_id) == manifest
    assert {item.manifest_id for item in recovered.list_manifests()} == {
        manifest.manifest_id,
        annotated_manifest.manifest_id,
    }
    assert recovered.latest_manifest() is not None
    assert recovered.finalize(expected_policy_version="policy-1") == manifest

    third = make_trajectory(
        "trajectory-3",
        task_id="task-3",
        group_id="group-3",
    )
    recovered.put(third)
    incomplete = recovered.verify(manifest)
    assert incomplete.valid
    assert not incomplete.complete
    assert incomplete.unexpected_paths

    shard_path = recovered.run_path / manifest.shards[0].relative_path
    shard_path.write_bytes(b"{}\n")
    verification = recovered.verify(manifest)
    assert not verification.valid
    assert manifest.shards[0].trajectory_id in verification.mismatched_trajectory_ids


@pytest.mark.asyncio
async def test_scheduler_streams_to_sqlite_shards_and_metrics(tmp_path: Path) -> None:
    sqlite_store = SQLiteTrajectoryStore(tmp_path / "trajectories.db")
    run = RunManifest(name="callback-run", kind=RunKind.ROLLOUT)
    sqlite_store.create_run(run)
    shard_store = ShardedTrajectoryStore(tmp_path / "shards", run_id=run.run_id)
    registry = MetricsRegistry()
    shard_callback = ShardedRolloutCallback(shard_store)

    def loop_factory() -> AgentLoop:
        return AgentLoop(
            policy=ScriptedPolicy(
                (
                    PolicyOutput(
                        action=AgentAction(
                            kind=ActionKind.FINAL,
                            final_answer="yes",
                            raw_text="<answer>yes</answer>",
                        ),
                        generated_token_count=2,
                    ),
                ),
                version="callback-policy",
            ),
            environment=LocalToolEnvironment(()),
            rewards=RewardEngine((ExactMatchOutcome(),)),
        )

    scheduler = RolloutScheduler(
        loop_factory,
        max_concurrency=4,
        callbacks=(
            SQLiteRolloutCallback(sqlite_store, run_id=run.run_id),
            shard_callback,
            MetricsRolloutCallback(registry),
        ),
    )
    batch = await scheduler.collect(
        (make_task("task-1"), make_task("task-2")),
        rollouts_per_task=2,
        seed=4,
    )

    assert len(batch.trajectories) == 4
    assert sqlite_store.count() == 4
    assert shard_callback.manifest is not None
    assert shard_callback.manifest.trajectory_count == 4
    assert shard_store.verify(shard_callback.manifest).complete
    snapshot = registry.snapshot()
    assert snapshot.counters['arf_trajectories_total{origin="on_policy",status="succeeded"}'] == 4.0
    assert snapshot.counters['arf_rollout_batches_total{policy_version="callback-policy"}'] == 1.0
    sqlite_store.finish_run(run.run_id)
    sqlite_store.close()


def test_shard_cli_can_import_verify_and_export(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    root = tmp_path / "archive"
    output = tmp_path / "output.jsonl"
    export_trajectories_jsonl(
        (
            make_trajectory("trajectory-1"),
            make_trajectory("trajectory-2", task_id="task-2", group_id="group-2"),
        ),
        source,
    )
    runner = CliRunner()

    imported = runner.invoke(
        app,
        ["shard-import", str(source), str(root), "--run-id", "run-1"],
    )
    manifest = ShardedTrajectoryStore(root, run_id="run-1").latest_manifest()
    assert manifest is not None
    verified = runner.invoke(
        app,
        [
            "shard-verify",
            str(root),
            "--run-id",
            "run-1",
            "--manifest-id",
            manifest.manifest_id,
        ],
    )
    exported = runner.invoke(
        app,
        [
            "shard-export",
            str(root),
            str(output),
            "--run-id",
            "run-1",
            "--manifest-id",
            manifest.manifest_id,
        ],
    )

    assert imported.exit_code == 0
    assert '"complete": true' in imported.stdout
    assert verified.exit_code == 0
    assert '"valid": true' in verified.stdout
    assert exported.exit_code == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

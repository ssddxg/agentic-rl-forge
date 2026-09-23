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
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
)
from agentic_rl_forge.integrations import TrainerBatchExporter
from agentic_rl_forge.storage import LocalBlobStore, export_trajectories_jsonl


def make_trajectory(
    trajectory_id: str,
    *,
    answer: str,
    reward: float,
    origin: DataOrigin = DataOrigin.ON_POLICY,
) -> Trajectory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    parent_ids = ("parent",) if origin is not DataOrigin.ON_POLICY else ()
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id="task-1",
        group_id="group-1",
        policy_version="policy-1",
        environment_version="env-1",
        provenance=Provenance(
            origin=origin,
            producer="test",
            producer_version="1",
            parent_ids=parent_ids,
            transform="test_transform" if parent_ids else None,
        ),
        status=(TrajectoryStatus.SUCCEEDED if reward > 0 else TrajectoryStatus.FAILED),
        steps=(
            TrajectoryStep(
                index=0,
                input_messages=(Message(role=MessageRole.USER, content="Return yes."),),
                action=AgentAction(kind=ActionKind.FINAL, final_answer=answer),
                generated_token_count=1,
                generated_token_mask=(1,),
                policy_logprobs=(-0.1,),
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


def test_trainer_batch_export_is_idempotent_and_verifiable(tmp_path: Path) -> None:
    trajectories = (
        make_trajectory("trajectory-1", answer="yes", reward=1.0),
        make_trajectory("trajectory-2", answer="no", reward=0.0),
    )
    blobs = LocalBlobStore(tmp_path / "blobs")
    exporter = TrainerBatchExporter(blobs)

    manifest = exporter.export(
        trajectories,
        expected_policy_version="policy-1",
        expected_group_size=2,
        source_run_id="run-1",
    )
    repeated = exporter.export(
        tuple(reversed(trajectories)),
        expected_policy_version="policy-1",
        expected_group_size=2,
        source_run_id="run-1",
    )

    assert repeated == manifest
    assert exporter.verify(manifest).trajectory_contents_match
    records = exporter.load_records(manifest)
    assert tuple(record.trajectory_id for record in records) == (
        "trajectory-1",
        "trajectory-2",
    )
    assert all(record.policy_version == "policy-1" for record in records)

    payload_path = blobs.root / manifest.payload_key
    payload_path.write_bytes(b"corrupt\n")
    verification = exporter.verify(manifest)
    assert not verification.valid
    assert not verification.payload_digest_matches


def test_trainer_batch_rejects_derived_or_incomplete_groups(tmp_path: Path) -> None:
    exporter = TrainerBatchExporter(LocalBlobStore(tmp_path / "blobs"))
    on_policy = make_trajectory("trajectory-1", answer="yes", reward=1.0)
    derived = make_trajectory(
        "trajectory-2",
        answer="no",
        reward=0.0,
        origin=DataOrigin.REPLAY,
    )

    with pytest.raises(ValueError, match="not on-policy"):
        exporter.export(
            (on_policy, derived),
            expected_policy_version="policy-1",
            expected_group_size=2,
        )
    with pytest.raises(ValueError, match="has 1 trajectories"):
        exporter.export(
            (on_policy,),
            expected_policy_version="policy-1",
            expected_group_size=2,
        )


def test_trainer_batch_cli_exports_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "filtered.jsonl"
    root = tmp_path / "trainer-store"
    trajectories = (
        make_trajectory("trajectory-1", answer="yes", reward=1.0),
        make_trajectory("trajectory-2", answer="no", reward=0.0),
    )
    export_trajectories_jsonl(trajectories, source)
    runner = CliRunner()

    exported = runner.invoke(
        app,
        [
            "trainer-batch-export",
            str(source),
            str(root),
            "--policy-version",
            "policy-1",
            "--group-size",
            "2",
            "--source-run-id",
            "run-1",
        ],
    )
    manifests = LocalBlobStore(root).list("trainer/batches")
    manifest_key = next(key for key in manifests if key.endswith("manifest.json"))
    verified = runner.invoke(
        app,
        [
            "trainer-batch-verify",
            str(root),
            "--manifest-key",
            manifest_key,
        ],
    )

    assert exported.exit_code == 0
    assert '"valid": true' in exported.stdout
    assert verified.exit_code == 0
    assert '"trajectory_ids_match": true' in verified.stdout

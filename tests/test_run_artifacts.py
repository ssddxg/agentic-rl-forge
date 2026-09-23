from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
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
    RunArtifactManifest,
    RunKind,
    RunManifest,
    RunStatus,
    TaskSpec,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    VerifierSpec,
)
from agentic_rl_forge.evaluation import BenchmarkAggregator
from agentic_rl_forge.rollout import RolloutPlanBuilder
from agentic_rl_forge.storage import (
    RunArtifactArchive,
    RunArtifactArchiveError,
    RunArtifactBundle,
    ShardedTrajectoryStore,
)


def read_archive_members(archive: Path) -> list[tuple[tarfile.TarInfo, bytes]]:
    members = []
    with tarfile.open(archive, mode="r:gz") as source:
        for member in source.getmembers():
            extracted = source.extractfile(member) if member.isreg() else None
            members.append((member, extracted.read() if extracted is not None else b""))
    return members


def write_test_archive(
    archive: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
) -> None:
    with (
        archive.open("wb") as raw_output,
        gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_output,
            mtime=0,
        ) as compressed,
        tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as output,
    ):
        for source_info, payload in members:
            info = tarfile.TarInfo(source_info.name)
            info.mode = source_info.mode
            info.uid = source_info.uid
            info.gid = source_info.gid
            info.uname = source_info.uname
            info.gname = source_info.gname
            info.mtime = source_info.mtime
            info.type = source_info.type
            info.linkname = source_info.linkname
            info.size = len(payload) if info.isreg() else 0
            output.addfile(info, io.BytesIO(payload) if info.isreg() else None)


def build_bundle(
    root: Path,
    *,
    summary_run_id: str | None = None,
) -> tuple[RunArtifactBundle, RunArtifactManifest, ShardedTrajectoryStore]:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    task = TaskSpec(
        task_id="artifact-task",
        messages=(Message(role=MessageRole.USER, content="Return yes."),),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "yes"}),
    )
    plan = RolloutPlanBuilder().build(
        (task,),
        policy_version="artifact-policy-v1",
        source_sha256="0" * 64,
        config_digest="1" * 64,
        rollouts_per_task=1,
        seed=3,
    )
    slot = plan.slots[0]
    trajectory = Trajectory(
        trajectory_id=slot.trajectory_id,
        task_id=task.task_id,
        group_id=slot.group_id,
        policy_version=plan.policy_version,
        environment_version="artifact-env-v1",
        provenance=Provenance(
            origin=DataOrigin.ON_POLICY,
            producer="artifact-test",
            producer_version="1",
            metadata={
                "rollout_plan_id": plan.plan_id,
                "rollout_slot_id": slot.slot_id,
                "rollout_index": slot.rollout_index,
                "rollout_seed": slot.seed,
            },
        ),
        status=TrajectoryStatus.SUCCEEDED,
        steps=(
            TrajectoryStep(
                index=0,
                input_messages=task.messages,
                action=AgentAction(kind=ActionKind.FINAL, final_answer="yes"),
                generated_token_count=1,
                generated_token_mask=(1,),
                started_at=created_at,
                completed_at=created_at,
            ),
        ),
        final_reward=RewardSummary(
            signals=(
                RewardSignal(
                    name="outcome",
                    source=RewardSource.OUTCOME,
                    value=1.0,
                    terminal=True,
                ),
            )
        ),
        started_at=created_at,
        completed_at=created_at,
    )
    run = RunManifest(
        name="artifact run",
        kind=RunKind.ROLLOUT,
        policy_version=plan.policy_version,
        environment_version=trajectory.environment_version,
        started_at=created_at,
    )
    shard_store = ShardedTrajectoryStore(root / "shards", run_id=run.run_id)
    shard_store.put(trajectory)
    shard_manifest = shard_store.finalize(expected_policy_version=plan.policy_version)
    finished = RunManifest.model_validate(
        {
            **run.model_dump(mode="python"),
            "status": RunStatus.COMPLETED,
            "completed_at": created_at,
            "metadata": {
                "trajectory_count": 1,
                "shard_manifest_id": shard_manifest.manifest_id,
            },
        }
    )
    report = BenchmarkAggregator().aggregate(
        (trajectory,),
        benchmark="artifact-benchmark",
        run_id=run.run_id,
    )
    run_path = root / "runs" / run.run_id
    plan_path = root / "plans" / f"{plan.plan_id}.json"
    run_path.mkdir(parents=True, exist_ok=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(plan.canonical_bytes() + b"\n")
    (run_path / "run-manifest.json").write_bytes(finished.canonical_bytes() + b"\n")
    (run_path / "trajectories.jsonl").write_bytes(trajectory.canonical_bytes() + b"\n")
    (run_path / "benchmark-report.json").write_bytes(report.canonical_bytes() + b"\n")
    (run_path / "metrics.prom").write_text("arf_trajectories_total 1\n", encoding="utf-8")
    (run_path / "summary.json").write_text(
        json.dumps(
            {
                "run_id": summary_run_id or run.run_id,
                "plan_id": plan.plan_id,
                "shard_manifest_id": shard_manifest.manifest_id,
                "trajectory_count": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    bundle = RunArtifactBundle(root)
    manifest = bundle.build(
        run_id=run.run_id,
        plan_id=plan.plan_id,
        shard_manifest_id=shard_manifest.manifest_id,
        created_at=created_at,
    )
    return bundle, manifest, shard_store


def test_run_artifact_bundle_builds_loads_and_verifies(tmp_path: Path) -> None:
    bundle, manifest, _ = build_bundle(tmp_path)

    verification = bundle.verify(manifest)

    assert bundle.load(tmp_path / bundle.manifest_relative_path(manifest.run_id)) == manifest
    assert verification.valid
    assert verification.verified_artifact_count == 7
    assert verification.shard_verification is not None
    assert verification.shard_verification.complete


def test_run_artifact_verifier_reports_file_and_semantic_failures(tmp_path: Path) -> None:
    missing_bundle, missing_manifest, _ = build_bundle(tmp_path / "missing")
    missing_metrics = (
        tmp_path
        / "missing"
        / next(item.relative_path for item in missing_manifest.artifacts if item.name == "metrics")
    )
    missing_metrics.unlink()
    missing = missing_bundle.verify(missing_manifest)
    assert not missing.valid
    assert missing.missing_artifacts == ("metrics",)

    changed_bundle, changed_manifest, _ = build_bundle(tmp_path / "changed")
    changed_metrics = (
        tmp_path
        / "changed"
        / next(item.relative_path for item in changed_manifest.artifacts if item.name == "metrics")
    )
    changed_metrics.write_text("changed\n", encoding="utf-8")
    changed = changed_bundle.verify(changed_manifest)
    assert not changed.valid
    assert changed.mismatched_artifacts == ("metrics",)

    semantic_bundle, semantic_manifest, _ = build_bundle(
        tmp_path / "semantic",
        summary_run_id="run_wrong",
    )
    semantic = semantic_bundle.verify(semantic_manifest)
    assert not semantic.valid
    assert "summary_run_id_mismatch" in semantic.semantic_errors


def test_run_artifact_verifier_recursively_checks_shard_completeness(tmp_path: Path) -> None:
    damaged_bundle, damaged_manifest, damaged_store = build_bundle(tmp_path / "damaged")
    shard_manifest = damaged_store.get_manifest(damaged_manifest.shard_manifest_id)
    assert shard_manifest is not None
    damaged_path = damaged_store.run_path / shard_manifest.shards[0].relative_path
    damaged_path.write_bytes(b"{}\n")
    damaged = damaged_bundle.verify(damaged_manifest)
    assert not damaged.valid
    assert damaged.shard_verification is not None
    assert damaged.shard_verification.mismatched_trajectory_ids == (
        shard_manifest.shards[0].trajectory_id,
    )

    extra_bundle, extra_manifest, extra_store = build_bundle(tmp_path / "extra")
    (extra_store.shard_path / "unexpected.json").write_text("{}\n", encoding="utf-8")
    extra = extra_bundle.verify(extra_manifest)
    assert not extra.valid
    assert extra.shard_verification is not None
    assert extra.shard_verification.valid
    assert not extra.shard_verification.complete
    assert extra.shard_verification.unexpected_paths == ("shards/unexpected.json",)

    absent_bundle, absent_manifest, absent_store = build_bundle(tmp_path / "absent")
    absent_shard_manifest = absent_store.get_manifest(absent_manifest.shard_manifest_id)
    assert absent_shard_manifest is not None
    absent_path = absent_store.run_path / absent_shard_manifest.shards[0].relative_path
    absent_path.unlink()
    absent = absent_bundle.verify(absent_manifest)
    assert not absent.valid
    assert absent.shard_verification is not None
    assert absent.shard_verification.missing_trajectory_ids == (
        absent_shard_manifest.shards[0].trajectory_id,
    )


def test_run_artifact_archive_is_deterministic_idempotent_and_portable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    bundle, manifest, _ = build_bundle(root)
    first_path = tmp_path / "first.tar.gz"
    second_path = tmp_path / "second.tar.gz"

    first = RunArtifactArchive(root).pack(manifest, first_path)
    second = RunArtifactArchive(root).pack(manifest, second_path)
    retried = RunArtifactArchive(root).pack(manifest, first_path)

    assert first == second == retried
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.member_count == 9
    assert (
        RunArtifactArchive.inspect(
            first_path,
            expected_sha256=first.content_digest,
        )
        == first
    )
    assert (
        RunArtifactArchive.read_checksum(RunArtifactArchive.checksum_path(first_path))
        == first.content_digest
    )
    assert RunArtifactArchive.checksum_path(first_path).read_text(encoding="ascii") == (
        f"{first.content_digest}\n"
    )
    if os.name != "nt":
        assert first_path.stat().st_mode & 0o777 == 0o644
        assert RunArtifactArchive.checksum_path(first_path).stat().st_mode & 0o777 == 0o644

    destination = tmp_path / "portable"
    unpacked = RunArtifactArchive.unpack(
        first_path,
        destination,
        expected_sha256=first.content_digest,
    )
    portable_manifest_path = destination / bundle.manifest_relative_path(manifest.run_id)
    portable_bundle = RunArtifactBundle(destination)
    portable_manifest = portable_bundle.load(portable_manifest_path)

    assert unpacked == first
    assert portable_manifest == manifest
    assert portable_bundle.verify(portable_manifest).valid
    if os.name != "nt":
        assert destination.stat().st_mode & 0o777 == 0o755
    assert not any(path.is_symlink() for path in destination.rglob("*"))


def test_run_artifact_archive_cli_packs_and_safely_unpacks(tmp_path: Path) -> None:
    root = tmp_path / "source"
    bundle, manifest, _ = build_bundle(root)
    manifest_path = root / bundle.manifest_relative_path(manifest.run_id)
    archive = tmp_path / "published.tar.gz"
    destination = tmp_path / "received"
    runner = CliRunner()

    packed = runner.invoke(app, ["run-artifacts-pack", str(manifest_path), str(archive)])
    unpacked = runner.invoke(app, ["run-artifacts-unpack", str(archive), str(destination)])
    verified = runner.invoke(
        app,
        [
            "run-artifacts-verify",
            str(destination / bundle.manifest_relative_path(manifest.run_id)),
        ],
    )

    assert packed.exit_code == 0
    assert unpacked.exit_code == 0
    assert verified.exit_code == 0
    assert '"archive_id"' in packed.stdout
    assert '"destination"' in unpacked.stdout
    assert '"valid": true' in verified.stdout

    RunArtifactArchive.checksum_path(archive).unlink()
    missing_checksum = runner.invoke(
        app,
        ["run-artifacts-unpack", str(archive), str(tmp_path / "without-checksum")],
    )
    assert missing_checksum.exit_code == 2
    assert "cannot read archive checksum" in missing_checksum.stderr


def test_run_artifact_archive_rejects_transport_and_source_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "source"
    bundle, manifest, _ = build_bundle(root)
    archive = tmp_path / "run.tar.gz"
    receipt = RunArtifactArchive(root).pack(manifest, archive)

    with pytest.raises(RunArtifactArchiveError, match="SHA-256 does not match"):
        RunArtifactArchive.unpack(
            archive,
            tmp_path / "checksum-mismatch",
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "checksum-mismatch").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RunArtifactArchiveError, match="destination already exists"):
        RunArtifactArchive.unpack(
            archive,
            existing,
            expected_sha256=receipt.content_digest,
        )

    RunArtifactArchive.checksum_path(archive).write_text("invalid\n", encoding="ascii")
    with pytest.raises(RunArtifactArchiveError, match="checksum file is not canonical"):
        RunArtifactArchive.read_checksum(RunArtifactArchive.checksum_path(archive))

    conflicting_output = tmp_path / "conflict.tar.gz"
    conflicting_output.write_bytes(b"existing")
    with pytest.raises(RunArtifactArchiveError, match="different content"):
        RunArtifactArchive(root).pack(manifest, conflicting_output)

    manifest_path = root / bundle.manifest_relative_path(manifest.run_id)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(RunArtifactArchiveError, match="not canonical"):
        RunArtifactArchive(root).pack(manifest, tmp_path / "noncanonical.tar.gz")


def test_run_artifact_archive_rejects_unsafe_or_noncanonical_members(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _, manifest, _ = build_bundle(root)
    canonical = tmp_path / "canonical.tar.gz"
    RunArtifactArchive(root).pack(manifest, canonical)
    original = read_archive_members(canonical)

    unsafe_archives: dict[str, list[tuple[tarfile.TarInfo, bytes]]] = {}

    traversal = list(original)
    traversal_info = tarfile.TarInfo("../artifact-manifest.json")
    traversal_info.mode = 0o644
    traversal_info.mtime = 0
    traversal[0] = (traversal_info, traversal[0][1])
    unsafe_archives["traversal"] = traversal

    noncanonical = list(original)
    noncanonical_info, noncanonical_payload = noncanonical[0]
    changed_metadata = tarfile.TarInfo(noncanonical_info.name)
    changed_metadata.mode = 0o644
    changed_metadata.mtime = 1
    noncanonical[0] = (changed_metadata, noncanonical_payload)
    unsafe_archives["metadata"] = noncanonical

    reordered = list(original)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    unsafe_archives["order"] = reordered

    duplicate = list(original)
    duplicate.append(original[-1])
    unsafe_archives["duplicate"] = duplicate

    corrupted = list(original)
    metrics_index = next(
        index for index, (info, _) in enumerate(corrupted) if info.name.endswith("metrics.prom")
    )
    metrics_info, metrics_payload = corrupted[metrics_index]
    corrupted[metrics_index] = (metrics_info, b"X" + metrics_payload[1:])
    unsafe_archives["corrupted"] = corrupted

    for link_type, label in ((tarfile.SYMTYPE, "symlink"), (tarfile.LNKTYPE, "hardlink")):
        linked = list(original)
        linked_info, _ = linked[1]
        link = tarfile.TarInfo(linked_info.name)
        link.mode = 0o644
        link.mtime = 0
        link.type = link_type
        link.linkname = original[0][0].name
        linked[1] = (link, b"")
        unsafe_archives[label] = linked

    for label, members in unsafe_archives.items():
        archive = tmp_path / f"{label}.tar.gz"
        destination = tmp_path / f"{label}-destination"
        write_test_archive(archive, members)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        with pytest.raises(RunArtifactArchiveError):
            RunArtifactArchive.unpack(
                archive,
                destination,
                expected_sha256=digest,
            )
        assert not destination.exists()
        assert not tuple(tmp_path.glob(".arf-unpack-*"))

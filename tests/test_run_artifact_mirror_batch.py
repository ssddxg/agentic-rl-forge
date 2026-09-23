from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    RunArtifactMirrorBatchIntent,
    RunArtifactMirrorBatchMemberState,
    RunArtifactMirrorBatchMemberStatus,
    RunArtifactMirrorBatchPlan,
    RunArtifactMirrorBatchRecord,
    RunArtifactMirrorBatchStatus,
    RunArtifactMirrorRecord,
    RunArtifactMirrorSelection,
    RunArtifactStoreInventory,
)
from agentic_rl_forge.data import Ed25519ManifestSigner, RunArtifactArchiveAttestor
from agentic_rl_forge.storage import (
    LocalBlobStore,
    RunArtifactMirror,
    RunArtifactMirrorError,
    RunArtifactTransport,
    S3ConditionalBlobStore,
)
from test_blob_stores import FakeS3Client
from test_run_artifact_mirror import InstrumentedMirrorStore
from test_run_artifact_transport import build_archive


def publish_releases(
    tmp_path: Path,
    names: tuple[str, ...],
    *,
    signed: bool = False,
) -> tuple[
    tuple[Path, ...],
    tuple[Any, ...],
    LocalBlobStore,
    Ed25519ManifestSigner | None,
]:
    source = LocalBlobStore(tmp_path / "batch-source")
    signer = Ed25519ManifestSigner.generate() if signed else None
    archives = []
    receipts = []
    for index, name in enumerate(names):
        archive, receipt = build_archive(tmp_path, name)
        attestation = (
            RunArtifactArchiveAttestor(signer).sign(
                receipt,
                signed_at=datetime(2026, 9, 18, 18, index, tzinfo=timezone.utc),
            )
            if signer is not None
            else None
        )
        RunArtifactTransport(source, chunk_size_bytes=128).publish(
            archive,
            expected_sha256=receipt.content_digest,
            attestation=attestation,
        )
        archives.append(archive)
        receipts.append(receipt)
    return tuple(archives), tuple(receipts), source, signer


def test_batch_plan_freezes_authenticated_inventories_and_selection(tmp_path: Path) -> None:
    _, receipts, source, signer = publish_releases(
        tmp_path,
        ("batch-plan-one", "batch-plan-two"),
        signed=True,
    )
    assert signer is not None
    destination = LocalBlobStore(tmp_path / "batch-plan-destination")
    mirror = RunArtifactMirror(source, destination)
    observed_at = datetime(2026, 9, 18, 19, 0, tzinfo=timezone.utc)

    plan = mirror.plan_batch(
        include_all_committed=True,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
        max_workers=2,
        now=observed_at,
    )
    repeated = mirror.plan_batch(
        include_all_committed=True,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
        max_workers=1,
        now=observed_at + timedelta(minutes=1),
    )
    explicit = mirror.plan_batch(
        archive_ids=(receipts[1].archive_id,),
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
        now=observed_at,
    )

    assert plan.selection is RunArtifactMirrorSelection.ALL_COMMITTED
    assert plan.selected_archive_ids == tuple(sorted(item.archive_id for item in receipts))
    assert plan.release_count == 2
    assert plan.copy_object_count == sum(item.copy_object_count for item in plan.releases)
    assert plan.copy_bytes == sum(item.copy_bytes for item in plan.releases)
    assert plan.plan_id == repeated.plan_id
    assert plan.source_inventory.state_digest == repeated.source_inventory.state_digest
    assert explicit.selection is RunArtifactMirrorSelection.EXPLICIT
    assert explicit.selected_archive_ids == (receipts[1].archive_id,)

    with pytest.raises(ValueError, match="either archive IDs"):
        mirror.plan_batch()
    with pytest.raises(ValueError, match="either archive IDs"):
        mirror.plan_batch(
            archive_ids=(receipts[0].archive_id,),
            include_all_committed=True,
        )
    with pytest.raises(ValueError, match="must be unique"):
        mirror.plan_batch(
            archive_ids=(receipts[0].archive_id, receipts[0].archive_id),
        )


def test_batch_execution_is_bounded_evidenced_and_idempotent(tmp_path: Path) -> None:
    archives, receipts, source, _ = publish_releases(
        tmp_path,
        ("batch-run-one", "batch-run-two", "batch-run-three"),
    )
    backing = LocalBlobStore(tmp_path / "batch-run-destination")
    instrumented = InstrumentedMirrorStore(backing, delay_seconds=0.002)
    mirror = RunArtifactMirror(source, instrumented)
    action_time = datetime(2026, 9, 18, 20, 0, tzinfo=timezone.utc)
    plan = mirror.plan_batch(include_all_committed=True, max_workers=2, now=action_time)

    record = mirror.execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-mirror-operator",
        reason="replicate reviewed release inventory",
        release_workers=2,
        object_workers=1,
        now=action_time,
    )
    repeated = RunArtifactMirror(source, backing).execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-mirror-operator",
        reason="replicate reviewed release inventory",
        release_workers=1,
        object_workers=2,
        now=action_time + timedelta(minutes=1),
    )
    status = RunArtifactMirror(source, backing).batch_status(
        plan,
        operator="batch-mirror-operator",
        reason="replicate reviewed release inventory",
        now=action_time + timedelta(minutes=2),
    )

    assert record == repeated
    assert record.release_count == 3
    assert record.commit_created_count == 3
    assert 2 <= instrumented.max_active_puts <= 2
    assert backing.head(mirror.batch_intent_key(record.intent.batch_id)) is not None
    assert backing.head(mirror.batch_record_key(record.intent.batch_id)) is not None
    assert status.record == record
    assert status.state_counts[RunArtifactMirrorBatchMemberState.COMPLETED.value] == 3
    assert status.remaining_copy_object_count == 0
    for archive, receipt in zip(archives, receipts, strict=True):
        output = tmp_path / f"received-{receipt.archive_id}.tar.gz"
        RunArtifactTransport(backing).fetch(receipt.archive_id, output)
        assert output.read_bytes() == archive.read_bytes()

    with pytest.raises(ValueError, match="product cannot exceed"):
        mirror.execute_batch(
            plan,
            confirm_plan_id=plan.plan_id,
            operator="batch-mirror-operator",
            reason="replicate reviewed release inventory",
            release_workers=9,
            object_workers=8,
        )


def test_batch_failure_exposes_progress_and_resumes_original_plan(tmp_path: Path) -> None:
    _, receipts, source, _ = publish_releases(
        tmp_path,
        ("batch-resume-one", "batch-resume-two"),
    )
    backing = LocalBlobStore(tmp_path / "batch-resume-destination")
    failing = InstrumentedMirrorStore(
        backing,
        fail_key_fragment=f"{receipts[1].archive_id}/chunks/00000001-",
    )
    mirror = RunArtifactMirror(source, failing)
    plan = mirror.plan_batch(include_all_committed=True)
    operator = "batch-resume-operator"
    reason = "resume the exact partially completed mirror batch"

    with pytest.raises(RuntimeError, match="injected mirror write failure"):
        mirror.execute_batch(
            plan,
            confirm_plan_id=plan.plan_id,
            operator=operator,
            reason=reason,
            release_workers=2,
            object_workers=1,
        )
    batch_id = RunArtifactMirrorBatchIntent.expected_batch_id(
        plan_id=plan.plan_id,
        operator=operator,
        reason=reason,
    )
    assert backing.head(mirror.batch_intent_key(batch_id)) is not None
    assert backing.head(mirror.batch_record_key(batch_id)) is None

    status = RunArtifactMirror(source, backing).batch_status(
        plan,
        operator=operator,
        reason=reason,
    )
    assert status.state_counts[RunArtifactMirrorBatchMemberState.COMPLETED.value] == 1
    assert status.state_counts[RunArtifactMirrorBatchMemberState.PARTIAL.value] == 1
    assert status.present_copy_object_count is not None
    assert status.remaining_copy_object_count == 2

    completed = RunArtifactMirror(source, backing).execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator=operator,
        reason=reason,
        release_workers=1,
        object_workers=2,
    )
    final_status = RunArtifactMirror(source, backing).batch_status(
        plan,
        operator=operator,
        reason=reason,
    )
    assert completed.release_count == 2
    assert final_status.record == completed
    assert final_status.state_counts[RunArtifactMirrorBatchMemberState.COMPLETED.value] == 2


def test_batch_execution_rejects_selected_drift_but_ignores_new_unselected_prefix(
    tmp_path: Path,
) -> None:
    _, receipts, source, _ = publish_releases(
        tmp_path,
        ("batch-drift-one", "batch-drift-two"),
    )
    destination = LocalBlobStore(tmp_path / "batch-drift-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    unrelated_id = "run_archive_" + "f" * 24
    unrelated_key = f"{RunArtifactTransport.release_root(unrelated_id)}/note.txt"
    destination.put_if_absent(unrelated_key, b"unselected")
    completed = mirror.execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-drift-operator",
        reason="mirror the explicitly selected release only",
    )
    assert completed.release_count == 1
    assert destination.get(unrelated_key) == b"unselected"
    selected_chunk = plan.releases[0].manifest.chunks[0]
    (destination.root / selected_chunk.key).write_bytes(b"X" * selected_chunk.size_bytes)
    post_completion_drift = mirror.batch_status(
        plan,
        operator="batch-drift-operator",
        reason="mirror the explicitly selected release only",
    )
    assert post_completion_drift.record == completed
    assert post_completion_drift.state_counts[RunArtifactMirrorBatchMemberState.INVALID.value] == 1

    drift_destination = LocalBlobStore(tmp_path / "batch-selected-drift")
    drift_mirror = RunArtifactMirror(source, drift_destination)
    drift_plan = drift_mirror.plan_batch(archive_ids=(receipts[1].archive_id,))
    drift_destination.put_if_absent(
        f"{RunArtifactTransport.release_root(receipts[1].archive_id)}/late.bin",
        b"late",
    )
    operator = "batch-drift-operator"
    reason = "reject changes inside the selected release prefix"
    with pytest.raises(RunArtifactMirrorError, match="unplanned object"):
        drift_mirror.execute_batch(
            drift_plan,
            confirm_plan_id=drift_plan.plan_id,
            operator=operator,
            reason=reason,
        )
    status = drift_mirror.batch_status(
        drift_plan,
        operator=operator,
        reason=reason,
    )
    assert status.state_counts[RunArtifactMirrorBatchMemberState.INVALID.value] == 1
    assert status.present_copy_object_count is None


def test_batch_local_to_s3_and_incomplete_destination_commit_are_handled(
    tmp_path: Path,
) -> None:
    archives, receipts, source, _ = publish_releases(
        tmp_path,
        ("batch-s3-one", "batch-s3-two"),
    )
    client = FakeS3Client()
    destination = S3ConditionalBlobStore(client, bucket="batch-bucket", prefix="replica")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(include_all_committed=True, max_workers=2)
    record = mirror.execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-s3-operator",
        reason="copy reviewed releases to S3 compatible storage",
        release_workers=2,
        object_workers=1,
    )
    assert record.release_count == 2
    assert RunArtifactTransport(destination).list_committed_archive_ids() == tuple(
        sorted(item.archive_id for item in receipts)
    )
    for archive, receipt in zip(archives, receipts, strict=True):
        output = tmp_path / f"s3-{receipt.archive_id}.tar.gz"
        RunArtifactTransport(destination).fetch(receipt.archive_id, output)
        assert output.read_bytes() == archive.read_bytes()

    broken = LocalBlobStore(tmp_path / "incomplete-commit")
    source_transport = RunArtifactTransport(source)
    commit, _, _, _ = source_transport.inspect_committed_release(receipts[0].archive_id)
    broken.put_if_absent(
        RunArtifactTransport.commit_key(receipts[0].archive_id),
        commit.canonical_bytes() + b"\n",
    )
    with pytest.raises(ValueError, match="complete destination graph"):
        RunArtifactMirror(source, broken).plan(receipts[0].archive_id)


def test_batch_cli_plans_executes_and_reports_authenticated_progress(tmp_path: Path) -> None:
    _, receipts, source, signer = publish_releases(
        tmp_path,
        ("batch-cli-one", "batch-cli-two"),
        signed=True,
    )
    assert signer is not None
    destination = tmp_path / "batch-cli-destination"
    public_key = tmp_path / "batch-cli.pub"
    public_key.write_text(signer.public_key_base64 + "\n", encoding="ascii")
    plan_path = tmp_path / "batch-cli.plan.json"
    record_path = tmp_path / "batch-cli.record.json"
    status_path = tmp_path / "batch-cli.status.json"
    runner = CliRunner()

    planned = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch-plan",
            "--all-committed",
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination),
            "--public-key",
            str(public_key),
            "--require-attestation",
            "--workers",
            "2",
            "--output",
            str(plan_path),
        ],
    )
    assert planned.exit_code == 0
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_payload["selected_archive_ids"] == sorted(item.archive_id for item in receipts)
    assert runner.invoke(app, ["run-artifacts-mirror-batch", str(plan_path)]).exit_code == 0

    operator = "batch-cli-operator"
    reason = "execute reviewed authenticated release batch"
    missing = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch",
            str(plan_path),
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination),
            "--execute",
        ],
    )
    assert missing.exit_code == 2
    executed = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch",
            str(plan_path),
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination),
            "--public-key",
            str(public_key),
            "--execute",
            "--confirm-plan-id",
            plan_payload["plan_id"],
            "--operator",
            operator,
            "--reason",
            reason,
            "--release-workers",
            "2",
            "--object-workers",
            "1",
            "--output",
            str(record_path),
        ],
    )
    assert executed.exit_code == 0
    assert record_path.is_file()
    status_result = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch-status",
            str(plan_path),
            "--operator",
            operator,
            "--reason",
            reason,
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination),
            "--public-key",
            str(public_key),
            "--fail-on-invalid",
            "--fail-on-incomplete",
            "--output",
            str(status_path),
        ],
    )
    assert status_result.exit_code == 0
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["state_counts"]["completed"] == 2


def test_batch_status_distinguishes_pending_and_destination_complete(tmp_path: Path) -> None:
    _, receipts, source, _ = publish_releases(tmp_path, ("batch-status-phases",))
    destination = LocalBlobStore(tmp_path / "batch-status-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    operator = "batch-status-operator"
    reason = "inspect pending and destination complete phases"

    with pytest.raises(RunArtifactMirrorError, match="intent was not found"):
        mirror.batch_status(plan, operator=operator, reason=reason)
    with pytest.raises(ValueError, match="operator cannot be empty"):
        mirror.batch_status(plan, operator=" ", reason=reason)
    with pytest.raises(ValueError, match="at least eight"):
        mirror.batch_status(plan, operator=operator, reason="short")

    batch_id = RunArtifactMirrorBatchIntent.expected_batch_id(
        plan_id=plan.plan_id,
        operator=operator,
        reason=reason,
    )
    intent = RunArtifactMirrorBatchIntent(
        batch_id=batch_id,
        plan=plan,
        operator=operator,
        reason=reason,
        created_at=datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc),
    )
    destination.put_if_absent(
        mirror.batch_intent_key(batch_id),
        intent.canonical_bytes() + b"\n",
    )
    pending = mirror.batch_status(plan, operator=operator, reason=reason)
    assert pending.state_counts[RunArtifactMirrorBatchMemberState.PENDING.value] == 1
    assert pending.present_copy_object_count == 0
    assert pending.remaining_copy_object_count == plan.copy_object_count

    for item in plan.releases[0].objects:
        destination.put_if_absent(item.reference.key, source.get(item.reference.key))
    destination_complete = mirror.batch_status(plan, operator=operator, reason=reason)
    assert (
        destination_complete.state_counts[
            RunArtifactMirrorBatchMemberState.DESTINATION_COMPLETE.value
        ]
        == 1
    )
    assert destination_complete.remaining_copy_object_count == 0
    assert destination_complete.record is None

    completed = mirror.execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator=operator,
        reason=reason,
        now=datetime(2026, 9, 18, 22, 1, tzinfo=timezone.utc),
    )
    assert completed.created_object_count == 0
    assert completed.reused_object_count == plan.object_count


def test_batch_rejects_invalid_selection_confirmation_and_worker_bounds(tmp_path: Path) -> None:
    _, receipts, source, _ = publish_releases(tmp_path, ("batch-input-validation",))
    destination = LocalBlobStore(tmp_path / "batch-input-destination")
    mirror = RunArtifactMirror(source, destination)
    missing_id = "run_archive_" + "e" * 24
    staged_id = "run_archive_" + "d" * 24
    source.put_if_absent(
        f"{RunArtifactTransport.release_root(staged_id)}/staged.bin",
        b"staged",
    )

    with pytest.raises(ValueError, match="planning worker count"):
        mirror.plan_batch(include_all_committed=True, max_workers=0)
    with pytest.raises(RunArtifactMirrorError, match="was not found"):
        mirror.plan_batch(archive_ids=(missing_id,))
    with pytest.raises(RunArtifactMirrorError, match="is not committed"):
        mirror.plan_batch(archive_ids=(staged_id,))
    with pytest.raises(RunArtifactMirrorError, match="contains no releases"):
        RunArtifactMirror(
            LocalBlobStore(tmp_path / "empty-source"),
            LocalBlobStore(tmp_path / "empty-destination"),
        ).plan_batch(include_all_committed=True)

    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    for kwargs, message in (
        ({"confirm_plan_id": "run_mirror_batch_plan_" + "0" * 24}, "plan ID"),
        ({"operator": " "}, "operator cannot be empty"),
        ({"reason": "short"}, "at least eight"),
        ({"release_workers": 0}, "release worker count"),
        ({"object_workers": 0}, "object worker count"),
    ):
        arguments: dict[str, Any] = {
            "confirm_plan_id": plan.plan_id,
            "operator": "batch-input-operator",
            "reason": "validate all batch execution inputs",
        }
        arguments.update(kwargs)
        with pytest.raises((RunArtifactMirrorError, ValueError), match=message):
            mirror.execute_batch(plan, **arguments)


def test_batch_contracts_reject_inconsistent_plans_records_and_status(tmp_path: Path) -> None:
    _, _, source, _ = publish_releases(
        tmp_path,
        ("batch-contract-one", "batch-contract-two"),
    )
    destination = LocalBlobStore(tmp_path / "batch-contract-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(include_all_committed=True)
    plan_payload = plan.model_dump(mode="python")
    changed_source_releases = (
        plan.source_inventory.releases[0].model_copy(
            update={"commit_id": "run_release_" + "0" * 24}
        ),
        *plan.source_inventory.releases[1:],
    )
    changed_source_inventory = RunArtifactStoreInventory.model_validate(
        {
            **plan.source_inventory.model_dump(mode="python"),
            "releases": changed_source_releases,
            "state_digest": RunArtifactStoreInventory.expected_state_digest(
                releases=changed_source_releases,
                invalid_keys=plan.source_inventory.invalid_keys,
            ),
        }
    )
    for update, message in (
        (
            {
                "destination_inventory": plan.destination_inventory.model_copy(
                    update={
                        "observed_at": plan.destination_inventory.observed_at + timedelta(seconds=1)
                    }
                )
            },
            "share one observation time",
        ),
        ({"release_count": plan.release_count + 1}, "release count"),
        ({"object_count": plan.object_count + 1}, "object count"),
        ({"copy_object_count": plan.copy_object_count + 1}, "copy count"),
        ({"reuse_object_count": plan.reuse_object_count + 1}, "reuse count"),
        ({"copy_bytes": plan.copy_bytes + 1}, "copy bytes"),
        ({"reuse_bytes": plan.reuse_bytes + 1}, "reuse bytes"),
        ({"plan_id": "run_mirror_batch_plan_" + "0" * 24}, "plan ID"),
        (
            {"selected_archive_ids": tuple(reversed(plan.selected_archive_ids))},
            "sorted and unique|release order",
        ),
        ({"releases": tuple(reversed(plan.releases))}, "release order"),
        (
            {"source_inventory": changed_source_inventory},
            "not committed in its source inventory",
        ),
        (
            {"require_attestation": True},
            "trust policy",
        ),
        (
            {
                "selected_archive_ids": plan.selected_archive_ids[:1],
                "releases": plan.releases[:1],
                "release_count": 1,
                "object_count": len(plan.releases[0].objects),
                "copy_object_count": plan.releases[0].copy_object_count,
                "reuse_object_count": plan.releases[0].reuse_object_count,
                "copy_bytes": plan.releases[0].copy_bytes,
                "reuse_bytes": plan.releases[0].reuse_bytes,
            },
            "all-committed.*incomplete",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchPlan.model_validate({**plan_payload, **update})

    action_time = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
    record = mirror.execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-contract-operator",
        reason="validate mirror batch evidence contracts",
        now=action_time,
    )
    intent_payload = record.intent.model_dump(mode="python")
    with pytest.raises(ValueError, match="time must be timezone-aware"):
        RunArtifactMirrorBatchIntent.model_validate(
            {**intent_payload, "created_at": action_time.replace(tzinfo=None)}
        )
    with pytest.raises(ValueError, match="batch ID"):
        RunArtifactMirrorBatchIntent.model_validate(
            {**intent_payload, "batch_id": "run_mirror_batch_" + "0" * 24}
        )
    record_payload = record.model_dump(mode="python")
    changed_member_operator = "other-operator"
    changed_member_record = RunArtifactMirrorRecord.model_validate(
        {
            **record.records[0].model_dump(mode="python"),
            "operator": changed_member_operator,
            "mirror_id": RunArtifactMirrorRecord.expected_mirror_id(
                plan_id=record.records[0].plan.plan_id,
                operator=changed_member_operator,
                reason=record.records[0].reason,
            ),
        }
    )
    for update, message in (
        (
            {"completed_at": record.completed_at.replace(tzinfo=None)},
            "completion time must be timezone-aware",
        ),
        ({"records": ()}, "cover every release"),
        ({"created_object_count": record.created_object_count + 1}, "created count"),
        ({"reused_object_count": record.reused_object_count + 1}, "reused count"),
        ({"created_bytes": record.created_bytes + 1}, "created bytes"),
        ({"reused_bytes": record.reused_bytes + 1}, "reused bytes"),
        ({"commit_created_count": record.commit_created_count + 1}, "commit count"),
        (
            {
                "records": (
                    changed_member_record,
                    *record.records[1:],
                )
            },
            "member evidence",
        ),
        ({"completed_at": action_time - timedelta(seconds=1)}, "completion precedes"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchRecord.model_validate({**record_payload, **update})

    status = mirror.batch_status(
        plan,
        operator="batch-contract-operator",
        reason="validate mirror batch evidence contracts",
        now=action_time + timedelta(seconds=1),
    )
    status_payload = status.model_dump(mode="python")
    member = status.members[0]
    member_payload = member.model_dump(mode="python")
    member_mutations = (
        (
            {
                "state": RunArtifactMirrorBatchMemberState.INVALID,
                "record": None,
            },
            "cannot claim verified progress",
        ),
        ({"present_copy_object_count": None}, "requires complete progress"),
        (
            {"present_copy_object_count": member.present_copy_object_count + 1},
            "progress count",
        ),
        ({"present_copy_bytes": member.present_copy_bytes + 1}, "progress bytes"),
        ({"state": RunArtifactMirrorBatchMemberState.PENDING, "record": None}, "pending"),
        (
            {
                "state": RunArtifactMirrorBatchMemberState.PARTIAL,
                "present_copy_object_count": 0,
                "remaining_copy_object_count": member.planned_copy_object_count,
                "present_copy_bytes": 0,
                "remaining_copy_bytes": member.planned_copy_bytes,
                "record": None,
            },
            "partial",
        ),
        (
            {
                "state": RunArtifactMirrorBatchMemberState.DESTINATION_COMPLETE,
                "present_copy_object_count": member.planned_copy_object_count - 1,
                "remaining_copy_object_count": 1,
                "record": None,
            },
            "complete mirror member",
        ),
        ({"record": None}, "requires matching evidence"),
        (
            {
                "state": RunArtifactMirrorBatchMemberState.DESTINATION_COMPLETE,
            },
            "incomplete mirror member",
        ),
    )
    for update, message in member_mutations:
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchMemberStatus.model_validate({**member_payload, **update})

    for update, message in (
        (
            {"observed_at": status.observed_at.replace(tzinfo=None)},
            "status time must be timezone-aware",
        ),
        ({"release_count": status.release_count + 1}, "cover every release"),
        ({"members": tuple(reversed(status.members))}, "status order"),
        (
            {
                "members": (
                    status.members[0].model_copy(
                        update={
                            "planned_copy_object_count": status.members[0].planned_copy_object_count
                            + 1,
                            "present_copy_object_count": status.members[0].present_copy_object_count
                            + 1,
                        }
                    ),
                    *status.members[1:],
                )
            },
            "member status does not match",
        ),
        ({"state_counts": {}}, "status counts"),
        (
            {"planned_copy_object_count": status.planned_copy_object_count + 1},
            "planned copy count",
        ),
        ({"planned_copy_bytes": status.planned_copy_bytes + 1}, "planned bytes"),
        ({"present_copy_object_count": None}, "requires aggregate progress"),
        (
            {"present_copy_object_count": status.present_copy_object_count + 1},
            "present count",
        ),
        ({"present_copy_bytes": status.present_copy_bytes + 1}, "present bytes"),
        (
            {"remaining_copy_object_count": status.remaining_copy_object_count + 1},
            "remaining count",
        ),
        (
            {"remaining_copy_bytes": status.remaining_copy_bytes + 1},
            "remaining bytes",
        ),
        (
            {
                "members": (
                    status.members[0].model_copy(
                        update={
                            "state": RunArtifactMirrorBatchMemberState.INVALID,
                            "present_copy_object_count": None,
                            "present_copy_bytes": None,
                            "remaining_copy_object_count": None,
                            "remaining_copy_bytes": None,
                            "record": None,
                        }
                    ),
                    *status.members[1:],
                ),
                "state_counts": {
                    **status.state_counts,
                    RunArtifactMirrorBatchMemberState.COMPLETED.value: 1,
                    RunArtifactMirrorBatchMemberState.INVALID.value: 1,
                },
                "record": None,
            },
            "cannot claim aggregate progress",
        ),
        ({"state_digest": "0" * 64}, "status digest"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchStatus.model_validate({**status_payload, **update})

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    BlobInfo,
    RunArtifactMirrorBatchDecision,
    RunArtifactMirrorBatchDecisionKind,
    RunArtifactMirrorBatchEvidenceState,
    RunArtifactMirrorBatchHealth,
    RunArtifactMirrorBatchInspection,
    RunArtifactMirrorBatchIntent,
    RunArtifactMirrorBatchLedgerEntry,
    RunArtifactMirrorBatchPolicy,
    RunArtifactMirrorBatchResolution,
    RunArtifactMirrorBatchResolutionBasis,
    RunArtifactMirrorBatchResolutionKind,
    RunArtifactMirrorBatchStatus,
    RunArtifactMirrorOperationsLedger,
)
from agentic_rl_forge.storage import (
    LocalBlobStore,
    RunArtifactMirror,
    RunArtifactMirrorError,
    RunArtifactTransport,
    S3ConditionalBlobStore,
)
from test_blob_stores import FakeS3Client
from test_run_artifact_mirror import InstrumentedMirrorStore
from test_run_artifact_mirror_batch import publish_releases


class PausingReleaseWriteStore:
    def __init__(self, delegate: LocalBlobStore) -> None:
        self.delegate = delegate
        self.write_started = threading.Event()
        self.resume_write = threading.Event()
        self._paused = False
        self._lock = threading.Lock()

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        with self._lock:
            pause = key.startswith("run-releases/") and not self._paused
            if pause:
                self._paused = True
        if pause:
            self.write_started.set()
            if not self.resume_write.wait(timeout=60):
                raise RuntimeError("timed out waiting to resume the release write")
        return self.delegate.put_if_absent(key, data, metadata=metadata)

    def get(self, key: str) -> bytes:
        return self.delegate.get(key)

    def head(self, key: str) -> BlobInfo | None:
        return self.delegate.head(key)

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return self.delegate.list(prefix)

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        return self.delegate.delete_if_match(key, expected)


def persist_intent(
    mirror: RunArtifactMirror,
    plan: Any,
    *,
    operator: str,
    reason: str,
    created_at: datetime,
) -> RunArtifactMirrorBatchIntent:
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
        created_at=created_at,
    )
    mirror.destination.put_if_absent(
        mirror.batch_intent_key(batch_id),
        intent.canonical_bytes() + b"\n",
    )
    return intent


def test_batch_policy_caps_release_count_copy_bytes_and_plan_identity(tmp_path: Path) -> None:
    _, receipts, source, _ = publish_releases(
        tmp_path,
        ("ledger-policy-one", "ledger-policy-two"),
    )
    destination = LocalBlobStore(tmp_path / "ledger-policy-destination")
    mirror = RunArtifactMirror(source, destination)

    with pytest.raises(RunArtifactMirrorError, match="release count exceeds"):
        mirror.plan_batch(include_all_committed=True, max_release_count=1)
    with pytest.raises(RunArtifactMirrorError, match="copy bytes exceed"):
        mirror.plan_batch(
            archive_ids=(receipts[0].archive_id,),
            max_copy_bytes=0,
        )

    plan = mirror.plan_batch(
        include_all_committed=True,
        max_release_count=2,
        max_copy_bytes=1_000_000,
    )
    broader = mirror.plan_batch(
        include_all_committed=True,
        max_release_count=3,
        max_copy_bytes=1_000_000,
    )
    assert plan.policy == RunArtifactMirrorBatchPolicy(
        max_release_count=2,
        max_copy_bytes=1_000_000,
    )
    assert plan.release_count == 2
    assert plan.copy_bytes <= plan.policy.max_copy_bytes
    assert plan.plan_id != broader.plan_id


def test_destination_ledger_discovers_intent_complete_and_unclassified_evidence(
    tmp_path: Path,
) -> None:
    _, receipts, source, _ = publish_releases(
        tmp_path,
        ("ledger-list-one", "ledger-list-two"),
    )
    destination = LocalBlobStore(tmp_path / "ledger-list-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    pending_intent = persist_intent(
        mirror,
        plan,
        operator="ledger-pending-operator",
        reason="retain an unstarted mirror intent",
        created_at=datetime(2026, 9, 18, 23, 0, tzinfo=timezone.utc),
    )
    complete = mirror.execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="ledger-complete-operator",
        reason="complete a separately identified mirror operation",
        now=datetime(2026, 9, 18, 23, 1, tzinfo=timezone.utc),
    )
    unknown_key = f"{mirror.BATCH_ROOT}/{pending_intent.batch_id}/operator-note.txt"
    destination.put_if_absent(unknown_key, b"note")

    ledger = mirror.operations_ledger(now=datetime(2026, 9, 18, 23, 2, tzinfo=timezone.utc))
    repeated = mirror.operations_ledger(now=datetime(2026, 9, 18, 23, 3, tzinfo=timezone.utc))
    entries = {item.batch_id: item for item in ledger.entries}

    assert ledger.batch_count == 2
    assert ledger.state_digest == repeated.state_digest
    assert ledger.unclassified_keys == (unknown_key,)
    assert entries[pending_intent.batch_id].state is RunArtifactMirrorBatchEvidenceState.INTENT_ONLY
    assert entries[complete.intent.batch_id].state is RunArtifactMirrorBatchEvidenceState.COMPLETE
    assert entries[complete.intent.batch_id].member_record_count == 1
    inspection = mirror.inspect_batch(complete.intent.batch_id)
    assert inspection.health is RunArtifactMirrorBatchHealth.COMPLETE
    assert inspection.status.record == complete


def test_non_started_batch_can_be_cancelled_and_blocks_future_execution(tmp_path: Path) -> None:
    _, receipts, source, _ = publish_releases(tmp_path, ("ledger-cancel",))
    destination = LocalBlobStore(tmp_path / "ledger-cancel-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    operator = "ledger-cancel-operator"
    reason = "prepare an operation that will be cancelled"
    intent = persist_intent(
        mirror,
        plan,
        operator=operator,
        reason=reason,
        created_at=datetime(2026, 9, 18, 23, 10, tzinfo=timezone.utc),
    )
    inspection = mirror.inspect_batch(intent.batch_id)
    assert inspection.health is RunArtifactMirrorBatchHealth.PENDING
    assert inspection.resolution_allowed
    assert inspection.resolution_basis is RunArtifactMirrorBatchResolutionBasis.NOT_STARTED

    with pytest.raises(RunArtifactMirrorError, match="digest does not match"):
        mirror.resolve_batch(
            intent.batch_id,
            kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
            confirm_status_digest="0" * 64,
            resolver="release-manager",
            reason="cancel the reviewed but unstarted operation",
        )
    resolution = mirror.resolve_batch(
        intent.batch_id,
        kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
        confirm_status_digest=inspection.status.state_digest,
        resolver="release-manager",
        reason="cancel the reviewed but unstarted operation",
        now=datetime(2026, 9, 18, 23, 11, tzinfo=timezone.utc),
    )
    repeated = mirror.resolve_batch(
        intent.batch_id,
        kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
        confirm_status_digest=inspection.status.state_digest,
        resolver="release-manager",
        reason="cancel the reviewed but unstarted operation",
    )
    assert resolution == repeated
    assert resolution.basis is RunArtifactMirrorBatchResolutionBasis.NOT_STARTED
    assert mirror.inspect_batch(intent.batch_id).health is RunArtifactMirrorBatchHealth.RESOLVED
    with pytest.raises(RunArtifactMirrorError, match="has been resolved"):
        mirror.execute_batch(
            plan,
            confirm_plan_id=plan.plan_id,
            operator=operator,
            reason=reason,
        )
    with pytest.raises(RunArtifactMirrorError, match="differs from this command"):
        mirror.resolve_batch(
            intent.batch_id,
            kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
            confirm_status_digest=inspection.status.state_digest,
            resolver="another-manager",
            reason="cancel the reviewed but unstarted operation",
        )


def test_terminal_decision_recovers_missing_completion_and_resolution_sidecars(
    tmp_path: Path,
) -> None:
    _, receipts, source, _ = publish_releases(
        tmp_path,
        ("ledger-decision-complete", "ledger-decision-resolve"),
    )
    destination = LocalBlobStore(tmp_path / "ledger-decision-destination")
    mirror = RunArtifactMirror(source, destination)
    complete_plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    completed = mirror.execute_batch(
        complete_plan,
        confirm_plan_id=complete_plan.plan_id,
        operator="ledger-decision-complete-operator",
        reason="verify completion decision sidecar recovery",
    )
    completion_decision = RunArtifactMirrorBatchDecision.model_validate_json(
        destination.get(mirror.batch_decision_key(completed.intent.batch_id))
    )
    assert completion_decision.kind is RunArtifactMirrorBatchDecisionKind.COMPLETED
    record_key = mirror.batch_record_key(completed.intent.batch_id)
    record_identity = destination.head(record_key)
    assert record_identity is not None
    assert destination.delete_if_match(record_key, record_identity)
    inspection = mirror.inspect_batch(completed.intent.batch_id)
    assert inspection.health is RunArtifactMirrorBatchHealth.COMPLETE
    assert inspection.status.record == completed
    recovered = mirror.execute_batch(
        complete_plan,
        confirm_plan_id=complete_plan.plan_id,
        operator=completed.intent.operator,
        reason=completed.intent.reason,
    )
    assert recovered == completed
    assert destination.head(record_key) is not None

    resolve_plan = mirror.plan_batch(archive_ids=(receipts[1].archive_id,))
    resolve_intent = persist_intent(
        mirror,
        resolve_plan,
        operator="ledger-decision-resolve-operator",
        reason="verify resolution decision sidecar recovery",
        created_at=datetime(2026, 9, 18, 23, 15, tzinfo=timezone.utc),
    )
    pending = mirror.inspect_batch(resolve_intent.batch_id)
    resolved = mirror.resolve_batch(
        resolve_intent.batch_id,
        kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
        confirm_status_digest=pending.status.state_digest,
        resolver="release-manager",
        reason="cancel and recover the resolution sidecar",
    )
    resolution_decision = RunArtifactMirrorBatchDecision.model_validate_json(
        destination.get(mirror.batch_decision_key(resolve_intent.batch_id))
    )
    assert resolution_decision.kind is RunArtifactMirrorBatchDecisionKind.RESOLVED
    resolution_key = mirror.batch_resolution_key(resolve_intent.batch_id)
    resolution_identity = destination.head(resolution_key)
    assert resolution_identity is not None
    assert destination.delete_if_match(resolution_key, resolution_identity)
    assert mirror.inspect_batch(resolve_intent.batch_id).resolution == resolved
    with pytest.raises(RunArtifactMirrorError, match="has been resolved"):
        mirror.execute_batch(
            resolve_plan,
            confirm_plan_id=resolve_plan.plan_id,
            operator=resolve_intent.operator,
            reason=resolve_intent.reason,
        )
    recovered_resolution = mirror.resolve_batch(
        resolve_intent.batch_id,
        kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
        confirm_status_digest=pending.status.state_digest,
        resolver="release-manager",
        reason="cancel and recover the resolution sidecar",
    )
    assert recovered_resolution == resolved
    assert destination.head(resolution_key) is not None


def test_resolution_decision_wins_atomically_against_in_flight_completion(
    tmp_path: Path,
) -> None:
    _, receipts, source, _ = publish_releases(tmp_path, ("ledger-decision-race",))
    destination = LocalBlobStore(tmp_path / "ledger-decision-race-destination")
    pausing = PausingReleaseWriteStore(destination)
    mirror = RunArtifactMirror(source, pausing)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    operator = "ledger-decision-race-operator"
    reason = "race terminal resolution against completion"

    with ThreadPoolExecutor(max_workers=1) as executor:
        execution = executor.submit(
            mirror.execute_batch,
            plan,
            confirm_plan_id=plan.plan_id,
            operator=operator,
            reason=reason,
        )
        try:
            assert pausing.write_started.wait(timeout=60)
            batch_id = RunArtifactMirrorBatchIntent.expected_batch_id(
                plan_id=plan.plan_id,
                operator=operator,
                reason=reason,
            )
            pending = mirror.inspect_batch(batch_id)
            assert pending.health is RunArtifactMirrorBatchHealth.PENDING
            resolution = mirror.resolve_batch(
                batch_id,
                kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
                confirm_status_digest=pending.status.state_digest,
                resolver="release-manager",
                reason="cancel before the first destination object is created",
            )
        finally:
            pausing.resume_write.set()
        with pytest.raises(RunArtifactMirrorError, match="resolved while member execution"):
            execution.result(timeout=60)

    decision = RunArtifactMirrorBatchDecision.model_validate_json(
        destination.get(mirror.batch_decision_key(batch_id))
    )
    assert decision.kind is RunArtifactMirrorBatchDecisionKind.RESOLVED
    assert decision.resolved == resolution
    assert destination.head(mirror.batch_record_key(batch_id)) is None
    ledger = RunArtifactMirror(source, destination).operations_ledger()
    assert ledger.entries[0].state is RunArtifactMirrorBatchEvidenceState.RESOLVED


def test_invalid_batch_can_be_superseded_but_active_or_complete_batches_cannot(
    tmp_path: Path,
) -> None:
    _, receipts, source, _ = publish_releases(
        tmp_path,
        ("ledger-supersede-one", "ledger-supersede-two"),
    )
    destination = LocalBlobStore(tmp_path / "ledger-supersede-destination")
    mirror = RunArtifactMirror(source, destination)
    invalid_plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    destination.put_if_absent(
        f"{RunArtifactTransport.release_root(receipts[0].archive_id)}/late.bin",
        b"late",
    )
    invalid_operator = "ledger-invalid-operator"
    invalid_reason = "retire a permanently drifted selected prefix"
    with pytest.raises(RunArtifactMirrorError, match="unplanned object"):
        mirror.execute_batch(
            invalid_plan,
            confirm_plan_id=invalid_plan.plan_id,
            operator=invalid_operator,
            reason=invalid_reason,
        )
    invalid_batch_id = RunArtifactMirrorBatchIntent.expected_batch_id(
        plan_id=invalid_plan.plan_id,
        operator=invalid_operator,
        reason=invalid_reason,
    )
    blocked = mirror.inspect_batch(invalid_batch_id)
    assert blocked.health is RunArtifactMirrorBatchHealth.BLOCKED
    replacement_plan_id = "run_mirror_batch_plan_" + "a" * 24
    superseded = mirror.resolve_batch(
        invalid_batch_id,
        kind=RunArtifactMirrorBatchResolutionKind.SUPERSEDED,
        confirm_status_digest=blocked.status.state_digest,
        resolver="release-manager",
        reason="supersede the permanently drifted mirror intent",
        replacement_plan_id=replacement_plan_id,
    )
    assert superseded.basis is RunArtifactMirrorBatchResolutionBasis.INVALID
    assert superseded.replacement_plan_id == replacement_plan_id

    active_destination = LocalBlobStore(tmp_path / "ledger-active-destination")
    active_plan = RunArtifactMirror(source, active_destination).plan_batch(
        archive_ids=(receipts[1].archive_id,)
    )
    failing = InstrumentedMirrorStore(
        active_destination,
        fail_key_fragment=f"{receipts[1].archive_id}/chunks/00000001-",
    )
    active_mirror = RunArtifactMirror(source, failing)
    active_operator = "ledger-active-operator"
    active_reason = "leave exact partial progress for inspection"
    with pytest.raises(RuntimeError, match="injected mirror write failure"):
        active_mirror.execute_batch(
            active_plan,
            confirm_plan_id=active_plan.plan_id,
            operator=active_operator,
            reason=active_reason,
        )
    active_batch_id = RunArtifactMirrorBatchIntent.expected_batch_id(
        plan_id=active_plan.plan_id,
        operator=active_operator,
        reason=active_reason,
    )
    active_inspection = RunArtifactMirror(source, active_destination).inspect_batch(active_batch_id)
    assert active_inspection.health is RunArtifactMirrorBatchHealth.ACTIVE
    assert not active_inspection.resolution_allowed
    with pytest.raises(RunArtifactMirrorError, match="not eligible"):
        RunArtifactMirror(source, active_destination).resolve_batch(
            active_batch_id,
            kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
            confirm_status_digest=active_inspection.status.state_digest,
            resolver="release-manager",
            reason="attempt to cancel an active mirror operation",
        )

    completed_destination = LocalBlobStore(tmp_path / "ledger-completed-destination")
    completed_mirror = RunArtifactMirror(source, completed_destination)
    completed_plan = completed_mirror.plan_batch(archive_ids=(receipts[1].archive_id,))
    completed = completed_mirror.execute_batch(
        completed_plan,
        confirm_plan_id=completed_plan.plan_id,
        operator="ledger-completed-operator",
        reason="finish a mirror operation before resolution",
    )
    completed_inspection = completed_mirror.inspect_batch(completed.intent.batch_id)
    with pytest.raises(RunArtifactMirrorError, match=r"completed.*cannot be resolved"):
        completed_mirror.resolve_batch(
            completed.intent.batch_id,
            kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
            confirm_status_digest=completed_inspection.status.state_digest,
            resolver="release-manager",
            reason="attempt to cancel a completed mirror operation",
        )


def test_s3_destination_ledger_and_batch_lookup_are_provider_neutral(tmp_path: Path) -> None:
    _, receipts, source, _ = publish_releases(tmp_path, ("ledger-s3",))
    client = FakeS3Client()
    destination = S3ConditionalBlobStore(client, bucket="ledger-bucket", prefix="replica")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    record = mirror.execute_batch(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="ledger-s3-operator",
        reason="record provider neutral mirror ledger evidence",
    )

    ledger = mirror.operations_ledger()
    inspection = mirror.inspect_batch(record.intent.batch_id)
    assert ledger.batch_count == 1
    assert ledger.entries[0].state is RunArtifactMirrorBatchEvidenceState.COMPLETE
    assert inspection.health is RunArtifactMirrorBatchHealth.COMPLETE
    assert client.last_put["Key"] == (f"replica/{mirror.batch_record_key(record.intent.batch_id)}")


def test_ledger_and_resolution_cli_are_preview_first_and_canonical(tmp_path: Path) -> None:
    _, receipts, source, _ = publish_releases(tmp_path, ("ledger-cli",))
    destination = LocalBlobStore(tmp_path / "ledger-cli-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    intent = persist_intent(
        mirror,
        plan,
        operator="ledger-cli-operator",
        reason="preview and cancel a CLI mirror operation",
        created_at=datetime(2026, 9, 18, 23, 30, tzinfo=timezone.utc),
    )
    runner = CliRunner()
    ledger_path = tmp_path / "ledger.json"
    inspection_path = tmp_path / "inspection.json"
    resolution_path = tmp_path / "resolution.json"

    listed = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch-list",
            "--destination-store-root",
            str(destination.root),
            "--output",
            str(ledger_path),
        ],
    )
    inspected = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch-inspect",
            intent.batch_id,
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination.root),
            "--fail-on-unhealthy",
            "--output",
            str(inspection_path),
        ],
    )
    assert listed.exit_code == 0
    assert inspected.exit_code == 1
    inspection_payload = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert inspection_payload["health"] == "pending"

    preview = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch-resolve",
            intent.batch_id,
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination.root),
        ],
    )
    assert preview.exit_code == 0
    missing = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch-resolve",
            intent.batch_id,
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination.root),
            "--execute",
        ],
    )
    assert missing.exit_code == 2
    resolved = runner.invoke(
        app,
        [
            "run-artifacts-mirror-batch-resolve",
            intent.batch_id,
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination.root),
            "--execute",
            "--kind",
            "cancelled",
            "--confirm-status-digest",
            inspection_payload["status"]["state_digest"],
            "--resolver",
            "release-manager",
            "--reason",
            "cancel the previewed CLI mirror operation",
            "--output",
            str(resolution_path),
        ],
    )
    assert resolved.exit_code == 0
    assert json.loads(resolution_path.read_text(encoding="utf-8"))["kind"] == "cancelled"


def test_ledger_resolution_and_inspection_contracts_reject_inconsistency(
    tmp_path: Path,
) -> None:
    _, receipts, source, _ = publish_releases(tmp_path, ("ledger-contracts",))
    destination = LocalBlobStore(tmp_path / "ledger-contract-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan_batch(archive_ids=(receipts[0].archive_id,))
    intent = persist_intent(
        mirror,
        plan,
        operator="ledger-contract-operator",
        reason="validate operations ledger contracts",
        created_at=datetime(2026, 9, 18, 23, 40, tzinfo=timezone.utc),
    )
    inspection = mirror.inspect_batch(intent.batch_id)
    resolution = mirror.resolve_batch(
        intent.batch_id,
        kind=RunArtifactMirrorBatchResolutionKind.CANCELLED,
        confirm_status_digest=inspection.status.state_digest,
        resolver="release-manager",
        reason="cancel the contract validation operation",
    )
    decision = RunArtifactMirrorBatchDecision.model_validate_json(
        destination.get(mirror.batch_decision_key(intent.batch_id))
    )

    decision_payload = decision.model_dump(mode="python")
    for update, message in (
        ({"decided_at": decision.decided_at.replace(tzinfo=None)}, "timezone-aware"),
        ({"kind": RunArtifactMirrorBatchDecisionKind.COMPLETED}, "requires only"),
        ({"decision_id": "run_mirror_batch_decision_" + "0" * 24}, "decision ID"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchDecision.model_validate({**decision_payload, **update})

    resolution_payload = resolution.model_dump(mode="python")
    for update, message in (
        ({"created_at": resolution.created_at.replace(tzinfo=None)}, "timezone-aware"),
        ({"confirmed_state_counts": {}}, "state counts"),
        ({"release_count": 2}, "cover every release"),
        ({"basis": RunArtifactMirrorBatchResolutionBasis.INVALID}, "eligible status"),
        ({"replacement_plan_id": "run_mirror_batch_plan_" + "a" * 24}, "cannot name"),
        (
            {"resolution_id": "run_mirror_batch_resolution_" + "0" * 24},
            "resolution ID",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchResolution.model_validate({**resolution_payload, **update})

    resolved_inspection = mirror.inspect_batch(intent.batch_id)
    inspection_payload = resolved_inspection.model_dump(mode="python")
    for update, message in (
        ({"health": RunArtifactMirrorBatchHealth.PENDING}, "health is inconsistent"),
        ({"resolution_allowed": True}, "health is inconsistent"),
        ({"state_digest": "0" * 64}, "inspection digest"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchInspection.model_validate({**inspection_payload, **update})

    ledger = mirror.operations_ledger()
    entry_payload = ledger.entries[0].model_dump(mode="python")
    for update, message in (
        ({"member_record_count": 2}, "member count exceeds"),
        ({"resolution_kind": None}, "resolution fields"),
        ({"batch_record_present": True}, "both completed and resolved"),
        ({"state": RunArtifactMirrorBatchEvidenceState.INTENT_ONLY}, "state is inconsistent"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorBatchLedgerEntry.model_validate({**entry_payload, **update})
    ledger_payload = ledger.model_dump(mode="python")
    for update, message in (
        ({"observed_at": ledger.observed_at.replace(tzinfo=None)}, "timezone-aware"),
        ({"batch_count": 2}, "batch count"),
        ({"state_counts": {}}, "state counts"),
        ({"state_digest": "0" * 64}, "ledger digest"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorOperationsLedger.model_validate({**ledger_payload, **update})

    status_payload = inspection.status.model_dump(mode="python")
    with pytest.raises(ValueError):
        RunArtifactMirrorBatchStatus.model_validate({**status_payload, "state_digest": "0" * 64})

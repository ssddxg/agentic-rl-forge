from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentic_rl_forge.contracts import RunKind, RunLivenessState, RunManifest, RunStatus
from agentic_rl_forge.storage import (
    RunLeaseConflictError,
    RunReconciliationConflictError,
    SQLiteTrajectoryStore,
)


def test_stale_run_reconciliation_is_fenced_audited_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "reconciliation.db"
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run = RunManifest(
        name="stale rollout",
        kind=RunKind.ROLLOUT,
        started_at=started_at,
    )

    with SQLiteTrajectoryStore(database) as store:
        store.create_run(run)
        lease = store.acquire_run_lease(
            run.run_id,
            owner_id="lost-worker",
            ttl_s=10,
            now=started_at,
        )
        preview = store.preview_run_reconciliation(
            run.run_id,
            stale_after_s=60,
            now=started_at + timedelta(seconds=20),
        )
        later_preview = store.preview_run_reconciliation(
            run.run_id,
            stale_after_s=60,
            now=started_at + timedelta(seconds=30),
        )

        assert preview.eligible
        assert preview.liveness.state is RunLivenessState.STALE
        assert preview.liveness.detail == "lease_expired"
        assert later_preview.state_digest == preview.state_digest
        record = store.reconcile_stale_run(
            preview,
            operator_id="ops@example.com",
            reason="worker host was terminated",
            now=started_at + timedelta(seconds=31),
        )

        finished = store.get_run(run.run_id)
        heartbeat = store.get_run_heartbeat(run.run_id)
        assert finished is not None
        assert finished.status is RunStatus.FAILED
        assert finished.digest() == record.terminal_run_digest
        assert finished.metadata["reconciliation"]["reconciliation_id"] == record.reconciliation_id
        assert heartbeat is not None
        assert heartbeat.owner_id == record.takeover_owner_id
        assert heartbeat.epoch == 2
        assert heartbeat.released_at == record.reconciled_at
        assert record.prior_lease_epoch == 1
        assert record.takeover_epoch == 2
        assert store.get_run_reconciliation(run.run_id) == record
        assert store.list_run_reconciliations() == (record,)
        assert (
            store.reconcile_stale_run(
                preview,
                operator_id="ops@example.com",
                reason="worker host was terminated",
                now=started_at + timedelta(seconds=40),
            )
            == record
        )
        with pytest.raises(RunReconciliationConflictError, match="another reconciliation"):
            store.reconcile_stale_run(
                preview,
                operator_id="another-operator",
                reason="another reconciliation reason",
                now=started_at + timedelta(seconds=40),
            )
        with pytest.raises(RunLeaseConflictError, match="not running"):
            store.renew_run_lease(
                run.run_id,
                lease.token,
                ttl_s=10,
                now=started_at + timedelta(seconds=32),
            )

    connection = sqlite3.connect(database)
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        connection.execute(
            "UPDATE run_reconciliations SET operator_id = 'changed' WHERE run_id = ?",
            (run.run_id,),
        )
    connection.rollback()
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        connection.execute("DELETE FROM run_reconciliations WHERE run_id = ?", (run.run_id,))
    connection.close()


def test_reconciliation_preview_is_invalidated_when_worker_recovers(tmp_path: Path) -> None:
    database = tmp_path / "recovery-race.db"
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run = RunManifest(name="recovering rollout", kind=RunKind.ROLLOUT, started_at=started_at)

    with SQLiteTrajectoryStore(database) as store:
        store.create_run(run)
        first = store.acquire_run_lease(
            run.run_id,
            owner_id="worker-a",
            ttl_s=10,
            now=started_at,
        )
        stale_preview = store.preview_run_reconciliation(
            run.run_id,
            stale_after_s=60,
            now=started_at + timedelta(seconds=11),
        )
        takeover = store.acquire_run_lease(
            run.run_id,
            owner_id="worker-b",
            ttl_s=30,
            now=started_at + timedelta(seconds=12),
        )

        assert first.epoch == 1
        assert takeover.epoch == 2
        with pytest.raises(RunReconciliationConflictError, match="no longer eligible"):
            store.reconcile_stale_run(
                stale_preview,
                operator_id="operator",
                reason="suspected abandoned worker",
                now=started_at + timedelta(seconds=13),
            )
        active_preview = store.preview_run_reconciliation(
            run.run_id,
            stale_after_s=60,
            now=started_at + timedelta(seconds=13),
        )
        assert not active_preview.eligible
        assert active_preview.liveness.state is RunLivenessState.ACTIVE
        assert store.get_run(run.run_id).status is RunStatus.RUNNING
        assert store.get_run_reconciliation(run.run_id) is None


def test_missing_heartbeat_reconciliation_starts_epoch_one(tmp_path: Path) -> None:
    database = tmp_path / "missing-heartbeat.db"
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run = RunManifest(name="never leased", kind=RunKind.ROLLOUT, started_at=started_at)

    with SQLiteTrajectoryStore(database) as store:
        store.create_run(run)
        preview = store.preview_run_reconciliation(
            run.run_id,
            stale_after_s=60,
            now=started_at + timedelta(seconds=61),
        )
        record = store.reconcile_stale_run(
            preview,
            operator_id="operator",
            reason="startup process exited early",
            now=started_at + timedelta(seconds=62),
        )

    assert preview.liveness.detail == "heartbeat_missing"
    assert record.prior_lease_epoch is None
    assert record.takeover_epoch == 1

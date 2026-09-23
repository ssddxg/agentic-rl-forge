from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from agentic_rl_forge.contracts import (
    DataOrigin,
    RunHeartbeat,
    RunLeaseToken,
    RunLiveness,
    RunLivenessState,
    RunManifest,
    RunReconciliationPreview,
    RunReconciliationRecord,
    RunStatus,
    Trajectory,
    TrajectoryStatus,
    utc_now,
)


class RunLeaseConflictError(RuntimeError):
    pass


class RunReconciliationConflictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrajectoryQuery:
    run_id: str | None = None
    task_id: str | None = None
    group_id: str | None = None
    policy_version: str | None = None
    environment_version: str | None = None
    statuses: tuple[TrajectoryStatus, ...] = ()
    origins: tuple[DataOrigin, ...] = ()
    limit: int | None = None
    offset: int = 0
    newest_first: bool = False

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 1:
            raise ValueError("query limit must be positive")
        if self.offset < 0:
            raise ValueError("query offset cannot be negative")


@dataclass(frozen=True, slots=True)
class StoreSummary:
    trajectory_count: int
    run_count: int
    task_count: int
    policy_versions: tuple[str, ...]
    environment_versions: tuple[str, ...]


class SQLiteTrajectoryStore:
    """Content-verified SQLite storage for runs and immutable trajectories."""

    _SCHEMA_VERSION = 3

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_digest TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    digest TEXT NOT NULL,
                    payload BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    environment_version TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_reward REAL NOT NULL,
                    generated_tokens INTEGER NOT NULL,
                    observation_tokens INTEGER NOT NULL,
                    step_count INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    digest TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    inserted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_trajectories (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
                    trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id)
                        ON DELETE RESTRICT,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (run_id, trajectory_id)
                );

                CREATE TABLE IF NOT EXISTS run_heartbeats (
                    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE RESTRICT,
                    owner_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL,
                    released_at TEXT
                );

                CREATE TABLE IF NOT EXISTS run_reconciliations (
                    reconciliation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id) ON DELETE RESTRICT,
                    state_digest TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    reconciled_at TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    payload BLOB NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS run_reconciliations_no_update
                BEFORE UPDATE ON run_reconciliations
                BEGIN
                    SELECT RAISE(ABORT, 'run reconciliation records are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS run_reconciliations_no_delete
                BEFORE DELETE ON run_reconciliations
                BEGIN
                    SELECT RAISE(ABORT, 'run reconciliation records are immutable');
                END;

                CREATE INDEX IF NOT EXISTS idx_trajectories_task
                    ON trajectories(task_id);
                CREATE INDEX IF NOT EXISTS idx_trajectories_group
                    ON trajectories(group_id);
                CREATE INDEX IF NOT EXISTS idx_trajectories_policy
                    ON trajectories(policy_version);
                CREATE INDEX IF NOT EXISTS idx_trajectories_status
                    ON trajectories(status);
                CREATE INDEX IF NOT EXISTS idx_run_trajectories_run
                    ON run_trajectories(run_id, position);
                """
            )
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(self._SCHEMA_VERSION),),
                )
            elif int(row["value"]) in {1, 2}:
                self._connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (str(self._SCHEMA_VERSION),),
                )
            elif int(row["value"]) != self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported trajectory store schema version {row['value']}")

    def create_run(self, manifest: RunManifest) -> bool:
        if manifest.status is not RunStatus.RUNNING:
            raise ValueError("new runs must start with running status")
        payload = manifest.canonical_bytes()
        digest = manifest.digest()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT digest FROM runs WHERE run_id = ?", (manifest.run_id,)
            ).fetchone()
            if existing is not None:
                if existing["digest"] != digest:
                    raise ValueError(f"run {manifest.run_id!r} already exists with other content")
                return False
            self._connection.execute(
                """
                INSERT INTO runs(
                    run_id, name, kind, status, config_digest, started_at,
                    completed_at, digest, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.name,
                    manifest.kind.value,
                    manifest.status.value,
                    manifest.config_digest,
                    manifest.started_at.isoformat(),
                    None,
                    digest,
                    payload,
                ),
            )
        return True

    def get_run(self, run_id: str) -> RunManifest | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return RunManifest.model_validate_json(row["payload"])

    def acquire_run_lease(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_s: float,
        now: datetime | None = None,
    ) -> RunHeartbeat:
        if not owner_id:
            raise ValueError("lease owner_id cannot be empty")
        if ttl_s <= 0:
            raise ValueError("lease ttl_s must be positive")
        current_time = self._aware_now(now)
        expires_at = current_time + timedelta(seconds=ttl_s)
        with self._lock, self._connection:
            self._require_running_run(run_id)
            row = self._connection.execute(
                "SELECT * FROM run_heartbeats WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None:
                heartbeat = self._heartbeat_from_row(row)
                active = heartbeat.released_at is None and heartbeat.lease_expires_at > current_time
                if active and heartbeat.owner_id != owner_id:
                    raise RunLeaseConflictError(
                        f"run {run_id!r} is leased by {heartbeat.owner_id!r} "
                        f"until {heartbeat.lease_expires_at.isoformat()}"
                    )
                if active:
                    if current_time < heartbeat.heartbeat_at:
                        raise ValueError("lease time cannot move backwards")
                    acquired_at = heartbeat.acquired_at
                    epoch = heartbeat.epoch
                else:
                    acquired_at = current_time
                    epoch = heartbeat.epoch + 1
            else:
                acquired_at = current_time
                epoch = 1
            self._connection.execute(
                """
                INSERT INTO run_heartbeats(
                    run_id, owner_id, epoch, acquired_at, heartbeat_at,
                    lease_expires_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(run_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    epoch = excluded.epoch,
                    acquired_at = excluded.acquired_at,
                    heartbeat_at = excluded.heartbeat_at,
                    lease_expires_at = excluded.lease_expires_at,
                    released_at = NULL
                """,
                (
                    run_id,
                    owner_id,
                    epoch,
                    acquired_at.isoformat(),
                    current_time.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return RunHeartbeat(
            run_id=run_id,
            owner_id=owner_id,
            epoch=epoch,
            acquired_at=acquired_at,
            heartbeat_at=current_time,
            lease_expires_at=expires_at,
        )

    def renew_run_lease(
        self,
        run_id: str,
        token: RunLeaseToken,
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> RunHeartbeat:
        if ttl_s <= 0:
            raise ValueError("lease ttl_s must be positive")
        current_time = self._aware_now(now)
        expires_at = current_time + timedelta(seconds=ttl_s)
        with self._lock, self._connection:
            self._require_running_run(run_id)
            heartbeat = self._require_active_lease(run_id, token, now=current_time)
            self._connection.execute(
                """
                UPDATE run_heartbeats
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE run_id = ?
                """,
                (current_time.isoformat(), expires_at.isoformat(), run_id),
            )
        return heartbeat.model_copy(
            update={"heartbeat_at": current_time, "lease_expires_at": expires_at}
        )

    def release_run_lease(
        self,
        run_id: str,
        token: RunLeaseToken,
        *,
        now: datetime | None = None,
    ) -> RunHeartbeat:
        current_time = self._aware_now(now)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM run_heartbeats WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunLeaseConflictError(f"run {run_id!r} has no lease")
            heartbeat = self._heartbeat_from_row(row)
            if heartbeat.owner_id != token.owner_id or heartbeat.epoch != token.epoch:
                raise RunLeaseConflictError(f"run {run_id!r} lease token does not match")
            if heartbeat.released_at is not None:
                return heartbeat
            if current_time < heartbeat.heartbeat_at:
                raise ValueError("lease time cannot move backwards")
            self._connection.execute(
                """
                UPDATE run_heartbeats
                SET lease_expires_at = ?, released_at = ?
                WHERE run_id = ?
                """,
                (current_time.isoformat(), current_time.isoformat(), run_id),
            )
        return heartbeat.model_copy(
            update={"lease_expires_at": current_time, "released_at": current_time}
        )

    def get_run_heartbeat(self, run_id: str) -> RunHeartbeat | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM run_heartbeats WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._heartbeat_from_row(row) if row is not None else None

    def list_run_liveness(
        self,
        *,
        stale_after_s: float = 300.0,
        now: datetime | None = None,
    ) -> tuple[RunLiveness, ...]:
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        current_time = self._aware_now(now)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT r.payload, h.owner_id, h.epoch, h.acquired_at,
                       h.heartbeat_at, h.lease_expires_at, h.released_at
                FROM runs AS r
                LEFT JOIN run_heartbeats AS h ON h.run_id = r.run_id
                ORDER BY r.started_at, r.run_id
                """
            ).fetchall()
        results = []
        for row in rows:
            run = RunManifest.model_validate_json(row["payload"])
            heartbeat = self._heartbeat_from_joined_row(run.run_id, row)
            results.append(
                self._classify_run_liveness(
                    run,
                    heartbeat,
                    stale_after_s=stale_after_s,
                    now=current_time,
                )
            )
        return tuple(results)

    def preview_run_reconciliation(
        self,
        run_id: str,
        *,
        stale_after_s: float = 300.0,
        now: datetime | None = None,
    ) -> RunReconciliationPreview:
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        current_time = self._aware_now(now)
        with self._lock:
            run, heartbeat = self._load_run_and_heartbeat(run_id)
        return self._build_reconciliation_preview(
            run,
            heartbeat,
            stale_after_s=stale_after_s,
            observed_at=current_time,
        )

    def reconcile_stale_run(
        self,
        preview: RunReconciliationPreview,
        *,
        operator_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> RunReconciliationRecord:
        normalized_operator = operator_id.strip()
        normalized_reason = reason.strip()
        if not normalized_operator:
            raise ValueError("reconciliation operator_id cannot be empty")
        if len(normalized_reason) < 8:
            raise ValueError("reconciliation reason must contain at least 8 characters")
        current_time = self._aware_now(now)
        if current_time < preview.observed_at:
            raise ValueError("reconciliation time cannot precede preview observation")
        with self._lock, self._connection:
            existing_row = self._connection.execute(
                "SELECT payload FROM run_reconciliations WHERE run_id = ?",
                (preview.run_id,),
            ).fetchone()
            if existing_row is not None:
                existing = RunReconciliationRecord.model_validate_json(existing_row["payload"])
                if (
                    existing.preview.state_digest == preview.state_digest
                    and existing.operator_id == normalized_operator
                    and existing.reason == normalized_reason
                ):
                    return existing
                raise RunReconciliationConflictError(
                    f"run {preview.run_id!r} already has another reconciliation record"
                )
            run, heartbeat = self._load_run_and_heartbeat(preview.run_id)
            current_preview = self._build_reconciliation_preview(
                run,
                heartbeat,
                stale_after_s=preview.stale_after_s,
                observed_at=current_time,
            )
            if not current_preview.eligible:
                raise RunReconciliationConflictError(
                    f"run {preview.run_id!r} is no longer eligible for stale reconciliation"
                )
            if current_preview.state_digest != preview.state_digest:
                raise RunReconciliationConflictError(
                    f"run {preview.run_id!r} changed after reconciliation preview"
                )
            prior_epoch = heartbeat.epoch if heartbeat is not None else None
            takeover_epoch = 1 if prior_epoch is None else prior_epoch + 1
            takeover_owner_id = RunReconciliationRecord.owner_id_for(normalized_operator)
            reconciliation_id = RunReconciliationRecord.expected_reconciliation_id(
                run_id=run.run_id,
                state_digest=current_preview.state_digest,
                operator_id=normalized_operator,
                reason=normalized_reason,
                takeover_epoch=takeover_epoch,
                reconciled_at=current_time,
            )
            run_payload = run.model_dump(mode="python")
            run_payload.update(
                {
                    "status": RunStatus.FAILED,
                    "completed_at": current_time,
                    "metadata": {
                        **run.metadata,
                        "reconciliation": {
                            "reconciliation_id": reconciliation_id,
                            "operator_id": normalized_operator,
                            "reason": normalized_reason,
                            "preview_state_digest": current_preview.state_digest,
                            "stale_detail": current_preview.liveness.detail,
                            "prior_lease_epoch": prior_epoch,
                            "takeover_epoch": takeover_epoch,
                        },
                    },
                }
            )
            finished = RunManifest.model_validate(run_payload)
            record = RunReconciliationRecord(
                reconciliation_id=reconciliation_id,
                run_id=run.run_id,
                preview=preview,
                operator_id=normalized_operator,
                reason=normalized_reason,
                takeover_owner_id=takeover_owner_id,
                prior_lease_epoch=prior_epoch,
                takeover_epoch=takeover_epoch,
                reconciled_at=current_time,
                terminal_run_digest=finished.digest(),
            )
            self._connection.execute(
                """
                INSERT INTO run_heartbeats(
                    run_id, owner_id, epoch, acquired_at, heartbeat_at,
                    lease_expires_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    epoch = excluded.epoch,
                    acquired_at = excluded.acquired_at,
                    heartbeat_at = excluded.heartbeat_at,
                    lease_expires_at = excluded.lease_expires_at,
                    released_at = excluded.released_at
                """,
                (
                    run.run_id,
                    takeover_owner_id,
                    takeover_epoch,
                    current_time.isoformat(),
                    current_time.isoformat(),
                    current_time.isoformat(),
                    current_time.isoformat(),
                ),
            )
            self._connection.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = ?, digest = ?, payload = ?
                WHERE run_id = ?
                """,
                (
                    finished.status.value,
                    current_time.isoformat(),
                    finished.digest(),
                    finished.canonical_bytes(),
                    run.run_id,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO run_reconciliations(
                    reconciliation_id, run_id, state_digest, operator_id,
                    reconciled_at, digest, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.reconciliation_id,
                    record.run_id,
                    record.preview.state_digest,
                    record.operator_id,
                    record.reconciled_at.isoformat(),
                    record.digest(),
                    record.canonical_bytes(),
                ),
            )
        return record

    def get_run_reconciliation(self, run_id: str) -> RunReconciliationRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM run_reconciliations WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return RunReconciliationRecord.model_validate_json(row["payload"])

    def list_run_reconciliations(self) -> tuple[RunReconciliationRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM run_reconciliations ORDER BY reconciled_at, reconciliation_id"
            ).fetchall()
        return tuple(RunReconciliationRecord.model_validate_json(row["payload"]) for row in rows)

    def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        completed_at: datetime | None = None,
        metadata: dict[str, object] | None = None,
        lease: RunLeaseToken | None = None,
    ) -> RunManifest:
        if status is RunStatus.RUNNING:
            raise ValueError("finish_run requires a terminal status")
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            current = RunManifest.model_validate_json(row["payload"])
            if current.status is not RunStatus.RUNNING:
                if current.status is status:
                    return current
                raise ValueError(f"run {run_id!r} is already {current.status.value}")
            terminal_at = completed_at or utc_now()
            self._require_lease_if_present(run_id, lease, now=terminal_at)
            payload = current.model_dump(mode="python")
            payload.update(
                {
                    "status": status,
                    "completed_at": terminal_at,
                    "metadata": {**current.metadata, **(metadata or {})},
                }
            )
            finished = RunManifest.model_validate(payload)
            self._connection.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = ?, digest = ?, payload = ?
                WHERE run_id = ?
                """,
                (
                    finished.status.value,
                    finished.completed_at.isoformat() if finished.completed_at else None,
                    finished.digest(),
                    finished.canonical_bytes(),
                    run_id,
                ),
            )
            self._connection.execute(
                """
                UPDATE run_heartbeats
                SET lease_expires_at = ?, released_at = ?
                WHERE run_id = ? AND released_at IS NULL
                """,
                (terminal_at.isoformat(), terminal_at.isoformat(), run_id),
            )
        return finished

    def put(
        self,
        trajectory: Trajectory,
        *,
        run_id: str | None = None,
        lease: RunLeaseToken | None = None,
    ) -> bool:
        with self._lock, self._connection:
            if run_id is not None:
                self._require_run(run_id, lease=lease)
            inserted = self._put_trajectory(trajectory)
            if run_id is not None:
                position = self._next_position(run_id)
                self._attach(run_id, trajectory.trajectory_id, position)
        return inserted

    def put_many(
        self,
        trajectories: Iterable[Trajectory],
        *,
        run_id: str | None = None,
        lease: RunLeaseToken | None = None,
    ) -> int:
        items = tuple(trajectories)
        inserted = 0
        with self._lock, self._connection:
            if run_id is not None:
                self._require_run(run_id, lease=lease)
                position = self._next_position(run_id)
            else:
                position = 0
            for trajectory in items:
                inserted += int(self._put_trajectory(trajectory))
                if run_id is not None:
                    self._attach(run_id, trajectory.trajectory_id, position)
                    position += 1
        return inserted

    def _put_trajectory(self, trajectory: Trajectory) -> bool:
        digest = trajectory.digest()
        existing = self._connection.execute(
            "SELECT digest FROM trajectories WHERE trajectory_id = ?",
            (trajectory.trajectory_id,),
        ).fetchone()
        if existing is not None:
            if existing["digest"] != digest:
                raise ValueError(
                    f"trajectory {trajectory.trajectory_id!r} already exists with other content"
                )
            return False
        self._connection.execute(
            """
            INSERT INTO trajectories(
                trajectory_id, task_id, group_id, policy_version, environment_version,
                origin, status, total_reward, generated_tokens, observation_tokens,
                step_count, started_at, completed_at, digest, payload, inserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trajectory.trajectory_id,
                trajectory.task_id,
                trajectory.group_id,
                trajectory.policy_version,
                trajectory.environment_version,
                trajectory.provenance.origin.value,
                trajectory.status.value,
                trajectory.total_reward,
                trajectory.total_generated_tokens,
                trajectory.total_observation_tokens,
                len(trajectory.steps),
                trajectory.started_at.isoformat(),
                trajectory.completed_at.isoformat() if trajectory.completed_at else None,
                digest,
                trajectory.canonical_bytes(),
                utc_now().isoformat(),
            ),
        )
        return True

    def _require_run(self, run_id: str, *, lease: RunLeaseToken | None = None) -> None:
        row = self._connection.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] is not None and row["status"] != RunStatus.RUNNING.value:
            raise ValueError(f"cannot attach trajectories to finished run {run_id!r}")
        self._require_lease_if_present(run_id, lease, now=utc_now())

    def _load_run_and_heartbeat(
        self,
        run_id: str,
    ) -> tuple[RunManifest, RunHeartbeat | None]:
        run_row = self._connection.execute(
            "SELECT payload FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            raise KeyError(run_id)
        heartbeat_row = self._connection.execute(
            "SELECT * FROM run_heartbeats WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return (
            RunManifest.model_validate_json(run_row["payload"]),
            self._heartbeat_from_row(heartbeat_row) if heartbeat_row is not None else None,
        )

    @staticmethod
    def _build_reconciliation_preview(
        run: RunManifest,
        heartbeat: RunHeartbeat | None,
        *,
        stale_after_s: float,
        observed_at: datetime,
    ) -> RunReconciliationPreview:
        liveness = SQLiteTrajectoryStore._classify_run_liveness(
            run,
            heartbeat,
            stale_after_s=stale_after_s,
            now=observed_at,
        )
        run_digest = run.digest()
        state_digest = RunReconciliationPreview.expected_state_digest(
            run_digest=run_digest,
            heartbeat=heartbeat,
            state=liveness.state,
            detail=liveness.detail,
            stale_after_s=stale_after_s,
        )
        return RunReconciliationPreview(
            preview_id=RunReconciliationPreview.expected_preview_id(run.run_id, state_digest),
            run_id=run.run_id,
            observed_at=observed_at,
            stale_after_s=stale_after_s,
            liveness=liveness,
            run_digest=run_digest,
            state_digest=state_digest,
            eligible=(run.status is RunStatus.RUNNING and liveness.state is RunLivenessState.STALE),
        )

    @staticmethod
    def _classify_run_liveness(
        run: RunManifest,
        heartbeat: RunHeartbeat | None,
        *,
        stale_after_s: float,
        now: datetime,
    ) -> RunLiveness:
        if run.status is not RunStatus.RUNNING:
            state = RunLivenessState.TERMINAL
            detail = run.status.value
        elif heartbeat is None:
            missing_cutoff = now - timedelta(seconds=stale_after_s)
            stale = run.started_at <= missing_cutoff
            state = RunLivenessState.STALE if stale else RunLivenessState.ACTIVE
            detail = "heartbeat_missing" if stale else "startup_grace"
        elif heartbeat.released_at is not None:
            state = RunLivenessState.STALE
            detail = "lease_released_while_running"
        elif heartbeat.lease_expires_at <= now:
            state = RunLivenessState.STALE
            detail = "lease_expired"
        else:
            state = RunLivenessState.ACTIVE
            detail = "lease_active"
        return RunLiveness(
            run_id=run.run_id,
            name=run.name,
            status=run.status,
            state=state,
            detail=detail,
            started_at=run.started_at,
            heartbeat=heartbeat,
        )

    def _require_running_run(self, run_id: str) -> None:
        row = self._connection.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] != RunStatus.RUNNING.value:
            raise RunLeaseConflictError(f"run {run_id!r} is not running")

    def _require_lease_if_present(
        self,
        run_id: str,
        token: RunLeaseToken | None,
        *,
        now: datetime,
    ) -> None:
        row = self._connection.execute(
            "SELECT * FROM run_heartbeats WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return
        if token is None:
            raise RunLeaseConflictError(f"run {run_id!r} requires a lease token")
        self._require_active_lease(run_id, token, now=now)

    def _require_active_lease(
        self,
        run_id: str,
        token: RunLeaseToken,
        *,
        now: datetime,
    ) -> RunHeartbeat:
        row = self._connection.execute(
            "SELECT * FROM run_heartbeats WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunLeaseConflictError(f"run {run_id!r} has no lease")
        heartbeat = self._heartbeat_from_row(row)
        if heartbeat.owner_id != token.owner_id or heartbeat.epoch != token.epoch:
            raise RunLeaseConflictError(f"run {run_id!r} lease token does not match")
        if heartbeat.released_at is not None:
            raise RunLeaseConflictError(f"run {run_id!r} lease has been released")
        if now < heartbeat.heartbeat_at:
            raise ValueError("lease time cannot move backwards")
        if heartbeat.lease_expires_at <= now:
            raise RunLeaseConflictError(f"run {run_id!r} lease has expired")
        return heartbeat

    @staticmethod
    def _aware_now(value: datetime | None) -> datetime:
        current = value or utc_now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("lease timestamps must be timezone-aware")
        return current

    @staticmethod
    def _heartbeat_from_row(row: sqlite3.Row) -> RunHeartbeat:
        return RunHeartbeat(
            run_id=str(row["run_id"]),
            owner_id=str(row["owner_id"]),
            epoch=int(row["epoch"]),
            acquired_at=datetime.fromisoformat(str(row["acquired_at"])),
            heartbeat_at=datetime.fromisoformat(str(row["heartbeat_at"])),
            lease_expires_at=datetime.fromisoformat(str(row["lease_expires_at"])),
            released_at=(
                datetime.fromisoformat(str(row["released_at"]))
                if row["released_at"] is not None
                else None
            ),
        )

    @staticmethod
    def _heartbeat_from_joined_row(run_id: str, row: sqlite3.Row) -> RunHeartbeat | None:
        if row["owner_id"] is None:
            return None
        return RunHeartbeat(
            run_id=run_id,
            owner_id=str(row["owner_id"]),
            epoch=int(row["epoch"]),
            acquired_at=datetime.fromisoformat(str(row["acquired_at"])),
            heartbeat_at=datetime.fromisoformat(str(row["heartbeat_at"])),
            lease_expires_at=datetime.fromisoformat(str(row["lease_expires_at"])),
            released_at=(
                datetime.fromisoformat(str(row["released_at"]))
                if row["released_at"] is not None
                else None
            ),
        )

    def _next_position(self, run_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(position) + 1, 0) AS position "
            "FROM run_trajectories WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return int(row["position"])

    def _attach(self, run_id: str, trajectory_id: str, position: int) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO run_trajectories(run_id, trajectory_id, position)
            VALUES (?, ?, ?)
            """,
            (run_id, trajectory_id, position),
        )

    def get(self, trajectory_id: str) -> Trajectory | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
        if row is None:
            return None
        return Trajectory.model_validate_json(row["payload"])

    def query(self, query: TrajectoryQuery | None = None) -> tuple[Trajectory, ...]:
        spec = query or TrajectoryQuery()
        sql, parameters = self._query_sql(spec, count=False)
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(Trajectory.model_validate_json(row["payload"]) for row in rows)

    def iter_query(self, query: TrajectoryQuery | None = None) -> Iterator[Trajectory]:
        yield from self.query(query)

    def count(self, query: TrajectoryQuery | None = None) -> int:
        sql, parameters = self._query_sql(query or TrajectoryQuery(), count=True)
        with self._lock:
            row = self._connection.execute(sql, parameters).fetchone()
        return int(row["result_count"])

    @staticmethod
    def _query_sql(query: TrajectoryQuery, *, count: bool) -> tuple[str, list[object]]:
        select = "COUNT(DISTINCT t.trajectory_id) AS result_count" if count else "t.payload"
        sql = f"SELECT {select} FROM trajectories AS t"
        parameters: list[object] = []
        clauses: list[str] = []
        if query.run_id is not None:
            sql += " JOIN run_trajectories AS rt ON rt.trajectory_id = t.trajectory_id"
            clauses.append("rt.run_id = ?")
            parameters.append(query.run_id)
        for column, value in (
            ("task_id", query.task_id),
            ("group_id", query.group_id),
            ("policy_version", query.policy_version),
            ("environment_version", query.environment_version),
        ):
            if value is not None:
                clauses.append(f"t.{column} = ?")
                parameters.append(value)
        if query.statuses:
            clauses.append(f"t.status IN ({','.join('?' for _ in query.statuses)})")
            parameters.extend(status.value for status in query.statuses)
        if query.origins:
            clauses.append(f"t.origin IN ({','.join('?' for _ in query.origins)})")
            parameters.extend(origin.value for origin in query.origins)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if not count:
            direction = "DESC" if query.newest_first else "ASC"
            sql += f" ORDER BY t.started_at {direction}, t.trajectory_id {direction}"
            if query.limit is not None:
                sql += " LIMIT ? OFFSET ?"
                parameters.extend((query.limit, query.offset))
            elif query.offset:
                sql += " LIMIT -1 OFFSET ?"
                parameters.append(query.offset)
        return sql, parameters

    def summary(self) -> StoreSummary:
        with self._lock:
            trajectory_count = int(
                self._connection.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
            )
            run_count = int(self._connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
            task_count = int(
                self._connection.execute(
                    "SELECT COUNT(DISTINCT task_id) FROM trajectories"
                ).fetchone()[0]
            )
            policies = tuple(
                row[0]
                for row in self._connection.execute(
                    "SELECT DISTINCT policy_version FROM trajectories ORDER BY policy_version"
                )
            )
            environments = tuple(
                row[0]
                for row in self._connection.execute(
                    "SELECT DISTINCT environment_version FROM trajectories "
                    "ORDER BY environment_version"
                )
            )
        return StoreSummary(
            trajectory_count=trajectory_count,
            run_count=run_count,
            task_count=task_count,
            policy_versions=policies,
            environment_versions=environments,
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteTrajectoryStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def load_trajectories_jsonl(path: Path) -> tuple[Trajectory, ...]:
    trajectories = []
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                trajectories.append(Trajectory.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid trajectory on line {line_number}: {error}") from error
    return tuple(trajectories)


def export_trajectories_jsonl(trajectories: Sequence[Trajectory], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for trajectory in trajectories:
            output.write(trajectory.canonical_bytes())
            output.write(b"\n")
    return len(trajectories)

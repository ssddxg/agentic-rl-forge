from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject, new_id, utc_now


class RunKind(str, Enum):
    ROLLOUT = "rollout"
    EVALUATION = "evaluation"
    DATASET = "dataset"
    TRAINING = "training"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunLivenessState(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    TERMINAL = "terminal"


class RunLeaseToken(ContractModel):
    owner_id: str = Field(min_length=1)
    epoch: int = Field(ge=1)


class RunHeartbeat(ContractModel):
    run_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    epoch: int = Field(ge=1)
    acquired_at: datetime
    heartbeat_at: datetime
    lease_expires_at: datetime
    released_at: datetime | None = None

    @model_validator(mode="after")
    def validate_timestamps(self) -> RunHeartbeat:
        timestamps = (self.acquired_at, self.heartbeat_at, self.lease_expires_at)
        if any(value.tzinfo is None or value.utcoffset() is None for value in timestamps):
            raise ValueError("run heartbeat timestamps must be timezone-aware")
        if self.heartbeat_at < self.acquired_at:
            raise ValueError("heartbeat_at cannot precede acquired_at")
        if self.lease_expires_at < self.heartbeat_at:
            raise ValueError("lease_expires_at cannot precede heartbeat_at")
        if self.released_at is not None:
            if self.released_at.tzinfo is None or self.released_at.utcoffset() is None:
                raise ValueError("released_at must be timezone-aware")
            if self.released_at < self.heartbeat_at:
                raise ValueError("released_at cannot precede heartbeat_at")
        return self

    @property
    def token(self) -> RunLeaseToken:
        return RunLeaseToken(owner_id=self.owner_id, epoch=self.epoch)


class RunLiveness(ContractModel):
    run_id: str
    name: str
    status: RunStatus
    state: RunLivenessState
    detail: str
    started_at: datetime
    heartbeat: RunHeartbeat | None = None


class RunReconciliationPreview(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    preview_id: str = Field(pattern=r"^reconcile_preview_[0-9a-f]{24}$")
    run_id: str = Field(min_length=1)
    observed_at: datetime
    stale_after_s: float = Field(gt=0)
    liveness: RunLiveness
    run_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible: bool

    @model_validator(mode="after")
    def validate_preview(self) -> RunReconciliationPreview:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("reconciliation preview time must be timezone-aware")
        if self.liveness.run_id != self.run_id:
            raise ValueError("reconciliation preview run identity does not match liveness")
        expected_eligible = (
            self.liveness.status is RunStatus.RUNNING
            and self.liveness.state is RunLivenessState.STALE
        )
        if self.eligible is not expected_eligible:
            raise ValueError("reconciliation eligibility does not match liveness")
        expected_digest = self.expected_state_digest(
            run_digest=self.run_digest,
            heartbeat=self.liveness.heartbeat,
            state=self.liveness.state,
            detail=self.liveness.detail,
            stale_after_s=self.stale_after_s,
        )
        if self.state_digest != expected_digest:
            raise ValueError("reconciliation state digest does not match preview contents")
        if self.preview_id != self.expected_preview_id(self.run_id, self.state_digest):
            raise ValueError("reconciliation preview ID does not match run and state")
        return self

    @staticmethod
    def expected_state_digest(
        *,
        run_digest: str,
        heartbeat: RunHeartbeat | None,
        state: RunLivenessState,
        detail: str,
        stale_after_s: float,
    ) -> str:
        payload = orjson.dumps(
            {
                "run_digest": run_digest,
                "heartbeat_digest": heartbeat.digest() if heartbeat is not None else None,
                "state": state.value,
                "detail": detail,
                "stale_after_s": float(stale_after_s),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def expected_preview_id(run_id: str, state_digest: str) -> str:
        digest = hashlib.sha256(f"{run_id}:{state_digest}".encode()).hexdigest()
        return f"reconcile_preview_{digest[:24]}"


class RunReconciliationRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    reconciliation_id: str = Field(pattern=r"^reconcile_[0-9a-f]{24}$")
    run_id: str = Field(min_length=1)
    preview: RunReconciliationPreview
    operator_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=8, max_length=2048)
    takeover_owner_id: str = Field(pattern=r"^reconciler_[0-9a-f]{16}$")
    prior_lease_epoch: int | None = Field(default=None, ge=1)
    takeover_epoch: int = Field(ge=1)
    reconciled_at: datetime
    terminal_run_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> RunReconciliationRecord:
        if self.reconciled_at.tzinfo is None or self.reconciled_at.utcoffset() is None:
            raise ValueError("reconciliation time must be timezone-aware")
        if self.preview.run_id != self.run_id or not self.preview.eligible:
            raise ValueError("reconciliation requires an eligible preview for the same run")
        expected_epoch = 1 if self.prior_lease_epoch is None else self.prior_lease_epoch + 1
        if self.takeover_epoch != expected_epoch:
            raise ValueError("reconciliation takeover epoch must follow the prior lease epoch")
        if self.takeover_owner_id != self.owner_id_for(self.operator_id):
            raise ValueError("reconciliation owner ID does not match operator")
        if self.reconciled_at < self.preview.observed_at:
            raise ValueError("reconciliation cannot precede its preview")
        if self.reconciliation_id != self.expected_reconciliation_id(
            run_id=self.run_id,
            state_digest=self.preview.state_digest,
            operator_id=self.operator_id,
            reason=self.reason,
            takeover_epoch=self.takeover_epoch,
            reconciled_at=self.reconciled_at,
        ):
            raise ValueError("reconciliation ID does not match its evidence")
        return self

    @staticmethod
    def owner_id_for(operator_id: str) -> str:
        digest = hashlib.sha256(operator_id.encode()).hexdigest()
        return f"reconciler_{digest[:16]}"

    @staticmethod
    def expected_reconciliation_id(
        *,
        run_id: str,
        state_digest: str,
        operator_id: str,
        reason: str,
        takeover_epoch: int,
        reconciled_at: datetime,
    ) -> str:
        payload = orjson.dumps(
            {
                "run_id": run_id,
                "state_digest": state_digest,
                "operator_id": operator_id,
                "reason": reason,
                "takeover_epoch": takeover_epoch,
                "reconciled_at": reconciled_at.isoformat(),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"reconcile_{hashlib.sha256(payload).hexdigest()[:24]}"


class RunManifest(ContractModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    name: str = Field(min_length=1)
    kind: RunKind
    status: RunStatus = RunStatus.RUNNING
    config: JsonObject = Field(default_factory=dict)
    seed: int = 0
    policy_version: str | None = None
    environment_version: str | None = None
    benchmark: str | None = None
    package_version: str | None = None
    git_revision: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> RunManifest:
        if self.status is RunStatus.RUNNING and self.completed_at is not None:
            raise ValueError("running runs cannot have completed_at")
        if self.status is not RunStatus.RUNNING and self.completed_at is None:
            raise ValueError("finished runs require completed_at")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self

    @property
    def config_digest(self) -> str:
        payload = orjson.dumps(self.config, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(payload).hexdigest()

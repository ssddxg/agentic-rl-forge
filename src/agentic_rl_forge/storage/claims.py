from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import Enum
from pathlib import PurePosixPath
from typing import Protocol

from agentic_rl_forge.contracts import (
    RolloutPlan,
    RolloutSlot,
    SlotClaim,
    SlotClaimOutcome,
    SlotClaimPlanStatus,
    SlotClaimRelease,
    SlotClaimRenewal,
    SlotClaimState,
    SlotClaimStatus,
    utc_now,
)
from agentic_rl_forge.storage.blobs import BlobConflictError, ConditionalBlobStore


class ClaimStoreConsistency(str, Enum):
    STRONG_READ_AFTER_WRITE_AND_LIST = "strong-read-after-write-and-list"


class SlotClaimConflictError(RuntimeError):
    pass


class SlotClaimManager(Protocol):
    def acquire(
        self,
        plan_id: str,
        slot: RolloutSlot,
        *,
        owner_id: str,
        ttl_s: float,
        now: datetime | None = None,
    ) -> SlotClaim: ...

    def assert_current(self, claim: SlotClaim, *, now: datetime | None = None) -> None: ...

    def release(
        self,
        claim: SlotClaim,
        *,
        outcome: SlotClaimOutcome,
        trajectory_id: str | None = None,
        now: datetime | None = None,
    ) -> SlotClaimRelease: ...


class RenewableSlotClaimManager(SlotClaimManager, Protocol):
    def renew(
        self,
        claim: SlotClaim,
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> SlotClaimRenewal: ...


class SlotClaimCoordinator:
    _PLAN_ID = re.compile(r"^plan_[0-9a-f]{24}$")
    _SLOT_ID = re.compile(r"^slot_[0-9a-f]{24}$")
    _CLAIM_FILE = re.compile(r"^(\d{20})\.json$")

    def __init__(
        self,
        store: ConditionalBlobStore,
        *,
        consistency: ClaimStoreConsistency,
        prefix: str = "slot-claims",
    ) -> None:
        if consistency is not ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST:
            raise ValueError("slot claims require strong read-after-write and list consistency")
        prefix_path = PurePosixPath(prefix)
        normalized = prefix.strip("/")
        if not normalized or normalized == ".":
            raise ValueError("slot claim prefix cannot be empty")
        if prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise ValueError("slot claim prefix must be a safe relative path")
        self._store = store
        self._prefix = normalized
        self._consistency = consistency

    @property
    def consistency(self) -> ClaimStoreConsistency:
        return self._consistency

    def acquire(
        self,
        plan_id: str,
        slot: RolloutSlot,
        *,
        owner_id: str,
        ttl_s: float,
        now: datetime | None = None,
    ) -> SlotClaim:
        self._validate_identity(plan_id, slot)
        if not owner_id:
            raise ValueError("claim owner_id cannot be empty")
        if ttl_s <= 0:
            raise ValueError("claim ttl_s must be positive")
        current_time = self._aware_now(now)
        latest = self.latest_claim(plan_id, slot.slot_id)
        if latest is not None:
            if current_time < latest.acquired_at:
                raise ValueError("claim time cannot move backwards")
            release = self.get_release(latest)
            effective_expiry = self.effective_expiry(latest)
            if release is None and effective_expiry > current_time:
                if latest.owner_id == owner_id:
                    return latest
                raise SlotClaimConflictError(
                    f"slot {slot.slot_id!r} is claimed by {latest.owner_id!r} "
                    f"until {effective_expiry.isoformat()}"
                )
            epoch = latest.epoch + 1
        else:
            epoch = 1
        claim = SlotClaim(
            claim_id=SlotClaim.expected_claim_id(plan_id, slot.slot_id, owner_id, epoch),
            plan_id=plan_id,
            slot_id=slot.slot_id,
            trajectory_id=slot.trajectory_id,
            owner_id=owner_id,
            epoch=epoch,
            acquired_at=current_time,
            expires_at=current_time + timedelta(seconds=ttl_s),
        )
        key = self._claim_key(plan_id, slot.slot_id, epoch)
        try:
            self._store.put_if_absent(
                key,
                claim.canonical_bytes() + b"\n",
                metadata={
                    "plan_id": plan_id,
                    "slot_id": slot.slot_id,
                    "claim_id": claim.claim_id,
                    "epoch": epoch,
                },
            )
        except BlobConflictError:
            winner = self._load_claim(key)
            self._validate_claim_key_identity(
                winner,
                plan_id=plan_id,
                slot_id=slot.slot_id,
                epoch=epoch,
            )
            if winner.trajectory_id != slot.trajectory_id:
                raise SlotClaimConflictError(
                    f"slot {slot.slot_id!r} claim has another trajectory identity"
                ) from None
            if winner.owner_id == owner_id:
                return winner
            raise SlotClaimConflictError(
                f"slot {slot.slot_id!r} claim epoch {epoch} was won by {winner.owner_id!r}"
            ) from None
        return claim

    def assert_current(self, claim: SlotClaim, *, now: datetime | None = None) -> None:
        current_time = self._aware_now(now)
        latest = self.latest_claim(claim.plan_id, claim.slot_id)
        if latest is None or latest.claim_id != claim.claim_id:
            raise SlotClaimConflictError(f"claim {claim.claim_id!r} is no longer current")
        if self.get_release(claim) is not None:
            raise SlotClaimConflictError(f"claim {claim.claim_id!r} has been released")
        if current_time < claim.acquired_at:
            raise ValueError("claim time cannot move backwards")
        if self.effective_expiry(claim) <= current_time:
            raise SlotClaimConflictError(f"claim {claim.claim_id!r} has expired")

    def renew(
        self,
        claim: SlotClaim,
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> SlotClaimRenewal:
        self._require_persisted_claim(claim)
        if ttl_s <= 0:
            raise ValueError("claim renewal ttl_s must be positive")
        current_time = self._aware_now(now)
        latest_claim = self.latest_claim(claim.plan_id, claim.slot_id)
        if latest_claim is None or latest_claim.claim_id != claim.claim_id:
            raise SlotClaimConflictError(f"claim {claim.claim_id!r} is no longer current")
        if self.get_release(claim) is not None:
            raise SlotClaimConflictError(f"claim {claim.claim_id!r} has been released")
        if current_time < claim.acquired_at:
            raise ValueError("renewal time cannot precede claim acquisition")
        latest_renewal = self.latest_renewal(claim)
        effective_expiry = (
            latest_renewal.expires_at if latest_renewal is not None else claim.expires_at
        )
        if effective_expiry <= current_time:
            raise SlotClaimConflictError(f"claim {claim.claim_id!r} has expired")
        renewed_expiry = current_time + timedelta(seconds=ttl_s)
        if renewed_expiry <= effective_expiry:
            if (
                latest_renewal is not None
                and latest_renewal.renewed_at == current_time
                and latest_renewal.expires_at == renewed_expiry
            ):
                return latest_renewal
            raise ValueError("claim renewal must extend the effective expiry")
        renewal_index = latest_renewal.renewal_index + 1 if latest_renewal is not None else 1
        renewal = SlotClaimRenewal(
            renewal_id=SlotClaimRenewal.expected_renewal_id(claim.claim_id, renewal_index),
            claim_id=claim.claim_id,
            plan_id=claim.plan_id,
            slot_id=claim.slot_id,
            owner_id=claim.owner_id,
            epoch=claim.epoch,
            renewal_index=renewal_index,
            renewed_at=current_time,
            expires_at=renewed_expiry,
        )
        key = self._renewal_key(
            claim.plan_id,
            claim.slot_id,
            claim.epoch,
            renewal_index,
        )
        try:
            self._store.put_if_absent(
                key,
                renewal.canonical_bytes() + b"\n",
                metadata={
                    "claim_id": claim.claim_id,
                    "renewal_id": renewal.renewal_id,
                    "renewal_index": renewal_index,
                },
            )
        except BlobConflictError:
            winner = self._load_renewal(key)
            self._validate_renewal_identity(
                winner,
                claim=claim,
                renewal_index=renewal_index,
                key=key,
            )
            return winner
        return renewal

    def release(
        self,
        claim: SlotClaim,
        *,
        outcome: SlotClaimOutcome,
        trajectory_id: str | None = None,
        now: datetime | None = None,
    ) -> SlotClaimRelease:
        self._require_persisted_claim(claim)
        current_time = self._aware_now(now)
        if current_time < claim.acquired_at:
            raise ValueError("release time cannot precede claim acquisition")
        if outcome is SlotClaimOutcome.COMPLETED:
            expected_trajectory_id = trajectory_id or claim.trajectory_id
            if expected_trajectory_id != claim.trajectory_id:
                raise ValueError("completed claim trajectory ID does not match its slot")
            trajectory_id = expected_trajectory_id
        release = SlotClaimRelease(
            claim_id=claim.claim_id,
            plan_id=claim.plan_id,
            slot_id=claim.slot_id,
            owner_id=claim.owner_id,
            epoch=claim.epoch,
            outcome=outcome,
            released_at=current_time,
            trajectory_id=trajectory_id,
        )
        key = self._release_key(claim.plan_id, claim.slot_id, claim.epoch)
        existing = self._store.head(key)
        if existing is not None:
            return self._validate_existing_release(claim, release, key)
        try:
            self._store.put_if_absent(
                key,
                release.canonical_bytes() + b"\n",
                metadata={
                    "claim_id": claim.claim_id,
                    "outcome": outcome.value,
                    "trajectory_id": trajectory_id or "",
                },
            )
        except BlobConflictError:
            return self._validate_existing_release(claim, release, key)
        return release

    def latest_claim(self, plan_id: str, slot_id: str) -> SlotClaim | None:
        self._validate_ids(plan_id, slot_id)
        prefix = self._claims_prefix(plan_id, slot_id)
        candidates = []
        for key in self._store.list(prefix):
            name = key.rsplit("/", 1)[-1]
            match = self._CLAIM_FILE.fullmatch(name)
            if match is not None:
                candidates.append((int(match.group(1)), key))
        if not candidates:
            return None
        epoch, key = max(candidates)
        claim = self._load_claim(key)
        self._validate_claim_key_identity(
            claim,
            plan_id=plan_id,
            slot_id=slot_id,
            epoch=epoch,
            key=key,
        )
        return claim

    def get_release(self, claim: SlotClaim) -> SlotClaimRelease | None:
        key = self._release_key(claim.plan_id, claim.slot_id, claim.epoch)
        if self._store.head(key) is None:
            return None
        release = SlotClaimRelease.model_validate_json(self._store.get(key))
        if (
            release.claim_id != claim.claim_id
            or release.plan_id != claim.plan_id
            or release.slot_id != claim.slot_id
            or release.owner_id != claim.owner_id
            or release.epoch != claim.epoch
        ):
            raise SlotClaimConflictError(f"release object {key!r} has mismatched claim identity")
        return release

    def latest_renewal(self, claim: SlotClaim) -> SlotClaimRenewal | None:
        self._require_persisted_claim(claim)
        prefix = self._renewals_prefix(claim.plan_id, claim.slot_id, claim.epoch)
        candidates = []
        for key in self._store.list(prefix):
            name = key.rsplit("/", 1)[-1]
            match = self._CLAIM_FILE.fullmatch(name)
            if match is not None:
                candidates.append((int(match.group(1)), key))
        if not candidates:
            return None
        latest: SlotClaimRenewal | None = None
        effective_expiry = claim.expires_at
        previous_renewed_at = claim.acquired_at
        for expected_index, (renewal_index, key) in enumerate(sorted(candidates), 1):
            if renewal_index != expected_index:
                raise SlotClaimConflictError(
                    f"claim {claim.claim_id!r} has a non-contiguous renewal history"
                )
            renewal = self._load_renewal(key)
            self._validate_renewal_identity(
                renewal,
                claim=claim,
                renewal_index=renewal_index,
                key=key,
            )
            if renewal.renewed_at < previous_renewed_at or renewal.expires_at <= effective_expiry:
                raise SlotClaimConflictError(
                    f"renewal object {key!r} does not extend the prior renewal"
                )
            previous_renewed_at = renewal.renewed_at
            effective_expiry = renewal.expires_at
            latest = renewal
        return latest

    def effective_expiry(self, claim: SlotClaim) -> datetime:
        renewal = self.latest_renewal(claim)
        return renewal.expires_at if renewal is not None else claim.expires_at

    def inspect_plan(
        self,
        plan: RolloutPlan,
        *,
        now: datetime | None = None,
    ) -> SlotClaimPlanStatus:
        observed_at = self._aware_now(now)
        statuses: list[SlotClaimStatus] = []
        for slot in plan.slots:
            claim = self.latest_claim(plan.plan_id, slot.slot_id)
            if claim is None:
                statuses.append(
                    SlotClaimStatus(
                        slot_id=slot.slot_id,
                        trajectory_id=slot.trajectory_id,
                        state=SlotClaimState.UNCLAIMED,
                    )
                )
                continue
            if claim.trajectory_id != slot.trajectory_id:
                raise SlotClaimConflictError(
                    f"claim {claim.claim_id!r} trajectory does not match rollout plan"
                )
            if observed_at < claim.acquired_at:
                raise ValueError("claim status time cannot precede claim acquisition")
            release = self.get_release(claim)
            renewal = self.latest_renewal(claim)
            if release is not None:
                if observed_at < release.released_at:
                    raise ValueError("claim status time cannot precede claim release")
                state = (
                    SlotClaimState.COMPLETED
                    if release.outcome is SlotClaimOutcome.COMPLETED
                    else SlotClaimState.ABANDONED
                )
            elif (renewal.expires_at if renewal is not None else claim.expires_at) <= observed_at:
                state = SlotClaimState.EXPIRED
            else:
                state = SlotClaimState.ACTIVE
            statuses.append(
                SlotClaimStatus(
                    slot_id=slot.slot_id,
                    trajectory_id=slot.trajectory_id,
                    state=state,
                    claim=claim,
                    renewal=renewal,
                    release=release,
                )
            )
        status_tuple = tuple(statuses)
        return SlotClaimPlanStatus(
            plan_id=plan.plan_id,
            observed_at=observed_at,
            slots=status_tuple,
            state_counts={
                state: sum(item.state is state for item in status_tuple) for state in SlotClaimState
            },
        )

    def _validate_existing_release(
        self,
        claim: SlotClaim,
        requested: SlotClaimRelease,
        key: str,
    ) -> SlotClaimRelease:
        existing = SlotClaimRelease.model_validate_json(self._store.get(key))
        if (
            existing.claim_id != claim.claim_id
            or existing.plan_id != claim.plan_id
            or existing.slot_id != claim.slot_id
            or existing.owner_id != claim.owner_id
            or existing.epoch != claim.epoch
            or existing.outcome is not requested.outcome
            or existing.trajectory_id != requested.trajectory_id
        ):
            raise SlotClaimConflictError(f"claim {claim.claim_id!r} has another release record")
        return existing

    def _load_claim(self, key: str) -> SlotClaim:
        return SlotClaim.model_validate_json(self._store.get(key))

    def _load_renewal(self, key: str) -> SlotClaimRenewal:
        return SlotClaimRenewal.model_validate_json(self._store.get(key))

    def _require_persisted_claim(self, claim: SlotClaim) -> None:
        key = self._claim_key(claim.plan_id, claim.slot_id, claim.epoch)
        try:
            persisted = self._load_claim(key)
        except KeyError:
            raise SlotClaimConflictError(
                f"claim {claim.claim_id!r} has no persisted claim record"
            ) from None
        if persisted != claim:
            raise SlotClaimConflictError(f"claim object {key!r} has mismatched content")

    @staticmethod
    def _validate_claim_key_identity(
        claim: SlotClaim,
        *,
        plan_id: str,
        slot_id: str,
        epoch: int,
        key: str | None = None,
    ) -> None:
        if claim.plan_id != plan_id or claim.slot_id != slot_id or claim.epoch != epoch:
            location = key or f"claim epoch {epoch}"
            raise SlotClaimConflictError(f"claim object {location!r} has mismatched identity")

    @staticmethod
    def _validate_renewal_identity(
        renewal: SlotClaimRenewal,
        *,
        claim: SlotClaim,
        renewal_index: int,
        key: str,
    ) -> None:
        if (
            renewal.claim_id != claim.claim_id
            or renewal.plan_id != claim.plan_id
            or renewal.slot_id != claim.slot_id
            or renewal.owner_id != claim.owner_id
            or renewal.epoch != claim.epoch
            or renewal.renewal_index != renewal_index
            or renewal.renewed_at < claim.acquired_at
            or renewal.expires_at <= claim.expires_at
        ):
            raise SlotClaimConflictError(f"renewal object {key!r} has mismatched claim identity")

    def _claims_prefix(self, plan_id: str, slot_id: str) -> str:
        return f"{self._slot_prefix(plan_id, slot_id)}/claims/"

    def _claim_key(self, plan_id: str, slot_id: str, epoch: int) -> str:
        return f"{self._claims_prefix(plan_id, slot_id)}{epoch:020d}.json"

    def _release_key(self, plan_id: str, slot_id: str, epoch: int) -> str:
        return f"{self._slot_prefix(plan_id, slot_id)}/releases/{epoch:020d}.json"

    def _renewals_prefix(self, plan_id: str, slot_id: str, epoch: int) -> str:
        return f"{self._slot_prefix(plan_id, slot_id)}/renewals/{epoch:020d}/"

    def _renewal_key(
        self,
        plan_id: str,
        slot_id: str,
        epoch: int,
        renewal_index: int,
    ) -> str:
        return f"{self._renewals_prefix(plan_id, slot_id, epoch)}{renewal_index:020d}.json"

    def _slot_prefix(self, plan_id: str, slot_id: str) -> str:
        return f"{self._prefix}/plans/{plan_id}/slots/{slot_id}"

    @classmethod
    def _validate_identity(cls, plan_id: str, slot: RolloutSlot) -> None:
        cls._validate_ids(plan_id, slot.slot_id)
        if not slot.trajectory_id:
            raise ValueError("slot identity is incomplete")

    @classmethod
    def _validate_ids(cls, plan_id: str, slot_id: str) -> None:
        if cls._PLAN_ID.fullmatch(plan_id) is None:
            raise ValueError("plan_id has an invalid format")
        if cls._SLOT_ID.fullmatch(slot_id) is None:
            raise ValueError("slot_id has an invalid format")

    @staticmethod
    def _aware_now(value: datetime | None) -> datetime:
        current = value or utc_now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("claim timestamps must be timezone-aware")
        return current

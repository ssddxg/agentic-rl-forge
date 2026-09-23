from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel


class SlotClaimOutcome(str, Enum):
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class SlotClaimState(str, Enum):
    UNCLAIMED = "unclaimed"
    ACTIVE = "active"
    EXPIRED = "expired"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class SlotClaim(ContractModel):
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{24}$")
    slot_id: str = Field(pattern=r"^slot_[0-9a-f]{24}$")
    trajectory_id: str = Field(pattern=r"^traj_[0-9a-f]{32}$")
    owner_id: str = Field(min_length=1)
    epoch: int = Field(ge=1)
    acquired_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def validate_claim(self) -> SlotClaim:
        if self.acquired_at.tzinfo is None or self.acquired_at.utcoffset() is None:
            raise ValueError("claim timestamps must be timezone-aware")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("claim timestamps must be timezone-aware")
        if self.expires_at <= self.acquired_at:
            raise ValueError("claim expiry must follow acquisition")
        if self.claim_id != self.expected_claim_id(
            self.plan_id,
            self.slot_id,
            self.owner_id,
            self.epoch,
        ):
            raise ValueError("claim ID does not match plan, slot, owner, and epoch")
        return self

    @staticmethod
    def expected_claim_id(plan_id: str, slot_id: str, owner_id: str, epoch: int) -> str:
        digest = hashlib.sha256(f"{plan_id}:{slot_id}:{owner_id}:{epoch}".encode()).hexdigest()
        return f"claim_{digest[:24]}"


class SlotClaimRelease(ContractModel):
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{24}$")
    slot_id: str = Field(pattern=r"^slot_[0-9a-f]{24}$")
    owner_id: str = Field(min_length=1)
    epoch: int = Field(ge=1)
    outcome: SlotClaimOutcome
    released_at: datetime
    trajectory_id: str | None = Field(default=None, pattern=r"^traj_[0-9a-f]{32}$")

    @model_validator(mode="after")
    def validate_release(self) -> SlotClaimRelease:
        if self.released_at.tzinfo is None or self.released_at.utcoffset() is None:
            raise ValueError("release timestamp must be timezone-aware")
        if self.outcome is SlotClaimOutcome.COMPLETED and self.trajectory_id is None:
            raise ValueError("completed claims require a trajectory ID")
        if self.outcome is SlotClaimOutcome.ABANDONED and self.trajectory_id is not None:
            raise ValueError("abandoned claims cannot contain a trajectory ID")
        return self


class SlotClaimRenewal(ContractModel):
    renewal_id: str = Field(pattern=r"^renewal_[0-9a-f]{24}$")
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{24}$")
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{24}$")
    slot_id: str = Field(pattern=r"^slot_[0-9a-f]{24}$")
    owner_id: str = Field(min_length=1)
    epoch: int = Field(ge=1)
    renewal_index: int = Field(ge=1)
    renewed_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def validate_renewal(self) -> SlotClaimRenewal:
        if self.renewed_at.tzinfo is None or self.renewed_at.utcoffset() is None:
            raise ValueError("renewal timestamps must be timezone-aware")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("renewal timestamps must be timezone-aware")
        if self.expires_at <= self.renewed_at:
            raise ValueError("renewal expiry must follow renewal time")
        if self.renewal_id != self.expected_renewal_id(self.claim_id, self.renewal_index):
            raise ValueError("renewal ID does not match claim and renewal index")
        return self

    @staticmethod
    def expected_renewal_id(claim_id: str, renewal_index: int) -> str:
        digest = hashlib.sha256(f"{claim_id}:renewal:{renewal_index}".encode()).hexdigest()
        return f"renewal_{digest[:24]}"


class SlotClaimStatus(ContractModel):
    slot_id: str = Field(pattern=r"^slot_[0-9a-f]{24}$")
    trajectory_id: str = Field(pattern=r"^traj_[0-9a-f]{32}$")
    state: SlotClaimState
    claim: SlotClaim | None = None
    renewal: SlotClaimRenewal | None = None
    release: SlotClaimRelease | None = None

    @model_validator(mode="after")
    def validate_status(self) -> SlotClaimStatus:
        if self.claim is None:
            if (
                self.state is not SlotClaimState.UNCLAIMED
                or self.renewal is not None
                or self.release is not None
            ):
                raise ValueError("only unclaimed slots can omit claim details")
            return self
        if self.claim.slot_id != self.slot_id or self.claim.trajectory_id != self.trajectory_id:
            raise ValueError("claim identity does not match status slot")
        if self.renewal is not None:
            if (
                self.renewal.claim_id != self.claim.claim_id
                or self.renewal.plan_id != self.claim.plan_id
                or self.renewal.slot_id != self.claim.slot_id
                or self.renewal.owner_id != self.claim.owner_id
                or self.renewal.epoch != self.claim.epoch
            ):
                raise ValueError("renewal identity does not match status claim")
            if self.renewal.renewed_at < self.claim.acquired_at:
                raise ValueError("claim renewal cannot precede acquisition")
            if self.renewal.expires_at <= self.claim.expires_at:
                raise ValueError("claim renewal must extend the original expiry")
        if self.release is None:
            if self.state not in {SlotClaimState.ACTIVE, SlotClaimState.EXPIRED}:
                raise ValueError("unreleased claims must be active or expired")
            return self
        if (
            self.release.claim_id != self.claim.claim_id
            or self.release.slot_id != self.slot_id
            or self.release.plan_id != self.claim.plan_id
            or self.release.owner_id != self.claim.owner_id
            or self.release.epoch != self.claim.epoch
        ):
            raise ValueError("release identity does not match status claim")
        if (
            self.release.outcome is SlotClaimOutcome.COMPLETED
            and self.release.trajectory_id != self.trajectory_id
        ):
            raise ValueError("completed release trajectory does not match status slot")
        expected_state = (
            SlotClaimState.COMPLETED
            if self.release.outcome is SlotClaimOutcome.COMPLETED
            else SlotClaimState.ABANDONED
        )
        if self.state is not expected_state:
            raise ValueError("released claim state does not match its outcome")
        return self


class SlotClaimPlanStatus(ContractModel):
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{24}$")
    observed_at: datetime
    slots: tuple[SlotClaimStatus, ...]
    state_counts: dict[SlotClaimState, int]

    @model_validator(mode="after")
    def validate_report(self) -> SlotClaimPlanStatus:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("claim status observation time must be timezone-aware")
        if len({item.slot_id for item in self.slots}) != len(self.slots):
            raise ValueError("claim status slots must be unique")
        expected_counts = {
            state: sum(item.state is state for item in self.slots) for state in SlotClaimState
        }
        if self.state_counts != expected_counts:
            raise ValueError("claim status counts do not match slots")
        if any(
            item.claim is not None and item.claim.plan_id != self.plan_id for item in self.slots
        ):
            raise ValueError("claim status report mixes rollout plans")
        for item in self.slots:
            if item.claim is None:
                continue
            if item.renewal is not None and self.observed_at < item.renewal.renewed_at:
                raise ValueError("claim status time cannot precede claim renewal")
            if item.release is not None:
                continue
            expires_at = (
                item.renewal.expires_at if item.renewal is not None else item.claim.expires_at
            )
            expected_state = (
                SlotClaimState.ACTIVE if expires_at > self.observed_at else SlotClaimState.EXPIRED
            )
            if item.state is not expected_state:
                raise ValueError("claim status state does not match effective expiry")
        return self

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    Message,
    MessageRole,
    RolloutSlot,
    SlotClaim,
    SlotClaimOutcome,
    SlotClaimRelease,
    SlotClaimRenewal,
    TaskSpec,
    ToolSpec,
    Trajectory,
    VerifierSpec,
    utc_now,
)
from agentic_rl_forge.environments import LocalToolEnvironment
from agentic_rl_forge.rewards import ExactMatchOutcome, RewardEngine
from agentic_rl_forge.rollout import (
    AgentLoop,
    GenerationRequest,
    PolicyOutput,
    RolloutBatch,
    RolloutPlanBuilder,
    RolloutScheduler,
)
from agentic_rl_forge.storage import (
    ClaimStoreConsistency,
    LocalBlobStore,
    SlotClaimCoordinator,
)


class CountingPolicy:
    version = "planned-policy-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        request: GenerationRequest,
    ) -> PolicyOutput:
        del messages, tools, request
        self.calls += 1
        return PolicyOutput(
            action=AgentAction(kind=ActionKind.FINAL, final_answer="yes"),
            generated_token_count=1,
        )


class SlowPolicy:
    version = "slow-policy-v1"

    def __init__(self) -> None:
        self.cancelled = False

    async def generate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        request: GenerationRequest,
    ) -> PolicyOutput:
        del messages, tools, request
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return PolicyOutput(
            action=AgentAction(kind=ActionKind.FINAL, final_answer="yes"),
            generated_token_count=1,
        )


class FailingRenewalCoordinator(SlotClaimCoordinator):
    def renew(
        self,
        claim: SlotClaim,
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> SlotClaimRenewal:
        del claim, ttl_s, now
        raise RuntimeError("simulated claim renewal failure")


class AgedClaimManager:
    def __init__(self, *, acquisition_age_s: float) -> None:
        self.acquisition_age_s = acquisition_age_s
        self.claim: SlotClaim | None = None
        self.renewals: list[SlotClaimRenewal] = []
        self.releases: list[SlotClaimRelease] = []
        self.renewed = Event()

    def acquire(
        self,
        plan_id: str,
        slot: RolloutSlot,
        *,
        owner_id: str,
        ttl_s: float,
        now: datetime | None = None,
    ) -> SlotClaim:
        acquired_at = (now or utc_now()) - timedelta(seconds=self.acquisition_age_s)
        self.claim = SlotClaim(
            claim_id=SlotClaim.expected_claim_id(plan_id, slot.slot_id, owner_id, 1),
            plan_id=plan_id,
            slot_id=slot.slot_id,
            trajectory_id=slot.trajectory_id,
            owner_id=owner_id,
            epoch=1,
            acquired_at=acquired_at,
            expires_at=acquired_at + timedelta(seconds=ttl_s),
        )
        return self.claim

    def assert_current(self, claim: SlotClaim, *, now: datetime | None = None) -> None:
        assert self.claim == claim
        assert not self.releases
        effective_expiry = self.renewals[-1].expires_at if self.renewals else claim.expires_at
        assert effective_expiry > (now or utc_now())

    def renew(
        self,
        claim: SlotClaim,
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> SlotClaimRenewal:
        assert self.claim == claim
        renewed_at = now or utc_now()
        renewal_index = len(self.renewals) + 1
        renewal = SlotClaimRenewal(
            renewal_id=SlotClaimRenewal.expected_renewal_id(claim.claim_id, renewal_index),
            claim_id=claim.claim_id,
            plan_id=claim.plan_id,
            slot_id=claim.slot_id,
            owner_id=claim.owner_id,
            epoch=claim.epoch,
            renewal_index=renewal_index,
            renewed_at=renewed_at,
            expires_at=renewed_at + timedelta(seconds=ttl_s),
        )
        self.renewals.append(renewal)
        self.renewed.set()
        return renewal

    def release(
        self,
        claim: SlotClaim,
        *,
        outcome: SlotClaimOutcome,
        trajectory_id: str | None = None,
        now: datetime | None = None,
    ) -> SlotClaimRelease:
        assert self.claim == claim
        release = SlotClaimRelease(
            claim_id=claim.claim_id,
            plan_id=claim.plan_id,
            slot_id=claim.slot_id,
            owner_id=claim.owner_id,
            epoch=claim.epoch,
            outcome=outcome,
            released_at=now or utc_now(),
            trajectory_id=trajectory_id,
        )
        self.releases.append(release)
        return release


class RenewalSignalPolicy:
    version = "renewal-signal-policy-v1"

    def __init__(self, renewed: Event) -> None:
        self.renewed = renewed

    async def generate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        request: GenerationRequest,
    ) -> PolicyOutput:
        del messages, tools, request
        if not await asyncio.to_thread(self.renewed.wait, 0.1):
            raise RuntimeError("overdue claim renewal did not start promptly")
        return PolicyOutput(
            action=AgentAction(kind=ActionKind.FINAL, final_answer="yes"),
            generated_token_count=1,
        )


class PartialPersistenceCallback:
    def __init__(self) -> None:
        self.saved: list[Trajectory] = []

    async def on_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.provenance.metadata["rollout_index"] == 1:
            raise RuntimeError("simulated persistence failure")
        self.saved.append(trajectory)

    async def on_batch(self, batch: RolloutBatch) -> None:
        del batch


class RecordingCallback:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def on_trajectory(self, trajectory: Trajectory) -> None:
        self.seen.append(trajectory.trajectory_id)

    async def on_batch(self, batch: RolloutBatch) -> None:
        del batch


def planned_task() -> TaskSpec:
    return TaskSpec(
        task_id="planned-task",
        messages=(Message(role=MessageRole.USER, content="Return yes."),),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "yes"}),
    )


def loop_factory(policy: CountingPolicy) -> AgentLoop:
    return AgentLoop(
        policy=policy,
        environment=LocalToolEnvironment(()),
        rewards=RewardEngine((ExactMatchOutcome(),)),
    )


async def test_rollout_plan_resumes_only_missing_slots_after_callback_failure(
    tmp_path: Path,
) -> None:
    task = planned_task()
    plan = RolloutPlanBuilder().build(
        (task,),
        policy_version=CountingPolicy.version,
        source_sha256="0" * 64,
        config_digest="1" * 64,
        rollouts_per_task=2,
        seed=10,
    )
    assert plan == RolloutPlanBuilder().build(
        (task,),
        policy_version=CountingPolicy.version,
        source_sha256="0" * 64,
        config_digest="1" * 64,
        rollouts_per_task=2,
        seed=10,
    )
    first_policy = CountingPolicy()
    partial = PartialPersistenceCallback()
    claims = SlotClaimCoordinator(
        LocalBlobStore(tmp_path / "claims"),
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )

    try:
        await RolloutScheduler(
            lambda: loop_factory(first_policy),
            max_concurrency=1,
            callbacks=(partial,),
            slot_claims=claims,
            claim_owner_id="first-worker",
        ).collect(
            (task,),
            rollouts_per_task=2,
            plan=plan,
        )
    except RuntimeError as error:
        assert str(error) == "simulated persistence failure"
    else:
        raise AssertionError("collection should fail after the first persisted slot")

    assert len(partial.saved) == 1
    assert first_policy.calls == 2
    resumed_policy = CountingPolicy()
    recording = RecordingCallback()
    batch = await RolloutScheduler(
        lambda: loop_factory(resumed_policy),
        max_concurrency=1,
        callbacks=(recording,),
        slot_claims=claims,
        claim_owner_id="second-worker",
    ).collect(
        (task,),
        rollouts_per_task=2,
        plan=plan,
        existing_trajectories=tuple(partial.saved),
    )

    assert resumed_policy.calls == 1
    assert tuple(item.trajectory_id for item in batch.trajectories) == tuple(
        slot.trajectory_id for slot in plan.slots
    )
    assert recording.seen == [slot.trajectory_id for slot in plan.slots]
    resumed_claim = claims.latest_claim(plan.plan_id, plan.slots[1].slot_id)
    assert resumed_claim is not None
    assert resumed_claim.epoch == 2
    resumed_release = claims.get_release(resumed_claim)
    assert resumed_release is not None
    assert resumed_release.outcome is SlotClaimOutcome.COMPLETED

    bad_provenance = partial.saved[0].provenance.model_copy(
        update={"metadata": {"rollout_plan_id": "plan_wrong"}}
    )
    corrupted = partial.saved[0].model_copy(update={"provenance": bad_provenance})
    try:
        await RolloutScheduler(lambda: loop_factory(CountingPolicy())).collect(
            (task,),
            rollouts_per_task=2,
            plan=plan,
            existing_trajectories=(corrupted,),
        )
    except ValueError as error:
        assert "does not match its rollout slot" in str(error)
    else:
        raise AssertionError("corrupted slot metadata should be rejected")


async def test_slot_claim_renewal_failure_cancels_rollout_before_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyAsyncioTimeoutError(Exception):
        pass

    async def raise_legacy_asyncio_timeout(
        awaitable: object,
        timeout: float | None = None,
    ) -> object:
        del timeout
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise LegacyAsyncioTimeoutError

    monkeypatch.setattr(asyncio, "TimeoutError", LegacyAsyncioTimeoutError)
    monkeypatch.setattr(asyncio, "wait_for", raise_legacy_asyncio_timeout)
    task = planned_task()
    plan = RolloutPlanBuilder().build(
        (task,),
        policy_version=SlowPolicy.version,
        source_sha256="0" * 64,
        config_digest="1" * 64,
        rollouts_per_task=1,
        seed=10,
    )
    policy = SlowPolicy()
    recording = RecordingCallback()
    claims = FailingRenewalCoordinator(
        LocalBlobStore(tmp_path / "claims"),
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )

    with pytest.raises(RuntimeError, match="simulated claim renewal failure"):
        await RolloutScheduler(
            lambda: AgentLoop(
                policy=policy,
                environment=LocalToolEnvironment(()),
                rewards=RewardEngine((ExactMatchOutcome(),)),
            ),
            callbacks=(recording,),
            slot_claims=claims,
            claim_owner_id="failing-worker",
            claim_ttl_s=0.1,
            claim_renewal_interval_s=0.01,
        ).collect(
            (task,),
            rollouts_per_task=1,
            plan=plan,
        )

    assert policy.cancelled
    assert recording.seen == []
    claim = claims.latest_claim(plan.plan_id, plan.slots[0].slot_id)
    assert claim is not None
    release = claims.get_release(claim)
    assert release is not None
    assert release.outcome is SlotClaimOutcome.ABANDONED


async def test_slot_claim_renewal_uses_persisted_acquisition_deadline() -> None:
    task = planned_task()
    plan = RolloutPlanBuilder().build(
        (task,),
        policy_version=RenewalSignalPolicy.version,
        source_sha256="0" * 64,
        config_digest="1" * 64,
        rollouts_per_task=1,
        seed=10,
    )
    claims = AgedClaimManager(acquisition_age_s=0.5)
    policy = RenewalSignalPolicy(claims.renewed)

    batch = await RolloutScheduler(
        lambda: AgentLoop(
            policy=policy,
            environment=LocalToolEnvironment(()),
            rewards=RewardEngine((ExactMatchOutcome(),)),
        ),
        slot_claims=claims,
        claim_owner_id="delayed-acquisition-worker",
        claim_ttl_s=2.0,
        claim_renewal_interval_s=0.2,
    ).collect(
        (task,),
        rollouts_per_task=1,
        plan=plan,
    )

    assert len(batch.trajectories) == 1
    assert claims.renewals
    assert claims.releases[0].outcome is SlotClaimOutcome.COMPLETED

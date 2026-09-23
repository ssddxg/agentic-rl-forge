from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest

from agentic_rl_forge.contracts import (
    BlobPutResult,
    JsonObject,
    Message,
    MessageRole,
    RolloutPlan,
    SlotClaim,
    SlotClaimOutcome,
    SlotClaimState,
    TaskSpec,
    VerifierSpec,
)
from agentic_rl_forge.rollout import RolloutPlanBuilder
from agentic_rl_forge.storage import (
    ClaimStoreConsistency,
    LocalBlobStore,
    SlotClaimConflictError,
    SlotClaimCoordinator,
)


class ContendedLocalBlobStore(LocalBlobStore):
    def __init__(self, root: Path, *, segment: str = "/claims/") -> None:
        super().__init__(root)
        self._contended_segment = segment
        self._write_barrier = Barrier(2)

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: JsonObject | None = None,
    ) -> BlobPutResult:
        if self._contended_segment in key:
            self._write_barrier.wait(timeout=2)
        return super().put_if_absent(key, data, metadata=metadata)


def build_plan(*, rollouts_per_task: int = 1) -> RolloutPlan:
    task = TaskSpec(
        task_id="claim-task",
        messages=(Message(role=MessageRole.USER, content="Return yes."),),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "yes"}),
    )
    return RolloutPlanBuilder().build(
        (task,),
        policy_version="claim-policy-v1",
        source_sha256="0" * 64,
        config_digest="1" * 64,
        rollouts_per_task=rollouts_per_task,
        seed=7,
    )


def test_concurrent_slot_claim_has_one_winner(tmp_path: Path) -> None:
    coordinator = SlotClaimCoordinator(
        ContendedLocalBlobStore(tmp_path),
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )
    plan = build_plan()
    slot = plan.slots[0]

    def acquire(owner_id: str) -> SlotClaim | SlotClaimConflictError:
        try:
            return coordinator.acquire(
                plan.plan_id,
                slot,
                owner_id=owner_id,
                ttl_s=30,
                now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        except SlotClaimConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(acquire, ("worker-a", "worker-b")))

    winners = tuple(result for result in results if isinstance(result, SlotClaim))
    conflicts = tuple(result for result in results if isinstance(result, SlotClaimConflictError))
    assert len(winners) == 1
    assert len(conflicts) == 1
    assert coordinator.latest_claim(plan.plan_id, slot.slot_id) == winners[0]


def test_slot_claims_fence_competitors_and_allow_released_or_expired_takeover(
    tmp_path: Path,
) -> None:
    coordinator = SlotClaimCoordinator(
        LocalBlobStore(tmp_path),
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )
    plan = build_plan()
    slot = plan.slots[0]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    first = coordinator.acquire(
        plan.plan_id,
        slot,
        owner_id="worker-a",
        ttl_s=30,
        now=start,
    )
    assert (
        coordinator.acquire(
            plan.plan_id,
            slot,
            owner_id="worker-a",
            ttl_s=30,
            now=start + timedelta(seconds=1),
        )
        == first
    )
    with pytest.raises(SlotClaimConflictError, match="is claimed by"):
        coordinator.acquire(
            plan.plan_id,
            slot,
            owner_id="worker-b",
            ttl_s=30,
            now=start + timedelta(seconds=1),
        )
    coordinator.assert_current(first, now=start + timedelta(seconds=1))

    release = coordinator.release(
        first,
        outcome=SlotClaimOutcome.COMPLETED,
        trajectory_id=slot.trajectory_id,
        now=start + timedelta(seconds=2),
    )
    assert (
        coordinator.release(
            first,
            outcome=SlotClaimOutcome.COMPLETED,
            trajectory_id=slot.trajectory_id,
            now=start + timedelta(seconds=3),
        )
        == release
    )
    with pytest.raises(SlotClaimConflictError, match="another release record"):
        coordinator.release(
            first,
            outcome=SlotClaimOutcome.ABANDONED,
            now=start + timedelta(seconds=3),
        )

    second = coordinator.acquire(
        plan.plan_id,
        slot,
        owner_id="worker-b",
        ttl_s=10,
        now=start + timedelta(seconds=4),
    )
    assert second.epoch == 2
    with pytest.raises(SlotClaimConflictError, match="no longer current"):
        coordinator.assert_current(first, now=start + timedelta(seconds=5))

    third = coordinator.acquire(
        plan.plan_id,
        slot,
        owner_id="worker-c",
        ttl_s=10,
        now=start + timedelta(seconds=15),
    )
    assert third.epoch == 3
    assert coordinator.latest_claim(plan.plan_id, slot.slot_id) == third


def test_slot_claim_renewal_extends_effective_expiry_and_cannot_revive_old_epoch(
    tmp_path: Path,
) -> None:
    coordinator = SlotClaimCoordinator(
        LocalBlobStore(tmp_path),
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )
    plan = build_plan()
    slot = plan.slots[0]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = coordinator.acquire(
        plan.plan_id,
        slot,
        owner_id="worker-a",
        ttl_s=10,
        now=start,
    )

    renewal = coordinator.renew(
        first,
        ttl_s=10,
        now=start + timedelta(seconds=5),
    )
    assert renewal.renewal_index == 1
    assert renewal.expires_at == start + timedelta(seconds=15)
    assert (
        coordinator.renew(
            first,
            ttl_s=10,
            now=start + timedelta(seconds=5),
        )
        == renewal
    )
    coordinator.assert_current(first, now=start + timedelta(seconds=11))
    with pytest.raises(SlotClaimConflictError, match="is claimed by"):
        coordinator.acquire(
            plan.plan_id,
            slot,
            owner_id="worker-b",
            ttl_s=10,
            now=start + timedelta(seconds=11),
        )

    second = coordinator.acquire(
        plan.plan_id,
        slot,
        owner_id="worker-b",
        ttl_s=10,
        now=start + timedelta(seconds=15),
    )
    assert second.epoch == 2
    with pytest.raises(SlotClaimConflictError, match="no longer current"):
        coordinator.renew(
            first,
            ttl_s=10,
            now=start + timedelta(seconds=16),
        )


def test_concurrent_slot_claim_renewal_has_one_immutable_winner(tmp_path: Path) -> None:
    store = ContendedLocalBlobStore(tmp_path, segment="/renewals/")
    coordinator = SlotClaimCoordinator(
        store,
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )
    plan = build_plan()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    claim = coordinator.acquire(
        plan.plan_id,
        plan.slots[0],
        owner_id="worker-a",
        ttl_s=30,
        now=start,
    )

    def renew(offset_s: int):
        return coordinator.renew(
            claim,
            ttl_s=30,
            now=start + timedelta(seconds=offset_s),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        renewals = tuple(executor.map(renew, (5, 6)))

    assert renewals[0] == renewals[1]
    assert renewals[0].renewal_index == 1
    assert coordinator.latest_renewal(claim) == renewals[0]
    renewal_files = tuple(tmp_path.rglob("renewals/**/*.json"))
    assert len(renewal_files) == 1


def test_slot_claims_require_explicit_strong_consistency(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strong read-after-write"):
        SlotClaimCoordinator(
            LocalBlobStore(tmp_path),
            consistency="eventual",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="safe relative path"):
        SlotClaimCoordinator(
            LocalBlobStore(tmp_path),
            consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
            prefix="../unsafe",
        )


def test_slot_claims_reject_unpersisted_release_and_unsafe_lookup(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    coordinator = SlotClaimCoordinator(
        store,
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )
    plan = build_plan()
    slot = plan.slots[0]
    claim = coordinator.acquire(
        plan.plan_id,
        slot,
        owner_id="worker-a",
        ttl_s=30,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    forged = claim.model_copy(update={"epoch": 2})

    with pytest.raises(SlotClaimConflictError, match="no persisted claim record"):
        coordinator.release(forged, outcome=SlotClaimOutcome.ABANDONED)
    with pytest.raises(ValueError, match="slot_id has an invalid format"):
        coordinator.latest_claim(plan.plan_id, "../unsafe")

    store.put_if_absent(
        f"slot-claims/plans/{plan.plan_id}/slots/{slot.slot_id}/claims/{2:020d}.json",
        claim.canonical_bytes() + b"\n",
    )
    with pytest.raises(SlotClaimConflictError, match="mismatched identity"):
        coordinator.latest_claim(plan.plan_id, slot.slot_id)


def test_slot_claim_plan_status_reports_every_current_state_without_writes(tmp_path: Path) -> None:
    coordinator = SlotClaimCoordinator(
        LocalBlobStore(tmp_path),
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
    )
    plan = build_plan(rollouts_per_task=5)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    active = coordinator.acquire(
        plan.plan_id,
        plan.slots[1],
        owner_id="active-worker",
        ttl_s=30,
        now=start,
    )
    active_renewal = coordinator.renew(
        active,
        ttl_s=40,
        now=start + timedelta(seconds=1),
    )
    coordinator.acquire(
        plan.plan_id,
        plan.slots[2],
        owner_id="expired-worker",
        ttl_s=5,
        now=start,
    )
    completed = coordinator.acquire(
        plan.plan_id,
        plan.slots[3],
        owner_id="completed-worker",
        ttl_s=30,
        now=start,
    )
    coordinator.release(
        completed,
        outcome=SlotClaimOutcome.COMPLETED,
        trajectory_id=plan.slots[3].trajectory_id,
        now=start + timedelta(seconds=1),
    )
    abandoned = coordinator.acquire(
        plan.plan_id,
        plan.slots[4],
        owner_id="abandoned-worker",
        ttl_s=30,
        now=start,
    )
    coordinator.release(
        abandoned,
        outcome=SlotClaimOutcome.ABANDONED,
        now=start + timedelta(seconds=1),
    )
    files_before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json")))

    report = coordinator.inspect_plan(plan, now=start + timedelta(seconds=10))

    assert tuple(item.state for item in report.slots) == (
        SlotClaimState.UNCLAIMED,
        SlotClaimState.ACTIVE,
        SlotClaimState.EXPIRED,
        SlotClaimState.COMPLETED,
        SlotClaimState.ABANDONED,
    )
    assert report.slots[1].claim == active
    assert report.slots[1].renewal == active_renewal
    assert report.state_counts == {state: 1 for state in SlotClaimState}
    files_after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json")))
    assert files_after == files_before

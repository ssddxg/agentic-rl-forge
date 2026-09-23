from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import cast

from agentic_rl_forge.contracts import (
    RolloutPlan,
    RolloutSlot,
    SlotClaim,
    SlotClaimOutcome,
    TaskSpec,
    Trajectory,
    new_id,
    utc_now,
)
from agentic_rl_forge.rollout.callbacks import (
    CompositeRolloutCallback,
    RolloutCallback,
)
from agentic_rl_forge.rollout.loop import AgentLoop
from agentic_rl_forge.rollout.planning import validate_planned_trajectory
from agentic_rl_forge.storage import RenewableSlotClaimManager, SlotClaimManager


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    policy_version: str
    trajectories: tuple[Trajectory, ...]

    def grouped(self) -> dict[str, tuple[Trajectory, ...]]:
        groups: dict[str, list[Trajectory]] = {}
        for trajectory in self.trajectories:
            groups.setdefault(trajectory.group_id, []).append(trajectory)
        return {group_id: tuple(items) for group_id, items in groups.items()}

    def validate_on_policy(self, *, rollouts_per_task: int) -> None:
        for trajectory in self.trajectories:
            trajectory.require_on_policy(self.policy_version)
        for group_id, items in self.grouped().items():
            if len(items) != rollouts_per_task:
                raise ValueError(
                    f"group {group_id!r} contains {len(items)} rollouts, "
                    f"expected {rollouts_per_task}"
                )
            if len({item.task_id for item in items}) != 1:
                raise ValueError(f"group {group_id!r} mixes multiple tasks")


class RolloutScheduler:
    def __init__(
        self,
        loop_factory: Callable[[], AgentLoop],
        *,
        max_concurrency: int = 32,
        callbacks: Sequence[RolloutCallback] = (),
        slot_claims: SlotClaimManager | None = None,
        claim_owner_id: str | None = None,
        claim_ttl_s: float = 900.0,
        claim_renewal_interval_s: float | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if (slot_claims is None) != (claim_owner_id is None):
            raise ValueError("slot_claims and claim_owner_id must be configured together")
        if claim_ttl_s <= 0:
            raise ValueError("claim_ttl_s must be positive")
        if claim_renewal_interval_s is not None:
            if slot_claims is None:
                raise ValueError("claim renewal requires slot claims")
            if not callable(getattr(slot_claims, "renew", None)):
                raise ValueError("claim renewal requires a renewable slot claim manager")
            if claim_renewal_interval_s <= 0:
                raise ValueError("claim_renewal_interval_s must be positive")
            if claim_renewal_interval_s > claim_ttl_s / 2:
                raise ValueError("claim renewal interval must not exceed half of claim ttl")
        self._loop_factory = loop_factory
        self._max_concurrency = max_concurrency
        self._callbacks = CompositeRolloutCallback(callbacks)
        self._slot_claims = slot_claims
        self._renewable_slot_claims = (
            cast(RenewableSlotClaimManager, slot_claims)
            if claim_renewal_interval_s is not None
            else None
        )
        self._claim_owner_id = claim_owner_id
        self._claim_ttl_s = claim_ttl_s
        self._claim_renewal_interval_s = claim_renewal_interval_s

    async def collect(
        self,
        tasks: Sequence[TaskSpec],
        *,
        rollouts_per_task: int,
        seed: int = 0,
        plan: RolloutPlan | None = None,
        existing_trajectories: Sequence[Trajectory] = (),
    ) -> RolloutBatch:
        if rollouts_per_task < 1:
            raise ValueError("rollouts_per_task must be positive")
        if not tasks:
            raise ValueError("rollout collection requires at least one task")
        if plan is not None:
            self._validate_plan(plan, tasks, rollouts_per_task)
        elif existing_trajectories:
            raise ValueError("existing trajectories require a rollout plan")
        semaphore = asyncio.Semaphore(self._max_concurrency)
        jobs: list[asyncio.Task[Trajectory]] = []
        reused: dict[str, Trajectory] = {}
        if plan is not None:
            slots = plan.slots
            task_by_id = {task.task_id: task for task in tasks}
            reused = self._validate_existing(plan, existing_trajectories)
            for slot in slots:
                if slot.trajectory_id in reused:
                    await self._callbacks.on_trajectory(reused[slot.trajectory_id])
            for slot in slots:
                if slot.trajectory_id in reused:
                    continue
                jobs.append(
                    asyncio.create_task(
                        self._run_one(
                            semaphore,
                            task_by_id[slot.task_id],
                            slot.group_id,
                            slot.seed,
                            slot=slot,
                            plan_id=plan.plan_id,
                        )
                    )
                )
        else:
            slots = ()
            for task_index, task in enumerate(tasks):
                group_id = new_id("group")
                for rollout_index in range(rollouts_per_task):
                    rollout_seed = seed + task_index * rollouts_per_task + rollout_index
                    jobs.append(
                        asyncio.create_task(self._run_one(semaphore, task, group_id, rollout_seed))
                    )
        try:
            collected = tuple(await asyncio.gather(*jobs))
        except BaseException:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            raise
        if plan is not None:
            by_id = {**reused, **{item.trajectory_id: item for item in collected}}
            trajectories = tuple(by_id[slot.trajectory_id] for slot in slots)
        else:
            trajectories = collected
        versions = {trajectory.policy_version for trajectory in trajectories}
        if len(versions) > 1:
            raise ValueError(f"rollout batch contains multiple policy versions: {sorted(versions)}")
        version = next(iter(versions), "")
        batch = RolloutBatch(policy_version=version, trajectories=trajectories)
        if plan is not None and batch.policy_version != plan.policy_version:
            raise ValueError(
                f"rollout plan expects policy {plan.policy_version!r}, "
                f"received {batch.policy_version!r}"
            )
        batch.validate_on_policy(rollouts_per_task=rollouts_per_task)
        await self._callbacks.on_batch(batch)
        return batch

    async def _run_one(
        self,
        semaphore: asyncio.Semaphore,
        task: TaskSpec,
        group_id: str,
        seed: int,
        *,
        slot: RolloutSlot | None = None,
        plan_id: str | None = None,
    ) -> Trajectory:
        async with semaphore:
            claim: SlotClaim | None = None
            renewal_stop = asyncio.Event()
            renewal_task: asyncio.Task[None] | None = None
            rollout_task: asyncio.Task[Trajectory] | None = None
            if (
                slot is not None
                and plan_id is not None
                and self._slot_claims is not None
                and self._claim_owner_id is not None
            ):
                claim = await asyncio.to_thread(
                    self._slot_claims.acquire,
                    plan_id,
                    slot,
                    owner_id=self._claim_owner_id,
                    ttl_s=self._claim_ttl_s,
                )
                if self._claim_renewal_interval_s is not None:
                    renewal_task = asyncio.create_task(
                        self._renew_claim_until_stopped(claim, renewal_stop)
                    )
            try:
                rollout_task = asyncio.create_task(
                    self._execute_one(
                        task,
                        group_id,
                        seed,
                        slot=slot,
                        plan_id=plan_id,
                        claim=claim,
                    )
                )
                if renewal_task is not None:
                    done, _ = await asyncio.wait(
                        (rollout_task, renewal_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if renewal_task in done:
                        rollout_task.cancel()
                        await asyncio.gather(rollout_task, return_exceptions=True)
                        await renewal_task
                        raise RuntimeError("slot claim renewal stopped unexpectedly")
                trajectory = await rollout_task
                renewal_stop.set()
                if renewal_task is not None:
                    await renewal_task
                if claim is not None and self._slot_claims is not None:
                    await asyncio.to_thread(self._slot_claims.assert_current, claim)
                    await asyncio.to_thread(
                        self._slot_claims.release,
                        claim,
                        outcome=SlotClaimOutcome.COMPLETED,
                        trajectory_id=trajectory.trajectory_id,
                    )
                return trajectory
            except BaseException:
                if rollout_task is not None:
                    if not rollout_task.done():
                        rollout_task.cancel()
                    await asyncio.gather(rollout_task, return_exceptions=True)
                renewal_stop.set()
                if renewal_task is not None:
                    await asyncio.gather(renewal_task, return_exceptions=True)
                if claim is not None and self._slot_claims is not None:
                    with suppress(Exception):
                        await asyncio.to_thread(
                            self._slot_claims.release,
                            claim,
                            outcome=SlotClaimOutcome.ABANDONED,
                        )
                raise

    async def _execute_one(
        self,
        task: TaskSpec,
        group_id: str,
        seed: int,
        *,
        slot: RolloutSlot | None,
        plan_id: str | None,
        claim: SlotClaim | None,
    ) -> Trajectory:
        trajectory = await self._loop_factory().run(
            task,
            group_id=group_id,
            seed=seed,
            trajectory_id=slot.trajectory_id if slot is not None else None,
            provenance_metadata=(
                {
                    "rollout_plan_id": plan_id,
                    "rollout_slot_id": slot.slot_id,
                    "rollout_index": slot.rollout_index,
                    "rollout_seed": slot.seed,
                    **({"slot_claim_id": claim.claim_id} if claim is not None else {}),
                }
                if slot is not None and plan_id is not None
                else None
            ),
        )
        if claim is not None and self._slot_claims is not None:
            await asyncio.to_thread(self._slot_claims.assert_current, claim)
        await self._callbacks.on_trajectory(trajectory)
        return trajectory

    async def _renew_claim_until_stopped(
        self,
        claim: SlotClaim,
        stop: asyncio.Event,
    ) -> None:
        if self._renewable_slot_claims is None or self._claim_renewal_interval_s is None:
            raise RuntimeError("slot claim renewal is not configured")
        renewal_interval = timedelta(seconds=self._claim_renewal_interval_s)
        next_renewal_at = claim.acquired_at + renewal_interval
        while True:
            if stop.is_set():
                return
            delay_s = min(
                self._claim_renewal_interval_s,
                max(0.0, (next_renewal_at - utc_now()).total_seconds()),
            )
            if delay_s > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay_s)
                except asyncio.TimeoutError:
                    pass
                else:
                    return
            if stop.is_set():
                return
            renewal = await asyncio.to_thread(
                self._renewable_slot_claims.renew,
                claim,
                ttl_s=self._claim_ttl_s,
            )
            next_renewal_at = renewal.renewed_at + renewal_interval

    @staticmethod
    def _validate_plan(
        plan: RolloutPlan,
        tasks: Sequence[TaskSpec],
        rollouts_per_task: int,
    ) -> None:
        if plan.rollouts_per_task != rollouts_per_task:
            raise ValueError("rollout plan group size does not match collection request")
        if tuple(task.task_id for task in tasks) != plan.task_ids:
            raise ValueError("rollout plan tasks do not match collection tasks and order")

    @staticmethod
    def _validate_existing(
        plan: RolloutPlan,
        trajectories: Sequence[Trajectory],
    ) -> dict[str, Trajectory]:
        existing: dict[str, Trajectory] = {}
        for trajectory in trajectories:
            if trajectory.trajectory_id in existing:
                raise ValueError(f"duplicate existing trajectory {trajectory.trajectory_id!r}")
            validate_planned_trajectory(plan, trajectory)
            existing[trajectory.trajectory_id] = trajectory
        return existing

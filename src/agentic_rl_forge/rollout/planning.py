from __future__ import annotations

from collections.abc import Sequence

from agentic_rl_forge.contracts import RolloutPlan, RolloutSlot, TaskSpec, Trajectory


class RolloutPlanBuilder:
    def build(
        self,
        tasks: Sequence[TaskSpec],
        *,
        policy_version: str,
        source_sha256: str,
        config_digest: str,
        rollouts_per_task: int,
        seed: int = 0,
    ) -> RolloutPlan:
        if not tasks:
            raise ValueError("rollout plans require at least one task")
        if rollouts_per_task < 1:
            raise ValueError("rollouts_per_task must be positive")
        task_ids = tuple(task.task_id for task in tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("rollout plan tasks must have unique task IDs")
        plan_id = RolloutPlan.expected_plan_id(
            policy_version=policy_version,
            source_sha256=source_sha256,
            config_digest=config_digest,
            rollouts_per_task=rollouts_per_task,
            seed=seed,
            task_ids=task_ids,
        )
        slots: list[RolloutSlot] = []
        for task_index, task_id in enumerate(task_ids):
            group_id = RolloutPlan.group_id_for(plan_id, task_id)
            for rollout_index in range(rollouts_per_task):
                slot_seed = seed + task_index * rollouts_per_task + rollout_index
                slots.append(
                    RolloutPlan.build_slot(
                        plan_id,
                        task_id,
                        group_id,
                        rollout_index,
                        slot_seed,
                    )
                )
        return RolloutPlan(
            plan_id=plan_id,
            policy_version=policy_version,
            source_sha256=source_sha256,
            config_digest=config_digest,
            rollouts_per_task=rollouts_per_task,
            seed=seed,
            task_ids=task_ids,
            slots=tuple(slots),
        )


def validate_planned_trajectory(plan: RolloutPlan, trajectory: Trajectory) -> RolloutSlot:
    slot = next(
        (item for item in plan.slots if item.trajectory_id == trajectory.trajectory_id),
        None,
    )
    if slot is None:
        raise ValueError(f"trajectory {trajectory.trajectory_id!r} is not part of rollout plan")
    trajectory.require_on_policy(plan.policy_version)
    metadata = trajectory.provenance.metadata
    if (
        trajectory.task_id != slot.task_id
        or trajectory.group_id != slot.group_id
        or metadata.get("rollout_plan_id") != plan.plan_id
        or metadata.get("rollout_slot_id") != slot.slot_id
        or metadata.get("rollout_seed") != slot.seed
    ):
        raise ValueError(f"trajectory {trajectory.trajectory_id!r} does not match its rollout slot")
    return slot

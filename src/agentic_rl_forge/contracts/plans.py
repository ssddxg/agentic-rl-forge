from __future__ import annotations

import hashlib

import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel


class RolloutSlot(ContractModel):
    slot_id: str = Field(pattern=r"^slot_[0-9a-f]{24}$")
    trajectory_id: str = Field(pattern=r"^traj_[0-9a-f]{32}$")
    task_id: str = Field(min_length=1)
    group_id: str = Field(pattern=r"^group_[0-9a-f]{24}$")
    rollout_index: int = Field(ge=0)
    seed: int


class RolloutPlan(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{24}$")
    policy_version: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollouts_per_task: int = Field(ge=1)
    seed: int
    task_ids: tuple[str, ...]
    slots: tuple[RolloutSlot, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> RolloutPlan:
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("rollout plan task IDs must be non-empty and unique")
        if self.plan_id != self.expected_plan_id(
            policy_version=self.policy_version,
            source_sha256=self.source_sha256,
            config_digest=self.config_digest,
            rollouts_per_task=self.rollouts_per_task,
            seed=self.seed,
            task_ids=self.task_ids,
        ):
            raise ValueError("rollout plan ID does not match its identity fields")
        expected_slots: list[RolloutSlot] = []
        for task_id in self.task_ids:
            group_id = self.group_id_for(self.plan_id, task_id)
            for rollout_index in range(self.rollouts_per_task):
                slot_seed = self.seed + len(expected_slots)
                expected_slots.append(
                    self.build_slot(
                        self.plan_id,
                        task_id,
                        group_id,
                        rollout_index,
                        slot_seed,
                    )
                )
        if self.slots != tuple(expected_slots):
            raise ValueError("rollout plan slots do not match task order, group size, and seeds")
        return self

    @staticmethod
    def expected_plan_id(
        *,
        policy_version: str,
        source_sha256: str,
        config_digest: str,
        rollouts_per_task: int,
        seed: int,
        task_ids: tuple[str, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "policy_version": policy_version,
                "source_sha256": source_sha256,
                "config_digest": config_digest,
                "rollouts_per_task": rollouts_per_task,
                "seed": seed,
                "task_ids": task_ids,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"plan_{hashlib.sha256(payload).hexdigest()[:24]}"

    @staticmethod
    def group_id_for(plan_id: str, task_id: str) -> str:
        digest = hashlib.sha256(f"{plan_id}:group:{task_id}".encode()).hexdigest()
        return f"group_{digest[:24]}"

    @staticmethod
    def build_slot(
        plan_id: str,
        task_id: str,
        group_id: str,
        rollout_index: int,
        seed: int,
    ) -> RolloutSlot:
        digest = hashlib.sha256(
            f"{plan_id}:slot:{task_id}:{rollout_index}:{seed}".encode()
        ).hexdigest()
        return RolloutSlot(
            slot_id=f"slot_{digest[:24]}",
            trajectory_id=f"traj_{digest[:32]}",
            task_id=task_id,
            group_id=group_id,
            rollout_index=rollout_index,
            seed=seed,
        )

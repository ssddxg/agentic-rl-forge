from __future__ import annotations

import hashlib
from dataclasses import dataclass

import orjson

from agentic_rl_forge.contracts import (
    DataOrigin,
    Provenance,
    Trajectory,
    TrajectoryStatus,
    new_id,
)


@dataclass(frozen=True, slots=True)
class RejectedSample:
    trajectory_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class RejectionSamplingResult:
    accepted: tuple[Trajectory, ...]
    rejected: tuple[RejectedSample, ...]


class VerifiedRejectionSampler:
    def __init__(
        self,
        *,
        min_reward: float = 0.5,
        top_k_per_task: int = 4,
        require_success: bool = True,
    ) -> None:
        if top_k_per_task < 1:
            raise ValueError("top_k_per_task must be positive")
        self._min_reward = min_reward
        self._top_k_per_task = top_k_per_task
        self._require_success = require_success

    def select(self, trajectories: tuple[Trajectory, ...]) -> RejectionSamplingResult:
        candidates: dict[str, list[Trajectory]] = {}
        rejected: list[RejectedSample] = []
        seen: dict[str, set[str]] = {}
        for trajectory in trajectories:
            if self._require_success and trajectory.status is not TrajectoryStatus.SUCCEEDED:
                rejected.append(RejectedSample(trajectory.trajectory_id, "not_successful"))
                continue
            if trajectory.total_reward < self._min_reward:
                rejected.append(RejectedSample(trajectory.trajectory_id, "reward_below_threshold"))
                continue
            semantic_digest = self._semantic_digest(trajectory)
            task_seen = seen.setdefault(trajectory.task_id, set())
            if semantic_digest in task_seen:
                rejected.append(RejectedSample(trajectory.trajectory_id, "duplicate"))
                continue
            task_seen.add(semantic_digest)
            candidates.setdefault(trajectory.task_id, []).append(trajectory)
        accepted: list[Trajectory] = []
        for task_trajectories in candidates.values():
            ranked = sorted(
                task_trajectories,
                key=lambda item: (item.total_reward, -item.total_generated_tokens),
                reverse=True,
            )
            for rank, trajectory in enumerate(ranked):
                if rank >= self._top_k_per_task:
                    rejected.append(RejectedSample(trajectory.trajectory_id, "outside_top_k"))
                    continue
                accepted.append(self._derive(trajectory, rank=rank))
        return RejectionSamplingResult(accepted=tuple(accepted), rejected=tuple(rejected))

    @staticmethod
    def _derive(trajectory: Trajectory, *, rank: int) -> Trajectory:
        payload = trajectory.model_dump(mode="python")
        payload["trajectory_id"] = new_id("traj")
        payload["provenance"] = Provenance(
            origin=DataOrigin.REJECTION_SAMPLING,
            producer="verified-rejection-sampler",
            producer_version="1",
            parent_ids=(trajectory.trajectory_id,),
            transform="verified_top_k_selection",
            metadata={"rank": rank, "source_reward": trajectory.total_reward},
        )
        return Trajectory.model_validate(payload)

    @staticmethod
    def _semantic_digest(trajectory: Trajectory) -> str:
        actions = [
            {
                "kind": step.action.kind.value,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in step.action.tool_calls
                ],
                "final_answer": step.action.final_answer,
            }
            for step in trajectory.steps
        ]
        return hashlib.sha256(orjson.dumps(actions, option=orjson.OPT_SORT_KEYS)).hexdigest()

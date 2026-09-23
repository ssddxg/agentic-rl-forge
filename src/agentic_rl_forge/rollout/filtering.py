from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Sequence

import orjson
from pydantic import Field

from agentic_rl_forge.contracts import ContractModel, Trajectory


class RolloutGroupSignal(ContractModel):
    group_id: str
    task_id: str
    attempts: int = Field(ge=1)
    mean_reward: float
    reward_stddev: float = Field(ge=0.0)
    reward_range: float = Field(ge=0.0)
    unique_trajectory_ratio: float = Field(ge=0.0, le=1.0)
    signal_score: float = Field(ge=0.0)


class RejectedRolloutGroup(ContractModel):
    group_id: str
    task_id: str
    reasons: tuple[str, ...]
    signal: RolloutGroupSignal


class RolloutFilterResult(ContractModel):
    policy_version: str
    accepted: tuple[Trajectory, ...]
    rejected_groups: tuple[RejectedRolloutGroup, ...]
    signals: tuple[RolloutGroupSignal, ...]

    @property
    def accepted_group_count(self) -> int:
        return len({trajectory.group_id for trajectory in self.accepted})


class SignalAwareRolloutFilter:
    def __init__(
        self,
        *,
        expected_group_size: int | None = None,
        min_group_size: int = 2,
        min_reward_stddev: float = 1e-6,
        min_unique_trajectory_ratio: float = 0.0,
        keep_top_fraction: float = 1.0,
    ) -> None:
        if expected_group_size is not None and expected_group_size < 2:
            raise ValueError("expected_group_size must be at least two")
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least two")
        if min_reward_stddev < 0:
            raise ValueError("min_reward_stddev cannot be negative")
        if not 0.0 <= min_unique_trajectory_ratio <= 1.0:
            raise ValueError("min_unique_trajectory_ratio must be between zero and one")
        if not 0.0 < keep_top_fraction <= 1.0:
            raise ValueError("keep_top_fraction must be greater than zero and at most one")
        self._expected_group_size = expected_group_size
        self._min_group_size = min_group_size
        self._min_reward_stddev = min_reward_stddev
        self._min_unique_ratio = min_unique_trajectory_ratio
        self._keep_top_fraction = keep_top_fraction

    def filter(
        self,
        trajectories: Sequence[Trajectory],
        *,
        expected_policy_version: str | None = None,
    ) -> RolloutFilterResult:
        if not trajectories:
            raise ValueError("cannot filter an empty rollout collection")
        versions = {trajectory.policy_version for trajectory in trajectories}
        if len(versions) != 1:
            raise ValueError("rollout filtering requires exactly one policy version")
        policy_version = next(iter(versions))
        if expected_policy_version is not None:
            for trajectory in trajectories:
                trajectory.require_on_policy(expected_policy_version)

        groups: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            groups.setdefault(trajectory.group_id, []).append(trajectory)
        signals = {
            group_id: self._signal(group_id, tuple(items)) for group_id, items in groups.items()
        }
        rejection_reasons: dict[str, list[str]] = {}
        eligible = []
        for group_id, signal in signals.items():
            reasons = []
            if signal.attempts < self._min_group_size:
                reasons.append("group_too_small")
            if (
                self._expected_group_size is not None
                and signal.attempts != self._expected_group_size
            ):
                reasons.append("unexpected_group_size")
            if signal.reward_stddev < self._min_reward_stddev:
                reasons.append("reward_variance_below_threshold")
            if signal.unique_trajectory_ratio < self._min_unique_ratio:
                reasons.append("trajectory_diversity_below_threshold")
            if reasons:
                rejection_reasons[group_id] = reasons
            else:
                eligible.append(signal)

        eligible.sort(key=lambda item: (-item.signal_score, item.group_id))
        keep_count = max(1, math.ceil(len(eligible) * self._keep_top_fraction)) if eligible else 0
        accepted_group_ids = {item.group_id for item in eligible[:keep_count]}
        for signal in eligible[keep_count:]:
            rejection_reasons[signal.group_id] = ["outside_top_fraction"]

        accepted = tuple(
            trajectory for trajectory in trajectories if trajectory.group_id in accepted_group_ids
        )
        rejected = tuple(
            RejectedRolloutGroup(
                group_id=group_id,
                task_id=signals[group_id].task_id,
                reasons=tuple(reasons),
                signal=signals[group_id],
            )
            for group_id, reasons in sorted(rejection_reasons.items())
        )
        return RolloutFilterResult(
            policy_version=policy_version,
            accepted=accepted,
            rejected_groups=rejected,
            signals=tuple(signals[group_id] for group_id in sorted(signals)),
        )

    @staticmethod
    def _signal(
        group_id: str,
        trajectories: tuple[Trajectory, ...],
    ) -> RolloutGroupSignal:
        task_ids = {trajectory.task_id for trajectory in trajectories}
        if len(task_ids) != 1:
            raise ValueError(f"group {group_id!r} mixes multiple tasks")
        rewards = [trajectory.total_reward for trajectory in trajectories]
        reward_stddev = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        unique_ratio = len({SignalAwareRolloutFilter._digest(item) for item in trajectories}) / len(
            trajectories
        )
        score = reward_stddev * unique_ratio * math.log2(len(trajectories) + 1)
        return RolloutGroupSignal(
            group_id=group_id,
            task_id=next(iter(task_ids)),
            attempts=len(trajectories),
            mean_reward=sum(rewards) / len(rewards),
            reward_stddev=reward_stddev,
            reward_range=max(rewards) - min(rewards),
            unique_trajectory_ratio=unique_ratio,
            signal_score=score,
        )

    @staticmethod
    def _digest(trajectory: Trajectory) -> str:
        payload = [
            {
                "kind": step.action.kind.value,
                "reasoning": step.action.reasoning,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in step.action.tool_calls
                ],
                "final_answer": step.action.final_answer,
            }
            for step in trajectory.steps
        ]
        return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()

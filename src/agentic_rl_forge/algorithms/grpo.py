from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroupReward:
    sample_id: str
    reward: float


@dataclass(frozen=True, slots=True)
class GroupAdvantage:
    sample_id: str
    reward: float
    advantage: float


def group_relative_advantages(
    samples: Sequence[GroupReward],
    *,
    epsilon: float = 1e-8,
) -> tuple[GroupAdvantage, ...]:
    if len(samples) < 2:
        raise ValueError("GRPO requires at least two samples per group")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rewards = [sample.reward for sample in samples]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    standard_deviation = math.sqrt(variance)
    if standard_deviation < epsilon:
        return tuple(GroupAdvantage(sample.sample_id, sample.reward, 0.0) for sample in samples)
    return tuple(
        GroupAdvantage(
            sample_id=sample.sample_id,
            reward=sample.reward,
            advantage=(sample.reward - mean) / (standard_deviation + epsilon),
        )
        for sample in samples
    )

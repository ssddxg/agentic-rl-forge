from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from pathlib import Path

import orjson
from pydantic import Field

from agentic_rl_forge.contracts import (
    ContractModel,
    DataOrigin,
    JsonObject,
    Provenance,
    Trajectory,
)
from agentic_rl_forge.search import ProcessRewardInput


class PRMExample(ContractModel):
    example_id: str
    trajectory_id: str
    task_id: str
    group_id: str
    policy_version: str
    environment_version: str
    step_index: int = Field(ge=0)
    split: str = Field(pattern=r"^(train|validation|test)$")
    input: ProcessRewardInput
    target: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    sample_weight: float = Field(ge=0.0)
    provenance: Provenance
    metadata: JsonObject = Field(default_factory=dict)


class PRMDatasetSummary(ContractModel):
    example_count: int = Field(ge=0)
    trajectory_count: int = Field(ge=0)
    task_count: int = Field(ge=0)
    split_counts: dict[str, int]
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    mean_target: float


class PRMDatasetBuilder:
    def __init__(
        self,
        *,
        gamma: float = 1.0,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        min_confidence: float = 0.0,
        skip_zero_variance_groups: bool = False,
        reward_stddev_epsilon: float = 1e-6,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        ratios = (train_ratio, validation_ratio, test_ratio)
        if any(ratio < 0 for ratio in ratios) or not 0.999999 <= sum(ratios) <= 1.000001:
            raise ValueError("dataset split ratios must be non-negative and sum to one")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if reward_stddev_epsilon < 0:
            raise ValueError("reward_stddev_epsilon cannot be negative")
        self._gamma = gamma
        self._train_ratio = train_ratio
        self._validation_ratio = validation_ratio
        self._test_ratio = test_ratio
        self._min_confidence = min_confidence
        self._skip_zero_variance_groups = skip_zero_variance_groups
        self._reward_stddev_epsilon = reward_stddev_epsilon

    def build(self, trajectories: tuple[Trajectory, ...]) -> tuple[PRMExample, ...]:
        groups: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            groups.setdefault(trajectory.group_id, []).append(trajectory)
        group_statistics = {
            group_id: self._group_statistics(group_id, tuple(items))
            for group_id, items in groups.items()
        }
        examples = []
        for trajectory in trajectories:
            mean_reward, reward_stddev = group_statistics[trajectory.group_id]
            if self._skip_zero_variance_groups and (
                len(groups[trajectory.group_id]) < 2 or reward_stddev < self._reward_stddev_epsilon
            ):
                continue
            group_advantage = (
                (trajectory.total_reward - mean_reward) / reward_stddev
                if reward_stddev >= self._reward_stddev_epsilon
                else 0.0
            )
            returns = self._discounted_returns(trajectory)
            split = self._split_for_task(trajectory.task_id)
            for step, target in zip(trajectory.steps, returns, strict=True):
                confidence = self._step_confidence(trajectory, step.index)
                if confidence < self._min_confidence:
                    continue
                examples.append(
                    PRMExample(
                        example_id=self._example_id(trajectory.trajectory_id, step.index),
                        trajectory_id=trajectory.trajectory_id,
                        task_id=trajectory.task_id,
                        group_id=trajectory.group_id,
                        policy_version=trajectory.policy_version,
                        environment_version=trajectory.environment_version,
                        step_index=step.index,
                        split=split,
                        input=ProcessRewardInput(
                            state={
                                "messages": [
                                    message.model_dump(mode="json", exclude_none=True)
                                    for message in step.input_messages
                                ],
                                "snapshot": (
                                    step.snapshot_before.model_dump(mode="json", exclude_none=True)
                                    if step.snapshot_before is not None
                                    else None
                                ),
                            },
                            action=step.action.model_dump(mode="json", exclude_none=True),
                            history=tuple(
                                {
                                    "step_index": previous.index,
                                    "action": previous.action.model_dump(
                                        mode="json", exclude_none=True
                                    ),
                                    "tool_results": [
                                        result.model_dump(mode="json", exclude_none=True)
                                        for result in previous.tool_results
                                    ],
                                }
                                for previous in trajectory.steps[: step.index]
                            ),
                            metadata={
                                "trajectory_id": trajectory.trajectory_id,
                                "task_id": trajectory.task_id,
                                "step_index": step.index,
                            },
                        ),
                        target=max(-1.0, min(1.0, target)),
                        confidence=confidence,
                        sample_weight=confidence,
                        provenance=Provenance(
                            origin=DataOrigin.PRM_DATASET,
                            producer="prm-dataset-builder",
                            producer_version="1",
                            created_at=trajectory.completed_at or trajectory.started_at,
                            parent_ids=(trajectory.trajectory_id,),
                            transform="discounted_verified_return",
                            metadata={
                                "gamma": self._gamma,
                                "source_origin": trajectory.provenance.origin.value,
                            },
                        ),
                        metadata={
                            "raw_return": target,
                            "trajectory_reward": trajectory.total_reward,
                            "group_reward_mean": mean_reward,
                            "group_reward_stddev": reward_stddev,
                            "group_advantage": group_advantage,
                        },
                    )
                )
        return tuple(examples)

    def summarize(self, examples: tuple[PRMExample, ...]) -> PRMDatasetSummary:
        targets = [example.target for example in examples]
        return PRMDatasetSummary(
            example_count=len(examples),
            trajectory_count=len({example.trajectory_id for example in examples}),
            task_count=len({example.task_id for example in examples}),
            split_counts=dict(sorted(Counter(example.split for example in examples).items())),
            positive_count=sum(target > 0 for target in targets),
            negative_count=sum(target < 0 for target in targets),
            neutral_count=sum(target == 0 for target in targets),
            mean_target=sum(targets) / len(targets) if targets else 0.0,
        )

    def _discounted_returns(self, trajectory: Trajectory) -> tuple[float, ...]:
        running = trajectory.final_reward.total
        returns = [0.0] * len(trajectory.steps)
        for step in reversed(trajectory.steps):
            running = step.rewards.total + self._gamma * running
            returns[step.index] = running
        return tuple(returns)

    @staticmethod
    def _step_confidence(trajectory: Trajectory, step_index: int) -> float:
        signals = [
            signal for step in trajectory.steps[step_index:] for signal in step.rewards.signals
        ] + list(trajectory.final_reward.signals)
        if not signals:
            return 1.0
        return sum(signal.confidence for signal in signals) / len(signals)

    @staticmethod
    def _group_statistics(
        group_id: str,
        trajectories: tuple[Trajectory, ...],
    ) -> tuple[float, float]:
        task_ids = {trajectory.task_id for trajectory in trajectories}
        if len(task_ids) != 1:
            raise ValueError(f"group {group_id!r} mixes multiple tasks")
        rewards = [trajectory.total_reward for trajectory in trajectories]
        mean = sum(rewards) / len(rewards)
        stddev = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        return mean, stddev

    def _split_for_task(self, task_id: str) -> str:
        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < self._train_ratio:
            return "train"
        if value < self._train_ratio + self._validation_ratio:
            return "validation"
        return "test"

    def _example_id(self, trajectory_id: str, step_index: int) -> str:
        payload = orjson.dumps(
            {
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "gamma": self._gamma,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"prm_{hashlib.sha256(payload).hexdigest()[:24]}"


def export_prm_jsonl(examples: tuple[PRMExample, ...], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for example in examples:
            output.write(example.canonical_bytes())
            output.write(b"\n")
    return len(examples)

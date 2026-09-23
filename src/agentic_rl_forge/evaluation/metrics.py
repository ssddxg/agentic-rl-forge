from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from datetime import datetime

import orjson
from pydantic import Field

from agentic_rl_forge.contracts import (
    ContractModel,
    JsonObject,
    Trajectory,
    TrajectoryStatus,
    new_id,
    utc_now,
)


class MetricValue(ContractModel):
    value: float
    numerator: float
    denominator: int = Field(ge=0)
    unit: str

    @classmethod
    def mean(cls, numerator: float, denominator: int, *, unit: str) -> MetricValue:
        value = numerator / denominator if denominator else 0.0
        return cls(
            value=value,
            numerator=numerator,
            denominator=denominator,
            unit=unit,
        )


class GroupDiagnostic(ContractModel):
    group_id: str
    task_id: str
    attempts: int = Field(ge=1)
    successes: int = Field(ge=0)
    mean_reward: float
    reward_stddev: float = Field(ge=0.0)
    unique_trajectory_ratio: float = Field(ge=0.0, le=1.0)
    has_learning_signal: bool


class BenchmarkReport(ContractModel):
    report_id: str = Field(default_factory=lambda: new_id("report"))
    benchmark: str = Field(min_length=1)
    run_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    task_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    policy_versions: tuple[str, ...]
    environment_versions: tuple[str, ...]
    status_counts: dict[str, int]
    metrics: dict[str, MetricValue]
    group_diagnostics: tuple[GroupDiagnostic, ...]
    metadata: JsonObject = Field(default_factory=dict)


class BenchmarkAggregator:
    def __init__(self, *, min_reward_stddev: float = 1e-6) -> None:
        if min_reward_stddev < 0:
            raise ValueError("min_reward_stddev cannot be negative")
        self._min_reward_stddev = min_reward_stddev

    def aggregate(
        self,
        trajectories: tuple[Trajectory, ...],
        *,
        benchmark: str,
        run_id: str | None = None,
        metadata: JsonObject | None = None,
    ) -> BenchmarkReport:
        if not trajectories:
            raise ValueError("cannot aggregate an empty trajectory collection")
        groups: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            groups.setdefault(trajectory.group_id, []).append(trajectory)

        diagnostics = tuple(
            self._diagnose_group(group_id, tuple(items))
            for group_id, items in sorted(groups.items())
        )
        attempts = len(trajectories)
        success_count = sum(
            trajectory.status is TrajectoryStatus.SUCCEEDED for trajectory in trajectories
        )
        group_success_count = sum(item.successes > 0 for item in diagnostics)
        generated_tokens = sum(item.total_generated_tokens for item in trajectories)
        observation_tokens = sum(item.total_observation_tokens for item in trajectories)
        step_count = sum(len(item.steps) for item in trajectories)
        total_reward = sum(item.total_reward for item in trajectories)
        tool_results = [
            result
            for trajectory in trajectories
            for step in trajectory.steps
            for result in step.tool_results
        ]
        tool_calls = sum(
            len(step.action.tool_calls) for trajectory in trajectories for step in trajectory.steps
        )
        durations_ms = [
            (trajectory.completed_at - trajectory.started_at).total_seconds() * 1000.0
            for trajectory in trajectories
            if trajectory.completed_at is not None
        ]
        zero_variance_groups = sum(
            item.reward_stddev < self._min_reward_stddev for item in diagnostics
        )
        signal_groups = sum(item.has_learning_signal for item in diagnostics)
        unique_ratio_sum = sum(item.unique_trajectory_ratio for item in diagnostics)
        metrics = {
            "attempt_success_rate": MetricValue.mean(float(success_count), attempts, unit="ratio"),
            "group_pass_rate": MetricValue.mean(
                float(group_success_count), len(diagnostics), unit="ratio"
            ),
            "mean_reward": MetricValue.mean(total_reward, attempts, unit="reward"),
            "mean_generated_tokens": MetricValue.mean(
                float(generated_tokens), attempts, unit="tokens_per_attempt"
            ),
            "mean_observation_tokens": MetricValue.mean(
                float(observation_tokens), attempts, unit="tokens_per_attempt"
            ),
            "mean_steps": MetricValue.mean(float(step_count), attempts, unit="steps_per_attempt"),
            "mean_tool_calls": MetricValue.mean(
                float(tool_calls), attempts, unit="calls_per_attempt"
            ),
            "tool_success_rate": MetricValue.mean(
                float(sum(result.ok for result in tool_results)),
                len(tool_results),
                unit="ratio",
            ),
            "mean_duration_ms": MetricValue.mean(
                sum(durations_ms), len(durations_ms), unit="milliseconds"
            ),
            "error_rate": MetricValue.mean(
                float(
                    sum(trajectory.status is TrajectoryStatus.ERROR for trajectory in trajectories)
                ),
                attempts,
                unit="ratio",
            ),
            "truncation_rate": MetricValue.mean(
                float(
                    sum(
                        trajectory.status is TrajectoryStatus.TRUNCATED
                        for trajectory in trajectories
                    )
                ),
                attempts,
                unit="ratio",
            ),
            "zero_variance_group_rate": MetricValue.mean(
                float(zero_variance_groups), len(diagnostics), unit="ratio"
            ),
            "learning_signal_group_rate": MetricValue.mean(
                float(signal_groups), len(diagnostics), unit="ratio"
            ),
            "mean_unique_trajectory_ratio": MetricValue.mean(
                unique_ratio_sum, len(diagnostics), unit="ratio"
            ),
        }
        return BenchmarkReport(
            benchmark=benchmark,
            run_id=run_id,
            task_count=len({trajectory.task_id for trajectory in trajectories}),
            group_count=len(diagnostics),
            attempt_count=attempts,
            policy_versions=tuple(sorted({item.policy_version for item in trajectories})),
            environment_versions=tuple(sorted({item.environment_version for item in trajectories})),
            status_counts=dict(sorted(Counter(item.status.value for item in trajectories).items())),
            metrics=metrics,
            group_diagnostics=diagnostics,
            metadata=metadata or {},
        )

    def _diagnose_group(
        self,
        group_id: str,
        trajectories: tuple[Trajectory, ...],
    ) -> GroupDiagnostic:
        task_ids = {trajectory.task_id for trajectory in trajectories}
        if len(task_ids) != 1:
            raise ValueError(f"group {group_id!r} mixes multiple tasks")
        rewards = [trajectory.total_reward for trajectory in trajectories]
        stddev = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        unique_ratio = len({self._semantic_digest(item) for item in trajectories}) / len(
            trajectories
        )
        return GroupDiagnostic(
            group_id=group_id,
            task_id=next(iter(task_ids)),
            attempts=len(trajectories),
            successes=sum(item.status is TrajectoryStatus.SUCCEEDED for item in trajectories),
            mean_reward=sum(rewards) / len(rewards),
            reward_stddev=stddev,
            unique_trajectory_ratio=unique_ratio,
            has_learning_signal=len(trajectories) > 1 and stddev >= self._min_reward_stddev,
        )

    @staticmethod
    def _semantic_digest(trajectory: Trajectory) -> str:
        actions = [
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
        return hashlib.sha256(orjson.dumps(actions, option=orjson.OPT_SORT_KEYS)).hexdigest()

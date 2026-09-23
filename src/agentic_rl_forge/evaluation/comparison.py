from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import Field

from agentic_rl_forge.contracts import (
    ContractModel,
    JsonObject,
    Trajectory,
    TrajectoryStatus,
    new_id,
    utc_now,
)


class PairedMetricDelta(ContractModel):
    baseline_mean: float
    candidate_mean: float
    absolute_delta: float
    relative_delta: float | None
    confidence_low: float
    confidence_high: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    sample_count: int = Field(ge=1)
    unit: str


class BenchmarkComparisonReport(ContractModel):
    comparison_id: str = Field(default_factory=lambda: new_id("comparison"))
    benchmark: str = Field(min_length=1)
    baseline_name: str = Field(min_length=1)
    candidate_name: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    matched_task_count: int = Field(ge=1)
    baseline_policy_versions: tuple[str, ...]
    candidate_policy_versions: tuple[str, ...]
    metrics: dict[str, PairedMetricDelta]
    metadata: JsonObject = Field(default_factory=dict)


class BenchmarkComparator:
    def __init__(
        self,
        *,
        bootstrap_samples: int = 2000,
        confidence_level: float = 0.95,
        seed: int = 0,
        require_equal_attempts: bool = True,
    ) -> None:
        if bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        self._bootstrap_samples = bootstrap_samples
        self._confidence_level = confidence_level
        self._seed = seed
        self._require_equal_attempts = require_equal_attempts

    def compare(
        self,
        baseline: Sequence[Trajectory],
        candidate: Sequence[Trajectory],
        *,
        benchmark: str,
        baseline_name: str,
        candidate_name: str,
        metadata: JsonObject | None = None,
    ) -> BenchmarkComparisonReport:
        baseline_tasks = self._by_task(baseline)
        candidate_tasks = self._by_task(candidate)
        matched_tasks = sorted(set(baseline_tasks) & set(candidate_tasks))
        if not matched_tasks:
            raise ValueError("benchmark comparison requires at least one matched task")
        if self._require_equal_attempts:
            mismatched = [
                task_id
                for task_id in matched_tasks
                if len(baseline_tasks[task_id]) != len(candidate_tasks[task_id])
            ]
            if mismatched:
                raise ValueError(
                    "matched tasks have different attempt counts: " + ", ".join(mismatched)
                )

        metric_specs: dict[
            str,
            tuple[Callable[[tuple[Trajectory, ...]], float], str],
        ] = {
            "task_pass_rate": (
                lambda items: float(
                    any(item.status is TrajectoryStatus.SUCCEEDED for item in items)
                ),
                "ratio",
            ),
            "mean_reward": (
                lambda items: sum(item.total_reward for item in items) / len(items),
                "reward",
            ),
            "mean_generated_tokens": (
                lambda items: sum(item.total_generated_tokens for item in items) / len(items),
                "tokens_per_attempt",
            ),
            "mean_steps": (
                lambda items: sum(len(item.steps) for item in items) / len(items),
                "steps_per_attempt",
            ),
            "mean_tool_calls": (
                lambda items: (
                    sum(len(step.action.tool_calls) for item in items for step in item.steps)
                    / len(items)
                ),
                "calls_per_attempt",
            ),
        }
        metrics = {}
        for name, (extractor, unit) in metric_specs.items():
            baseline_values = [extractor(baseline_tasks[task_id]) for task_id in matched_tasks]
            candidate_values = [extractor(candidate_tasks[task_id]) for task_id in matched_tasks]
            metrics[name] = self._paired_delta(
                baseline_values,
                candidate_values,
                unit=unit,
                metric_name=name,
            )
        return BenchmarkComparisonReport(
            benchmark=benchmark,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            matched_task_count=len(matched_tasks),
            baseline_policy_versions=tuple(
                sorted({trajectory.policy_version for trajectory in baseline})
            ),
            candidate_policy_versions=tuple(
                sorted({trajectory.policy_version for trajectory in candidate})
            ),
            metrics=metrics,
            metadata={
                "bootstrap_samples": self._bootstrap_samples,
                "seed": self._seed,
                "require_equal_attempts": self._require_equal_attempts,
                **(metadata or {}),
            },
        )

    def _paired_delta(
        self,
        baseline: list[float],
        candidate: list[float],
        *,
        unit: str,
        metric_name: str,
    ) -> PairedMetricDelta:
        deltas = [
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
        ]
        baseline_mean = sum(baseline) / len(baseline)
        candidate_mean = sum(candidate) / len(candidate)
        delta = candidate_mean - baseline_mean
        rng = random.Random(f"{self._seed}:{metric_name}")
        bootstrap = []
        for _ in range(self._bootstrap_samples):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            bootstrap.append(sum(sample) / len(sample))
        bootstrap.sort()
        alpha = 1.0 - self._confidence_level
        lower_index = int((alpha / 2.0) * (len(bootstrap) - 1))
        upper_index = int((1.0 - alpha / 2.0) * (len(bootstrap) - 1))
        return PairedMetricDelta(
            baseline_mean=baseline_mean,
            candidate_mean=candidate_mean,
            absolute_delta=delta,
            relative_delta=delta / abs(baseline_mean) if baseline_mean != 0 else None,
            confidence_low=bootstrap[lower_index],
            confidence_high=bootstrap[upper_index],
            confidence_level=self._confidence_level,
            sample_count=len(deltas),
            unit=unit,
        )

    @staticmethod
    def _by_task(
        trajectories: Sequence[Trajectory],
    ) -> dict[str, tuple[Trajectory, ...]]:
        grouped: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            grouped.setdefault(trajectory.task_id, []).append(trajectory)
        return {task_id: tuple(items) for task_id, items in grouped.items()}

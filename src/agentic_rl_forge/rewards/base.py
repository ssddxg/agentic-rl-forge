from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_rl_forge.contracts import (
    AgentAction,
    RewardSignal,
    RewardSource,
    RewardSummary,
    TaskSpec,
    ToolResult,
    TrajectoryStatus,
    TrajectoryStep,
)


@dataclass(frozen=True, slots=True)
class StepRewardContext:
    task: TaskSpec
    prior_steps: tuple[TrajectoryStep, ...]
    action: AgentAction
    tool_results: tuple[ToolResult, ...]
    generated_token_count: int


@dataclass(frozen=True, slots=True)
class FinalRewardContext:
    task: TaskSpec
    steps: tuple[TrajectoryStep, ...]
    status: TrajectoryStatus


class RewardComponent(Protocol):
    @property
    def name(self) -> str: ...

    async def score_step(self, context: StepRewardContext) -> tuple[RewardSignal, ...]: ...

    async def score_final(self, context: FinalRewardContext) -> tuple[RewardSignal, ...]: ...


class RewardEngine:
    def __init__(self, components: tuple[RewardComponent, ...]) -> None:
        names = [component.name for component in components]
        if len(names) != len(set(names)):
            raise ValueError("reward component names must be unique")
        self._components = components

    async def score_step(self, context: StepRewardContext) -> RewardSummary:
        signals: list[RewardSignal] = []
        for component in self._components:
            signals.extend(await component.score_step(context))
        return RewardSummary(signals=tuple(signals))

    async def score_final(self, context: FinalRewardContext) -> RewardSummary:
        signals: list[RewardSignal] = []
        for component in self._components:
            signals.extend(await component.score_final(context))
        return RewardSummary(signals=tuple(signals))

    @staticmethod
    def is_success(summary: RewardSummary) -> bool:
        outcome_signals = [
            signal for signal in summary.signals if signal.source is RewardSource.OUTCOME
        ]
        return bool(outcome_signals) and all(
            signal.weighted_value > 0 for signal in outcome_signals
        )

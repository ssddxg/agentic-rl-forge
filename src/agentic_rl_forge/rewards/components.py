from __future__ import annotations

import re

import orjson

from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    RewardSignal,
    RewardSource,
    TrajectoryStatus,
)
from agentic_rl_forge.rewards.base import FinalRewardContext, StepRewardContext


class EmptyRewardComponent:
    @property
    def name(self) -> str:
        return type(self).__name__

    async def score_step(self, context: StepRewardContext) -> tuple[RewardSignal, ...]:
        del context
        return ()

    async def score_final(self, context: FinalRewardContext) -> tuple[RewardSignal, ...]:
        del context
        return ()


class ExactMatchOutcome(EmptyRewardComponent):
    def __init__(self, *, success_reward: float = 1.0, failure_reward: float = 0.0) -> None:
        self._success_reward = success_reward
        self._failure_reward = failure_reward

    async def score_final(self, context: FinalRewardContext) -> tuple[RewardSignal, ...]:
        if context.task.verifier.kind != "exact_match":
            return ()
        expected_value = context.task.verifier.config.get("answer")
        if isinstance(expected_value, str):
            expected = {self._normalize(expected_value)}
        elif isinstance(expected_value, list) and all(
            isinstance(value, str) for value in expected_value
        ):
            expected = {self._normalize(value) for value in expected_value}
        else:
            raise ValueError("exact_match verifier requires a string or list of strings")
        answer = None
        if context.steps and context.steps[-1].action.kind is ActionKind.FINAL:
            answer = context.steps[-1].action.final_answer
        matched = answer is not None and self._normalize(answer) in expected
        return (
            RewardSignal(
                name="exact_match",
                source=RewardSource.OUTCOME,
                value=self._success_reward if matched else self._failure_reward,
                terminal=True,
                metadata={"matched": matched},
            ),
        )

    @staticmethod
    def _normalize(value: str) -> str:
        lowered = value.casefold().strip()
        without_articles = re.sub(r"\b(a|an|the)\b", " ", lowered)
        without_punctuation = re.sub(r"[^\w\s]", " ", without_articles)
        return " ".join(without_punctuation.split())


class ToolExecutionReward(EmptyRewardComponent):
    def __init__(self, *, failure_penalty: float = -0.1) -> None:
        self._failure_penalty = failure_penalty

    async def score_step(self, context: StepRewardContext) -> tuple[RewardSignal, ...]:
        failures = [result for result in context.tool_results if not result.ok]
        if not failures:
            return ()
        return (
            RewardSignal(
                name="tool_execution_failures",
                source=RewardSource.TOOL,
                value=self._failure_penalty * len(failures),
                metadata={
                    "error_codes": [result.error_code for result in failures],
                    "failure_count": len(failures),
                },
            ),
        )


class CostReward(EmptyRewardComponent):
    def __init__(
        self,
        *,
        per_generated_token: float = -0.00001,
        per_tool_call: float = -0.002,
    ) -> None:
        self._per_generated_token = per_generated_token
        self._per_tool_call = per_tool_call

    async def score_step(self, context: StepRewardContext) -> tuple[RewardSignal, ...]:
        cost = (
            self._per_generated_token * context.generated_token_count
            + self._per_tool_call * len(context.action.tool_calls)
        )
        if cost == 0:
            return ()
        return (
            RewardSignal(
                name="compute_and_tool_cost",
                source=RewardSource.COST,
                value=cost,
                metadata={
                    "generated_tokens": context.generated_token_count,
                    "tool_calls": len(context.action.tool_calls),
                },
            ),
        )


class InvalidActionReward(EmptyRewardComponent):
    def __init__(self, *, penalty: float = -0.25) -> None:
        self._penalty = penalty

    async def score_step(self, context: StepRewardContext) -> tuple[RewardSignal, ...]:
        if context.action.kind is not ActionKind.INVALID:
            return ()
        return (
            RewardSignal(
                name="invalid_action",
                source=RewardSource.FORMAT,
                value=self._penalty,
            ),
        )


class RepeatedActionPenalty(EmptyRewardComponent):
    def __init__(self, *, repeat_threshold: int = 2, penalty: float = -0.2) -> None:
        if repeat_threshold < 1:
            raise ValueError("repeat_threshold must be positive")
        self._repeat_threshold = repeat_threshold
        self._penalty = penalty

    async def score_step(self, context: StepRewardContext) -> tuple[RewardSignal, ...]:
        current = self.signature(context.action)
        repeat_count = 0
        for step in reversed(context.prior_steps):
            if self.signature(step.action) != current:
                break
            repeat_count += 1
        if repeat_count < self._repeat_threshold:
            return ()
        return (
            RewardSignal(
                name="repeated_action",
                source=RewardSource.ANTI_HACKING,
                value=self._penalty * (repeat_count - self._repeat_threshold + 1),
                metadata={"consecutive_previous_repeats": repeat_count},
            ),
        )

    @staticmethod
    def signature(action: AgentAction) -> str:
        payload = {
            "kind": action.kind.value,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments} for call in action.tool_calls
            ],
            "final_answer": action.final_answer,
        }
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8")


class CompletionGuardReward(EmptyRewardComponent):
    def __init__(self, *, truncated_penalty: float = -0.1, error_penalty: float = -0.5) -> None:
        self._truncated_penalty = truncated_penalty
        self._error_penalty = error_penalty

    async def score_final(self, context: FinalRewardContext) -> tuple[RewardSignal, ...]:
        value = 0.0
        if context.status is TrajectoryStatus.TRUNCATED:
            value = self._truncated_penalty
        elif context.status is TrajectoryStatus.ERROR:
            value = self._error_penalty
        if value == 0:
            return ()
        return (
            RewardSignal(
                name="incomplete_trajectory",
                source=RewardSource.ANTI_HACKING,
                value=value,
                terminal=True,
                metadata={"status": context.status.value},
            ),
        )

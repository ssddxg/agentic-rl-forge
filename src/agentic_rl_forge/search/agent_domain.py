from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    EnvironmentSnapshot,
    Message,
    MessageRole,
    TaskSpec,
    ToolResult,
)
from agentic_rl_forge.environments import AgentEnvironment
from agentic_rl_forge.rewards.components import RepeatedActionPenalty
from agentic_rl_forge.rollout import AgentPolicy, GenerationRequest, PolicyOutput
from agentic_rl_forge.search.budget import AdaptiveComputeBudget, BudgetSignals
from agentic_rl_forge.search.mcts import (
    ActionCandidate,
    AsyncMCTS,
    MCTSConfig,
    MCTSResult,
    SearchTransition,
)
from agentic_rl_forge.search.prm import ProcessRewardInput, ProcessRewardModel, ProcessScore


@dataclass(frozen=True, slots=True)
class AgentSearchState:
    session_id: str
    task: TaskSpec
    messages: tuple[Message, ...]
    snapshot: EnvironmentSnapshot
    depth: int = 0
    action_history: tuple[AgentAction, ...] = ()
    latest_tool_results: tuple[ToolResult, ...] = ()
    terminal: bool = False


class PolicyCandidateGenerator:
    def __init__(
        self,
        policy: AgentPolicy,
        *,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self._policy = policy
        self._request = GenerationRequest(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    async def propose(
        self,
        state: AgentSearchState,
        *,
        branch_factor: int,
        seed: int,
    ) -> Sequence[ActionCandidate[AgentAction]]:
        outputs = await asyncio.gather(
            *(
                self._policy.generate(
                    state.messages,
                    state.task.tools,
                    self._request.model_copy(update={"seed": seed + index}),
                )
                for index in range(branch_factor)
            )
        )
        deduplicated: dict[str, PolicyOutput] = {}
        for output in outputs:
            signature = RepeatedActionPenalty.signature(output.action)
            deduplicated.setdefault(signature, output)
        candidates = []
        for output in deduplicated.values():
            mean_logprob = (
                sum(output.policy_logprobs) / len(output.policy_logprobs)
                if output.policy_logprobs
                else 0.0
            )
            candidates.append(
                ActionCandidate(
                    action=output.action,
                    policy_prior=math.exp(max(mean_logprob, -30.0)),
                    metadata={
                        "generated_token_count": output.generated_token_count,
                        "model": output.model,
                    },
                )
            )
        return candidates


class AgentSearchDomain:
    def __init__(
        self,
        *,
        environment: AgentEnvironment,
        candidate_generator: PolicyCandidateGenerator,
        process_reward_model: ProcessRewardModel,
        max_depth: int = 8,
        tool_failure_penalty: float = -0.1,
    ) -> None:
        self._environment = environment
        self._candidate_generator = candidate_generator
        self._prm = process_reward_model
        self._max_depth = max_depth
        self._tool_failure_penalty = tool_failure_penalty

    async def start(self, task: TaskSpec) -> AgentSearchState:
        session_id = await self._environment.create_session(task)
        snapshot = await self._environment.snapshot(session_id)
        return AgentSearchState(
            session_id=session_id,
            task=task,
            messages=task.messages,
            snapshot=snapshot,
        )

    async def propose(
        self,
        state: AgentSearchState,
        *,
        branch_factor: int,
        seed: int,
    ) -> Sequence[ActionCandidate[AgentAction]]:
        return await self._candidate_generator.propose(
            state,
            branch_factor=branch_factor,
            seed=seed,
        )

    async def transition(
        self,
        state: AgentSearchState,
        action: AgentAction,
    ) -> SearchTransition[AgentSearchState]:
        await self._environment.restore(state.session_id, state.snapshot)
        tool_results: tuple[ToolResult, ...] = ()
        reward = 0.0
        if action.kind is ActionKind.TOOL:
            tool_results = await self._environment.execute(state.session_id, action.tool_calls)
            reward += self._tool_failure_penalty * sum(not result.ok for result in tool_results)
        messages = list(state.messages)
        self._append_messages(messages, action, tool_results)
        snapshot = await self._environment.snapshot(state.session_id)
        depth = state.depth + 1
        terminal = action.kind in {ActionKind.FINAL, ActionKind.INVALID} or depth >= self._max_depth
        next_state = AgentSearchState(
            session_id=state.session_id,
            task=state.task,
            messages=tuple(messages),
            snapshot=snapshot,
            depth=depth,
            action_history=(*state.action_history, action),
            latest_tool_results=tool_results,
            terminal=terminal,
        )
        if action.kind is ActionKind.INVALID:
            reward -= 0.25
        return SearchTransition(state=next_state, reward=reward, terminal=terminal)

    async def evaluate(self, states: Sequence[AgentSearchState]) -> tuple[ProcessScore, ...]:
        scores: list[ProcessScore | None] = [None] * len(states)
        pending_inputs = []
        pending_indices = []
        for index, state in enumerate(states):
            terminal_score = self._terminal_score(state)
            if terminal_score is not None:
                scores[index] = terminal_score
                continue
            pending_indices.append(index)
            pending_inputs.append(
                ProcessRewardInput(
                    state={
                        "task_id": state.task.task_id,
                        "depth": state.depth,
                        "messages": [message.model_dump(mode="json") for message in state.messages],
                        "tool_results": [
                            result.model_dump(mode="json") for result in state.latest_tool_results
                        ],
                    },
                    action=(
                        state.action_history[-1].model_dump(mode="json")
                        if state.action_history
                        else None
                    ),
                    history=tuple(
                        action.model_dump(mode="json") for action in state.action_history
                    ),
                )
            )
        if pending_inputs:
            pending_scores = await self._prm.score_batch(pending_inputs)
            if len(pending_scores) != len(pending_inputs):
                raise ValueError("PRM did not return one score per agent state")
            for index, score in zip(pending_indices, pending_scores, strict=True):
                scores[index] = score
        if any(score is None for score in scores):
            raise RuntimeError("agent state evaluation left unscored states")
        return tuple(score for score in scores if score is not None)

    async def close(self, state: AgentSearchState) -> None:
        await self._environment.close_session(state.session_id)

    @staticmethod
    def _append_messages(
        messages: list[Message],
        action: AgentAction,
        results: tuple[ToolResult, ...],
    ) -> None:
        content = action.raw_text or action.reasoning or action.final_answer or ""
        messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        for result in results:
            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=result.content,
                    name=result.name,
                    tool_call_id=result.call_id,
                    metadata={"ok": result.ok},
                )
            )

    @staticmethod
    def _terminal_score(state: AgentSearchState) -> ProcessScore | None:
        if not state.action_history:
            return None
        action = state.action_history[-1]
        if action.kind is ActionKind.INVALID:
            return ProcessScore(value=-1.0, confidence=1.0, rationale="invalid action")
        if action.kind is not ActionKind.FINAL:
            return None
        if state.task.verifier.kind != "exact_match":
            return None
        expected = state.task.verifier.config.get("answer")
        expected_values = [expected] if isinstance(expected, str) else expected
        if not isinstance(expected_values, list):
            return None
        normalized = {AgentSearchDomain._normalize(str(value)) for value in expected_values}
        matched = (
            action.final_answer is not None
            and AgentSearchDomain._normalize(action.final_answer) in normalized
        )
        return ProcessScore(
            value=1.0 if matched else -1.0,
            confidence=1.0,
            rationale="verified final answer",
            metadata={"matched": matched},
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


class AgentTreeSearchController:
    def __init__(
        self,
        domain: AgentSearchDomain,
        budget_allocator: AdaptiveComputeBudget,
        *,
        exploration: float = 1.5,
        discount: float = 1.0,
        prune_below: float = -0.8,
    ) -> None:
        self._domain = domain
        self._budget_allocator = budget_allocator
        self._exploration = exploration
        self._discount = discount
        self._prune_below = prune_below

    async def choose(
        self,
        state: AgentSearchState,
        signals: BudgetSignals,
        *,
        seed: int = 0,
    ) -> MCTSResult[AgentAction]:
        budget = self._budget_allocator.allocate(signals)
        search = AsyncMCTS(
            self._domain,
            MCTSConfig(
                simulations=budget.simulations,
                branch_factor=budget.branch_factor,
                max_depth=budget.max_depth,
                exploration=self._exploration,
                discount=self._discount,
                prune_below=self._prune_below,
            ),
        )
        return await search.search(state, seed=seed)

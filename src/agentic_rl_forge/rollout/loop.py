from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from agentic_rl_forge.contracts import (
    ActionKind,
    DataOrigin,
    JsonObject,
    Message,
    MessageRole,
    Provenance,
    TaskSpec,
    ToolResult,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    new_id,
)
from agentic_rl_forge.environments import AgentEnvironment
from agentic_rl_forge.rewards import FinalRewardContext, RewardEngine, StepRewardContext
from agentic_rl_forge.rewards.components import RepeatedActionPenalty
from agentic_rl_forge.rollout.policy import AgentPolicy, GenerationRequest


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    max_steps: int | None = None
    max_tokens_per_step: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    repeated_action_limit: int = 3
    snapshot_each_step: bool = True
    close_session: bool = True


class AgentLoop:
    def __init__(
        self,
        *,
        policy: AgentPolicy,
        environment: AgentEnvironment,
        rewards: RewardEngine,
        config: RolloutConfig | None = None,
        token_counter: Callable[[str], int] = approximate_token_count,
    ) -> None:
        self._policy = policy
        self._environment = environment
        self._rewards = rewards
        self._config = config or RolloutConfig()
        self._token_counter = token_counter

    async def run(
        self,
        task: TaskSpec,
        *,
        group_id: str | None = None,
        seed: int | None = None,
        trajectory_id: str | None = None,
        provenance_metadata: JsonObject | None = None,
    ) -> Trajectory:
        metadata = dict(provenance_metadata or {})
        if "session_id" in metadata:
            raise ValueError("session_id is reserved provenance metadata")
        session_id = await self._environment.create_session(task)
        started_at = datetime.now(timezone.utc)
        messages = list(task.messages)
        steps: list[TrajectoryStep] = []
        status = TrajectoryStatus.RUNNING
        error: str | None = None
        max_steps = min(task.max_steps, self._config.max_steps or task.max_steps)
        try:
            for index in range(max_steps):
                snapshot_before = (
                    await self._environment.snapshot(session_id)
                    if self._config.snapshot_each_step
                    else None
                )
                output = await self._policy.generate(
                    tuple(messages),
                    task.tools,
                    GenerationRequest(
                        max_tokens=self._config.max_tokens_per_step,
                        temperature=self._config.temperature,
                        top_p=self._config.top_p,
                        seed=None if seed is None else seed + index,
                    ),
                )
                tool_results: tuple[ToolResult, ...] = ()
                if output.action.kind is ActionKind.TOOL:
                    tool_results = await self._environment.execute(
                        session_id, output.action.tool_calls
                    )
                observation_tokens = sum(
                    self._token_counter(result.content) for result in tool_results
                )
                step_rewards = await self._rewards.score_step(
                    StepRewardContext(
                        task=task,
                        prior_steps=tuple(steps),
                        action=output.action,
                        tool_results=tool_results,
                        generated_token_count=output.generated_token_count,
                    )
                )
                snapshot_after = (
                    await self._environment.snapshot(session_id)
                    if self._config.snapshot_each_step
                    else None
                )
                step = TrajectoryStep(
                    index=index,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    input_messages=tuple(messages),
                    action=output.action,
                    tool_results=tool_results,
                    rewards=step_rewards,
                    generated_token_count=output.generated_token_count,
                    observation_token_count=observation_tokens,
                    generated_token_mask=(1,) * output.generated_token_count
                    + (0,) * observation_tokens,
                    policy_logprobs=output.policy_logprobs,
                    snapshot_before=snapshot_before,
                    snapshot_after=snapshot_after,
                    metadata={"finish_reason": output.finish_reason, "model": output.model},
                )
                steps.append(step)
                self._append_messages(messages, step)
                if output.action.kind is ActionKind.FINAL:
                    status = TrajectoryStatus.FAILED
                    break
                if output.action.kind is ActionKind.INVALID:
                    status = TrajectoryStatus.FAILED
                    break
                if self._repeated_action_count(steps) >= self._config.repeated_action_limit:
                    status = TrajectoryStatus.TRUNCATED
                    break
            else:
                status = TrajectoryStatus.TRUNCATED
        except Exception as exception:
            status = TrajectoryStatus.ERROR
            error = f"{type(exception).__name__}: {exception}"
        provisional_status = status
        final_reward = await self._rewards.score_final(
            FinalRewardContext(task=task, steps=tuple(steps), status=provisional_status)
        )
        if steps and steps[-1].action.kind is ActionKind.FINAL:
            status = (
                TrajectoryStatus.SUCCEEDED
                if self._rewards.is_success(final_reward)
                else TrajectoryStatus.FAILED
            )
        completed_at = datetime.now(timezone.utc)
        if self._config.close_session:
            await self._environment.close_session(session_id)
        return Trajectory(
            trajectory_id=trajectory_id or new_id("traj"),
            task_id=task.task_id,
            group_id=group_id or new_id("group"),
            policy_version=self._policy.version,
            environment_version=self._environment.version,
            provenance=Provenance(
                origin=DataOrigin.ON_POLICY,
                producer="agent-loop",
                producer_version=self._policy.version,
                metadata={"session_id": session_id, **metadata},
            ),
            status=status,
            steps=tuple(steps),
            final_reward=final_reward,
            started_at=started_at,
            completed_at=completed_at,
            metadata={"error": error} if error else {},
        )

    @staticmethod
    def _append_messages(messages: list[Message], step: TrajectoryStep) -> None:
        action = step.action
        assistant_content = action.raw_text or action.reasoning
        if action.kind is ActionKind.FINAL and action.final_answer:
            assistant_content = action.raw_text or action.final_answer
        metadata = {}
        if action.kind is ActionKind.TOOL:
            metadata["tool_calls"] = [
                {
                    "id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in action.tool_calls
            ]
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=assistant_content,
                metadata=metadata,
            )
        )
        for result in step.tool_results:
            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    name=result.name,
                    tool_call_id=result.call_id,
                    content=result.content,
                    metadata={"ok": result.ok, "error_code": result.error_code},
                )
            )

    @staticmethod
    def _repeated_action_count(steps: list[TrajectoryStep]) -> int:
        if not steps:
            return 0
        signature = RepeatedActionPenalty.signature(steps[-1].action)
        count = 0
        for step in reversed(steps):
            if RepeatedActionPenalty.signature(step.action) != signature:
                break
            count += 1
        return count

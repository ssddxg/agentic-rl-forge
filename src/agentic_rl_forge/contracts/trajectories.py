from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject, new_id, utc_now
from agentic_rl_forge.contracts.messages import Message
from agentic_rl_forge.contracts.provenance import DataOrigin, Provenance
from agentic_rl_forge.contracts.rewards import RewardSummary
from agentic_rl_forge.contracts.tools import ToolCall, ToolResult


class ActionKind(str, Enum):
    TOOL = "tool"
    FINAL = "final"
    INVALID = "invalid"


class TrajectoryStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TRUNCATED = "truncated"
    ERROR = "error"
    REJECTED = "rejected"


class AgentAction(ContractModel):
    kind: ActionKind
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    final_answer: str | None = None
    raw_text: str = ""
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action(self) -> AgentAction:
        if self.kind is ActionKind.TOOL:
            if not self.tool_calls:
                raise ValueError("tool actions require at least one tool call")
            if self.final_answer is not None:
                raise ValueError("tool actions cannot include a final answer")
        elif self.kind is ActionKind.FINAL:
            if self.final_answer is None:
                raise ValueError("final actions require final_answer")
            if self.tool_calls:
                raise ValueError("final actions cannot include tool calls")
        elif self.tool_calls or self.final_answer is not None:
            raise ValueError("invalid actions cannot include parsed calls or final answers")
        return self


class EnvironmentSnapshot(ContractModel):
    snapshot_id: str = Field(default_factory=lambda: new_id("snapshot"))
    environment_id: str
    state_digest: str
    restorable: bool = True
    storage_uri: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class TrajectoryStep(ContractModel):
    index: int = Field(ge=0)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    input_messages: tuple[Message, ...]
    action: AgentAction
    tool_results: tuple[ToolResult, ...] = ()
    rewards: RewardSummary = Field(default_factory=RewardSummary)
    generated_token_count: int = Field(default=0, ge=0)
    observation_token_count: int = Field(default=0, ge=0)
    generated_token_mask: tuple[int, ...] = ()
    policy_logprobs: tuple[float, ...] = ()
    snapshot_before: EnvironmentSnapshot | None = None
    snapshot_after: EnvironmentSnapshot | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_step(self) -> TrajectoryStep:
        if self.action.kind is ActionKind.TOOL:
            expected = {call.call_id for call in self.action.tool_calls}
            received = {result.call_id for result in self.tool_results}
            if received != expected:
                raise ValueError("tool result call_ids must exactly match tool call_ids")
        elif self.tool_results:
            raise ValueError("only tool actions can have tool results")
        if self.generated_token_mask and len(self.generated_token_mask) != (
            self.generated_token_count + self.observation_token_count
        ):
            raise ValueError("generated_token_mask length must match all emitted tokens")
        if self.policy_logprobs and len(self.policy_logprobs) != self.generated_token_count:
            raise ValueError("policy_logprobs length must match generated_token_count")
        return self


class Trajectory(ContractModel):
    trajectory_id: str = Field(default_factory=lambda: new_id("traj"))
    task_id: str
    group_id: str
    policy_version: str = Field(min_length=1)
    environment_version: str = Field(min_length=1)
    provenance: Provenance
    status: TrajectoryStatus
    steps: tuple[TrajectoryStep, ...]
    final_reward: RewardSummary = Field(default_factory=RewardSummary)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trajectory(self) -> Trajectory:
        indices = [step.index for step in self.steps]
        if indices != list(range(len(indices))):
            raise ValueError("trajectory step indices must be contiguous and start at zero")
        if self.status is TrajectoryStatus.RUNNING and self.completed_at is not None:
            raise ValueError("running trajectories cannot have completed_at")
        if self.status is not TrajectoryStatus.RUNNING and self.completed_at is None:
            raise ValueError("completed trajectories require completed_at")
        if self.status is TrajectoryStatus.SUCCEEDED and (
            not self.steps or self.steps[-1].action.kind is not ActionKind.FINAL
        ):
            raise ValueError("successful trajectories must end with a final action")
        return self

    @property
    def total_reward(self) -> float:
        step_reward = sum(step.rewards.total for step in self.steps)
        return step_reward + self.final_reward.total

    @property
    def total_generated_tokens(self) -> int:
        return sum(step.generated_token_count for step in self.steps)

    @property
    def total_observation_tokens(self) -> int:
        return sum(step.observation_token_count for step in self.steps)

    @property
    def is_on_policy(self) -> bool:
        return self.provenance.origin is DataOrigin.ON_POLICY

    def require_on_policy(self, expected_policy_version: str) -> None:
        if not self.is_on_policy:
            raise ValueError(f"trajectory {self.trajectory_id} is not on-policy data")
        if self.policy_version != expected_policy_version:
            raise ValueError(
                f"trajectory policy version {self.policy_version!r} does not match "
                f"{expected_policy_version!r}"
            )

from __future__ import annotations

from pydantic import Field, PositiveInt, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject
from agentic_rl_forge.contracts.messages import Message, MessageRole
from agentic_rl_forge.contracts.tools import ToolSpec


class VerifierSpec(ContractModel):
    kind: str = Field(min_length=1)
    config: JsonObject = Field(default_factory=dict)
    private: bool = True


class TaskSpec(ContractModel):
    task_id: str = Field(min_length=1)
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    verifier: VerifierSpec
    max_steps: PositiveInt = 16
    max_generated_tokens: PositiveInt = 8192
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task(self) -> TaskSpec:
        if not self.messages:
            raise ValueError("task requires at least one message")
        if not any(message.role is MessageRole.USER for message in self.messages):
            raise ValueError("task requires a user message")
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool names must be unique within a task")
        return self

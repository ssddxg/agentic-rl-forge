from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(ContractModel):
    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_message(self) -> Message:
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role is not MessageRole.TOOL and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        return self

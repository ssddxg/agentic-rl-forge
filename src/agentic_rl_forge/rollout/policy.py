from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from pydantic import Field, PositiveInt

from agentic_rl_forge.contracts import AgentAction, ContractModel, Message, ToolSpec


class GenerationRequest(ContractModel):
    max_tokens: PositiveInt = 1024
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None


class PolicyOutput(ContractModel):
    action: AgentAction
    generated_token_count: int = Field(default=0, ge=0)
    policy_logprobs: tuple[float, ...] = ()
    finish_reason: str = "stop"
    model: str = ""

    def model_post_init(self, __context: object) -> None:
        if self.policy_logprobs and len(self.policy_logprobs) != self.generated_token_count:
            raise ValueError("policy_logprobs length must match generated_token_count")


class AgentPolicy(Protocol):
    @property
    def version(self) -> str: ...

    async def generate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        request: GenerationRequest,
    ) -> PolicyOutput: ...


class ScriptedPolicy:
    def __init__(self, outputs: Sequence[PolicyOutput], *, version: str = "scripted-v1") -> None:
        self._outputs = list(outputs)
        self._version = version
        self._index = 0
        self._lock = asyncio.Lock()

    @property
    def version(self) -> str:
        return self._version

    async def generate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        request: GenerationRequest,
    ) -> PolicyOutput:
        del messages, tools, request
        async with self._lock:
            if self._index >= len(self._outputs):
                raise RuntimeError("scripted policy has no remaining outputs")
            output = self._outputs[self._index]
            self._index += 1
            return output

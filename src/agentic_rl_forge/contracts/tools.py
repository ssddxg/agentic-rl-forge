from __future__ import annotations

from enum import Enum

from pydantic import Field, PositiveFloat, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject, new_id


class SideEffect(str, Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


class ToolSpec(ContractModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    description: str = Field(min_length=1)
    input_schema: JsonObject
    output_schema: JsonObject | None = None
    timeout_s: PositiveFloat = 30.0
    side_effect: SideEffect = SideEffect.NONE
    idempotent: bool = True
    tags: frozenset[str] = frozenset()
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_side_effects(self) -> ToolSpec:
        if self.side_effect in {SideEffect.WRITE, SideEffect.EXTERNAL} and self.idempotent:
            raise ValueError("write and external tools must explicitly be non-idempotent")
        return self


class ToolCall(ContractModel):
    call_id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: JsonObject = Field(default_factory=dict)


class ToolResult(ContractModel):
    call_id: str
    name: str
    content: str
    ok: bool
    error_code: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    state_digest: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_error(self) -> ToolResult:
        if self.ok and self.error_code is not None:
            raise ValueError("successful tool results cannot carry error_code")
        if not self.ok and not self.error_code:
            raise ValueError("failed tool results require error_code")
        return self

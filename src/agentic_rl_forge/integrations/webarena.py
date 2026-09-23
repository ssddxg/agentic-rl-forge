from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from agentic_rl_forge.contracts import (
    ContractModel,
    JsonObject,
    Message,
    MessageRole,
    SideEffect,
    TaskSpec,
    ToolSpec,
    VerifierSpec,
)


def webarena_tool_specs() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="browser.open",
            description="Open a URL in the current browser session.",
            input_schema=_object_schema(url={"type": "string", "minLength": 1}),
            side_effect=SideEffect.WRITE,
            idempotent=False,
            tags=frozenset({"browser", "navigation"}),
        ),
        ToolSpec(
            name="browser.click",
            description="Click an element identified by its accessibility-tree element ID.",
            input_schema=_object_schema(element_id={"type": "string", "minLength": 1}),
            side_effect=SideEffect.WRITE,
            idempotent=False,
            tags=frozenset({"browser", "interaction"}),
        ),
        ToolSpec(
            name="browser.type",
            description="Type text into an editable element.",
            input_schema={
                "type": "object",
                "properties": {
                    "element_id": {"type": "string", "minLength": 1},
                    "text": {"type": "string"},
                    "clear": {"type": "boolean", "default": True},
                },
                "required": ["element_id", "text"],
                "additionalProperties": False,
            },
            side_effect=SideEffect.WRITE,
            idempotent=False,
            tags=frozenset({"browser", "interaction"}),
        ),
        ToolSpec(
            name="browser.select",
            description="Select one option in a form control.",
            input_schema={
                "type": "object",
                "properties": {
                    "element_id": {"type": "string", "minLength": 1},
                    "value": {"type": "string"},
                },
                "required": ["element_id", "value"],
                "additionalProperties": False,
            },
            side_effect=SideEffect.WRITE,
            idempotent=False,
            tags=frozenset({"browser", "interaction"}),
        ),
        ToolSpec(
            name="browser.scroll",
            description="Scroll the current page by a signed pixel amount.",
            input_schema=_object_schema(delta_y={"type": "integer"}),
            side_effect=SideEffect.WRITE,
            idempotent=False,
            tags=frozenset({"browser", "navigation"}),
        ),
        ToolSpec(
            name="browser.back",
            description="Navigate to the previous page in browser history.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            side_effect=SideEffect.WRITE,
            idempotent=False,
            tags=frozenset({"browser", "navigation"}),
        ),
        ToolSpec(
            name="browser.observe",
            description="Return the current URL and accessibility tree without changing state.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            side_effect=SideEffect.READ,
            tags=frozenset({"browser", "observation"}),
        ),
    )


class WebArenaTaskRecord(ContractModel):
    task_id: int | str
    intent: str = Field(min_length=1)
    sites: tuple[str, ...] = ()
    start_url: str | None = None
    require_login: bool = False
    eval: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)

    def to_task_spec(self, *, max_steps: int = 30) -> TaskSpec:
        system = (
            "Complete the requested browser task using the available browser tools. "
            "Use element IDs from browser observations, verify state before consequential "
            "actions, and return a concise final answer when the task is complete."
        )
        return TaskSpec(
            task_id=f"webarena-{self.task_id}",
            messages=(
                Message(role=MessageRole.SYSTEM, content=system),
                Message(role=MessageRole.USER, content=self.intent),
            ),
            tools=webarena_tool_specs(),
            verifier=VerifierSpec(
                kind="webarena_remote",
                config={"task_id": self.task_id, "eval": self.eval},
                private=True,
            ),
            max_steps=max_steps,
            max_generated_tokens=32768,
            metadata={
                "sites": list(self.sites),
                "start_url": self.start_url,
                "require_login": self.require_login,
                **self.metadata,
            },
        )


def load_webarena_tasks(path: Path) -> tuple[WebArenaTaskRecord, ...]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, list):
        raise ValueError("WebArena task file must contain a JSON list")
    return tuple(WebArenaTaskRecord.model_validate(item) for item in payload)


def _object_schema(**properties: JsonObject) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }

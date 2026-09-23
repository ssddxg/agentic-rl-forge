import asyncio
import json

import pytest

from agentic_rl_forge.contracts import (
    Message,
    MessageRole,
    SideEffect,
    TaskSpec,
    ToolCall,
    ToolSpec,
    VerifierSpec,
)
from agentic_rl_forge.environments import (
    FunctionTool,
    InMemorySearchTool,
    LocalToolEnvironment,
    ToolContext,
)


def stateful_task(tool: ToolSpec) -> TaskSpec:
    return TaskSpec(
        task_id="stateful-1",
        messages=(Message(role=MessageRole.USER, content="Increment the counter."),),
        tools=(tool,),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "done"}),
    )


@pytest.mark.asyncio
async def test_in_memory_search_supports_chinese_queries() -> None:
    search = InMemorySearchTool(
        {
            "france": "巴黎是法国的首都.",
            "germany": "柏林是德国的首都.",
        }
    )

    result = await search.execute(
        {"query": "法国首都"},
        ToolContext(session_id="session", task_id="task", state={}),
    )

    payload = json.loads(result.content)
    assert search.search("法国首都") == payload
    assert payload[0]["id"] == "france"
    assert "巴黎" in payload[0]["content"]


@pytest.mark.asyncio
async def test_stateful_environment_snapshot_and_restore() -> None:
    spec = ToolSpec(
        name="increment",
        description="Increment a session counter.",
        input_schema={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
            "additionalProperties": False,
        },
        side_effect=SideEffect.WRITE,
        idempotent=False,
    )

    def increment(arguments: dict[str, object], context: ToolContext) -> dict[str, object]:
        amount = int(arguments["amount"])
        context.state["counter"] = int(context.state.get("counter", 0)) + amount
        return {"counter": context.state["counter"]}

    environment = LocalToolEnvironment(
        (FunctionTool(spec, increment),),
        initial_state_factory=lambda task: {"task": task.task_id, "counter": 0},
    )
    session_id = await environment.create_session(stateful_task(spec))
    initial = await environment.snapshot(session_id)

    result = await environment.execute(
        session_id,
        (ToolCall(call_id="call-1", name="increment", arguments={"amount": 3}),),
    )

    assert result[0].ok
    assert environment.session_state(session_id)["counter"] == 3
    await environment.restore(session_id, initial)
    assert environment.session_state(session_id)["counter"] == 0
    await environment.close_session(session_id)


@pytest.mark.asyncio
async def test_environment_rejects_invalid_arguments_without_executing() -> None:
    spec = ToolSpec(
        name="read",
        description="Read one value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        side_effect=SideEffect.READ,
    )
    executions = 0

    def read(arguments: dict[str, object], context: ToolContext) -> str:
        nonlocal executions
        del arguments, context
        executions += 1
        return "value"

    environment = LocalToolEnvironment((FunctionTool(spec, read),))
    session_id = await environment.create_session(stateful_task(spec))

    result = await environment.execute(
        session_id,
        (ToolCall(call_id="bad-call", name="read", arguments={"unknown": True}),),
    )

    assert not result[0].ok
    assert result[0].error_code == "invalid_arguments"
    assert executions == 0


@pytest.mark.asyncio
async def test_environment_handles_legacy_asyncio_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyAsyncioTimeoutError(Exception):
        pass

    async def raise_legacy_asyncio_timeout(
        awaitable: object,
        timeout: float | None = None,
    ) -> object:
        del timeout
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise LegacyAsyncioTimeoutError

    monkeypatch.setattr(asyncio, "TimeoutError", LegacyAsyncioTimeoutError)
    monkeypatch.setattr(asyncio, "wait_for", raise_legacy_asyncio_timeout)
    spec = ToolSpec(
        name="read",
        description="Read one value.",
        input_schema={"type": "object", "additionalProperties": False},
        side_effect=SideEffect.READ,
    )
    environment = LocalToolEnvironment((FunctionTool(spec, lambda arguments, context: "value"),))
    session_id = await environment.create_session(stateful_task(spec))

    result = await environment.execute(
        session_id,
        (ToolCall(call_id="timed-out-call", name="read", arguments={}),),
    )

    assert not result[0].ok
    assert result[0].error_code == "timeout"

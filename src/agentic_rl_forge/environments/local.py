from __future__ import annotations

import asyncio
import copy
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import orjson
from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate

from agentic_rl_forge.contracts import (
    EnvironmentSnapshot,
    JsonObject,
    SideEffect,
    TaskSpec,
    ToolCall,
    ToolResult,
    new_id,
)
from agentic_rl_forge.environments.base import AgentEnvironment
from agentic_rl_forge.environments.tools import ExecutableTool, ToolContext


@dataclass(slots=True)
class _Session:
    task: TaskSpec
    state: JsonObject
    revision: int = 0
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LocalToolEnvironment(AgentEnvironment):
    def __init__(
        self,
        tools: tuple[ExecutableTool, ...],
        *,
        version: str = "local-tools-v1",
        initial_state_factory: Callable[[TaskSpec], JsonObject] | None = None,
    ) -> None:
        names = [tool.spec.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("registered tool names must be unique")
        self._tools = {tool.spec.name: tool for tool in tools}
        self._version = version
        self._sessions: dict[str, _Session] = {}
        self._snapshots: dict[str, JsonObject] = {}
        self._initial_state_factory = initial_state_factory

    @property
    def version(self) -> str:
        return self._version

    async def create_session(self, task: TaskSpec) -> str:
        missing = {tool.name for tool in task.tools} - set(self._tools)
        if missing:
            raise ValueError(f"task references unregistered tools: {sorted(missing)}")
        state: JsonObject = {"revision": 0}
        if self._initial_state_factory is not None:
            generated = self._initial_state_factory(task)
            if not isinstance(generated, dict):
                raise TypeError("initial_state_factory must return a dictionary")
            state.update(copy.deepcopy(generated))
        session_id = new_id("session")
        self._sessions[session_id] = _Session(task=task, state=state)
        return session_id

    async def execute(self, session_id: str, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        session = self._get_session(session_id)
        if not calls:
            return ()
        tools = [self._tools.get(call.name) for call in calls]
        can_run_concurrently = all(
            tool is not None
            and tool.spec.idempotent
            and tool.spec.side_effect in {SideEffect.NONE, SideEffect.READ}
            for tool in tools
        )
        if can_run_concurrently:
            return tuple(
                await asyncio.gather(
                    *(self._execute_one(session_id, session, call) for call in calls)
                )
            )
        async with session.lock:
            results = []
            for call in calls:
                results.append(await self._execute_one(session_id, session, call))
            return tuple(results)

    async def snapshot(self, session_id: str) -> EnvironmentSnapshot:
        session = self._get_session(session_id)
        async with session.lock:
            state = copy.deepcopy(session.state)
            state_digest = self._state_digest(state)
            snapshot_id = new_id("snapshot")
            self._snapshots[snapshot_id] = state
        return EnvironmentSnapshot(
            snapshot_id=snapshot_id,
            environment_id=session_id,
            state_digest=state_digest,
            restorable=True,
            storage_uri=f"memory://{snapshot_id}",
            metadata={"revision": session.revision},
        )

    async def restore(self, session_id: str, snapshot: EnvironmentSnapshot) -> None:
        session = self._get_session(session_id)
        if snapshot.environment_id != session_id:
            raise ValueError("snapshot belongs to a different environment session")
        if not snapshot.restorable:
            raise ValueError("snapshot is not restorable")
        state = self._snapshots.get(snapshot.snapshot_id)
        if state is None:
            raise KeyError(f"unknown snapshot: {snapshot.snapshot_id}")
        if self._state_digest(state) != snapshot.state_digest:
            raise ValueError("snapshot digest does not match stored state")
        async with session.lock:
            session.state = copy.deepcopy(state)
            session.revision = int(session.state.get("revision", session.revision))

    async def close_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        async with session.lock:
            session.closed = True
        del self._sessions[session_id]

    def session_state(self, session_id: str) -> JsonObject:
        return copy.deepcopy(self._get_session(session_id).state)

    async def _execute_one(
        self,
        session_id: str,
        session: _Session,
        call: ToolCall,
    ) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return self._error_result(call, "unknown_tool", started)
        if call.name not in {spec.name for spec in session.task.tools}:
            return self._error_result(call, "tool_not_allowed", started)
        try:
            validate(instance=call.arguments, schema=tool.spec.input_schema)
        except JsonSchemaError as error:
            return self._error_result(
                call,
                "invalid_arguments",
                started,
                content=error.message,
            )
        context = ToolContext(
            session_id=session_id,
            task_id=session.task.task_id,
            state=session.state,
        )
        try:
            execution = await asyncio.wait_for(
                tool.execute(call.arguments, context),
                timeout=float(tool.spec.timeout_s),
            )
        except asyncio.TimeoutError:
            return self._error_result(call, "timeout", started)
        except Exception as error:
            return self._error_result(
                call,
                "execution_error",
                started,
                content=f"{type(error).__name__}: {error}",
            )
        if tool.spec.side_effect in {SideEffect.WRITE, SideEffect.EXTERNAL}:
            session.revision += 1
            session.state["revision"] = session.revision
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            content=execution.content,
            ok=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            state_digest=self._state_digest(session.state),
            metadata=execution.metadata,
        )

    def _get_session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            raise KeyError(f"unknown or closed session: {session_id}")
        return session

    @staticmethod
    def _state_digest(state: JsonObject) -> str:
        payload = orjson.dumps(state, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _error_result(
        call: ToolCall,
        code: str,
        started: float,
        *,
        content: str = "",
    ) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            content=content,
            ok=False,
            error_code=code,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

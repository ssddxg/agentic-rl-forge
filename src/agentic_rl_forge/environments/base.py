from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_rl_forge.contracts import (
    EnvironmentSnapshot,
    TaskSpec,
    ToolCall,
    ToolResult,
)


class AgentEnvironment(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def create_session(self, task: TaskSpec) -> str:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, session_id: str, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        raise NotImplementedError

    @abstractmethod
    async def snapshot(self, session_id: str) -> EnvironmentSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def restore(self, session_id: str, snapshot: EnvironmentSnapshot) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        raise NotImplementedError

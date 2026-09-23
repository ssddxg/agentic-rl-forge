from __future__ import annotations

from typing import Any

import httpx

from agentic_rl_forge.contracts import EnvironmentSnapshot, TaskSpec, ToolCall, ToolResult
from agentic_rl_forge.environments.base import AgentEnvironment


class RemoteToolEnvironment(AgentEnvironment):
    def __init__(
        self,
        *,
        base_url: str,
        version: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)
        self._owns_client = client is None
        self._base_url = base_url.rstrip("/")
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    async def create_session(self, task: TaskSpec) -> str:
        payload = await self._request(
            "POST",
            "/sessions",
            json={"task": task.model_dump(mode="json", exclude_none=True)},
        )
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("remote environment did not return a session_id")
        return session_id

    async def execute(self, session_id: str, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        payload = await self._request(
            "POST",
            f"/sessions/{session_id}/execute",
            json={"calls": [call.model_dump(mode="json") for call in calls]},
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("remote environment did not return tool results")
        parsed = tuple(ToolResult.model_validate(result) for result in results)
        if {result.call_id for result in parsed} != {call.call_id for call in calls}:
            raise ValueError("remote tool results do not match requested call IDs")
        return parsed

    async def snapshot(self, session_id: str) -> EnvironmentSnapshot:
        payload = await self._request("POST", f"/sessions/{session_id}/snapshots", json={})
        return EnvironmentSnapshot.model_validate(payload)

    async def restore(self, session_id: str, snapshot: EnvironmentSnapshot) -> None:
        await self._request(
            "POST",
            f"/sessions/{session_id}/restore",
            json={"snapshot": snapshot.model_dump(mode="json", exclude_none=True)},
        )

    async def close_session(self, session_id: str) -> None:
        await self._request("DELETE", f"/sessions/{session_id}")

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._client.request(method, f"{self._base_url}{path}", json=json)
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("remote environment response must be a JSON object")
        return payload

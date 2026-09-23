from __future__ import annotations

import asyncio
from typing import Any

import httpx
import orjson

from agentic_rl_forge.contracts import Message, MessageRole, ToolSpec
from agentic_rl_forge.rollout.parsers import NativeToolParser, ResponseParser
from agentic_rl_forge.rollout.policy import GenerationRequest, PolicyOutput


class OpenAICompatiblePolicy:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        version: str,
        parser: ResponseParser | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_concurrency: int = 64,
        max_retries: int = 3,
        native_tools: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._version = version
        self._parser = parser or NativeToolParser()
        self._max_retries = max_retries
        self._native_tools = native_tools
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)

    @property
    def version(self) -> str:
        return self._version

    async def generate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        request: GenerationRequest,
    ) -> PolicyOutput:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                self._message_payload(message, native_tools=self._native_tools)
                for message in messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "logprobs": True,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if tools and self._native_tools:
            payload["tools"] = [self._tool_payload(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        response_data = await self._post(payload)
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("model response does not contain choices")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        native_tool_calls = message.get("tool_calls")
        action = self._parser.parse(content, native_tool_calls)
        usage = response_data.get("usage", {})
        logprobs = self._extract_logprobs(choice)
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int):
            completion_tokens = len(logprobs)
        if logprobs and len(logprobs) != completion_tokens:
            logprobs = ()
        return PolicyOutput(
            action=action,
            generated_token_count=completion_tokens,
            policy_logprobs=logprobs,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            model=str(response_data.get("model") or self._model),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await self._client.post(
                        f"{self._base_url}/v1/chat/completions", json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict):
                        raise TypeError("model response must be a JSON object")
                    return data
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError):
                    if attempt >= self._max_retries:
                        raise
                    await asyncio.sleep(min(0.25 * (2**attempt), 4.0))
        raise RuntimeError("unreachable retry state")

    @staticmethod
    def _message_payload(message: Message, *, native_tools: bool) -> dict[str, Any]:
        if message.role is MessageRole.TOOL and not native_tools:
            return {
                "role": MessageRole.USER.value,
                "content": f"<information>{message.content}</information>",
            }
        payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.role is MessageRole.TOOL:
            payload["tool_call_id"] = message.tool_call_id
        if message.role is MessageRole.ASSISTANT and native_tools:
            calls = message.metadata.get("tool_calls")
            if isinstance(calls, list):
                payload["tool_calls"] = [
                    {
                        "id": str(call["id"]),
                        "type": "function",
                        "function": {
                            "name": str(call["name"]),
                            "arguments": orjson.dumps(call["arguments"]).decode("utf-8"),
                        },
                    }
                    for call in calls
                    if isinstance(call, dict) and {"id", "name", "arguments"}.issubset(call)
                ]
        return payload

    @staticmethod
    def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _extract_logprobs(choice: dict[str, Any]) -> tuple[float, ...]:
        content = (choice.get("logprobs") or {}).get("content") or []
        values = []
        for item in content:
            value = item.get("logprob") if isinstance(item, dict) else None
            if isinstance(value, int | float):
                values.append(float(value))
        return tuple(values)

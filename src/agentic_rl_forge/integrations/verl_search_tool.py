from __future__ import annotations

import time
from typing import Any

import httpx
import orjson
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)


class AgenticRLForgeSearchTool(BaseTool):
    def __init__(self, config: dict[str, Any], tool_schema: Any = None) -> None:
        schema = tool_schema or OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search",
                description="Retrieve ranked passages for a natural-language query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A concise standalone search query.",
                        }
                    },
                    "required": ["query"],
                },
            ),
        )
        super().__init__(config, schema)
        self._endpoint = str(config["retrieval_service_url"])
        self._top_k = int(config.get("top_k", 3))
        self._timeout_s = float(config.get("timeout", 30.0))

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del instance_id, kwargs
        started = time.perf_counter()
        query = str(parameters.get("query", "")).strip()
        if not query:
            return (
                ToolResponse(text="Search query cannot be empty."),
                -0.1,
                {"error": "empty_query"},
            )
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                self._endpoint,
                json={"queries": [query], "topk": self._top_k, "return_scores": True},
            )
            response.raise_for_status()
            payload = response.json()
        results = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results or not isinstance(results[0], list):
            raise ValueError("retrieval service returned an invalid Search-R1 response")
        passages = results[0]
        text = orjson.dumps(passages).decode("utf-8")
        return (
            ToolResponse(text=text),
            0.0,
            {
                "query": query,
                "result_count": len(passages),
                "latency_ms": (time.perf_counter() - started) * 1000,
            },
        )

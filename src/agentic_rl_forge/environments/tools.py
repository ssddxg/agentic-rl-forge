from __future__ import annotations

import ast
import inspect
import math
import operator
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import httpx
import orjson

from agentic_rl_forge.contracts import JsonObject, SideEffect, ToolSpec
from agentic_rl_forge.search.tokenization import tokenize_text


@dataclass(slots=True)
class ToolContext:
    session_id: str
    task_id: str
    state: JsonObject
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolExecution:
    content: str
    metadata: JsonObject = field(default_factory=dict)


ToolReturn = ToolExecution | str | int | float | bool | JsonObject | list[Any] | None
ToolHandler = Callable[[JsonObject, ToolContext], ToolReturn | Awaitable[ToolReturn]]


class ExecutableTool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def execute(self, arguments: JsonObject, context: ToolContext) -> ToolExecution: ...


class FunctionTool:
    def __init__(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._spec = spec
        self._handler = handler

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, arguments: JsonObject, context: ToolContext) -> ToolExecution:
        result = self._handler(arguments, context)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolExecution):
            return result
        if isinstance(result, str):
            return ToolExecution(content=result)
        return ToolExecution(content=orjson.dumps(result).decode("utf-8"))


class InMemorySearchTool(FunctionTool):
    def __init__(self, documents: Mapping[str, str], *, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._documents = dict(documents)
        self._top_k = top_k
        spec = ToolSpec(
            name="search",
            description="Search a local document collection and return ranked passages.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "array"},
            side_effect=SideEffect.READ,
            tags=frozenset({"search", "local"}),
        )
        super().__init__(spec, self._search)

    def search(self, query: str) -> list[JsonObject]:
        """Search the configured documents without constructing a tool context."""
        if not query.strip():
            raise ValueError("query must not be empty")
        query_terms = self._terms(query)
        ranked: list[tuple[float, str, str]] = []
        for document_id, content in self._documents.items():
            terms = self._terms(content)
            overlap = query_terms & terms
            if not overlap:
                continue
            coverage = len(overlap) / max(len(query_terms), 1)
            density = len(overlap) / max(math.sqrt(len(terms)), 1.0)
            ranked.append((coverage + 0.1 * density, document_id, content))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"id": document_id, "score": round(score, 6), "content": content}
            for score, document_id, content in ranked[: self._top_k]
        ]

    def _search(self, arguments: JsonObject, context: ToolContext) -> ToolExecution:
        del context
        query = str(arguments["query"])
        results = self.search(query)
        return ToolExecution(
            content=orjson.dumps(results).decode("utf-8"),
            metadata={"query": query, "result_count": len(results)},
        )

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(tokenize_text(text))


class HTTPRetrievalTool(FunctionTool):
    def __init__(
        self,
        endpoint: str,
        *,
        top_k: int = 3,
        timeout_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._endpoint = endpoint
        self._top_k = top_k
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        spec = ToolSpec(
            name="search",
            description="Search an external retrieval service and return ranked passages.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "array"},
            timeout_s=timeout_s,
            side_effect=SideEffect.READ,
            tags=frozenset({"search", "remote", "search-r1"}),
        )
        super().__init__(spec, self._search)

    async def _search(self, arguments: JsonObject, context: ToolContext) -> ToolExecution:
        del context
        query = str(arguments["query"])
        response = await self._client.post(
            self._endpoint,
            json={"queries": [query], "topk": self._top_k, "return_scores": True},
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], list):
            raise ValueError("retrieval service returned an invalid Search-R1 response")
        return ToolExecution(
            content=orjson.dumps(results[0]).decode("utf-8"),
            metadata={"query": query, "result_count": len(results[0])},
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class CalculatorTool(FunctionTool):
    _binary_ops: ClassVar[dict[type[ast.operator], Callable[[float, float], float]]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_ops: ClassVar[dict[type[ast.unaryop], Callable[[float], float]]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self) -> None:
        spec = ToolSpec(
            name="calculator",
            description="Evaluate a bounded arithmetic expression.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string", "minLength": 1}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            side_effect=SideEffect.NONE,
            tags=frozenset({"math", "local"}),
        )
        super().__init__(spec, self._calculate)

    def _calculate(self, arguments: JsonObject, context: ToolContext) -> str:
        del context
        expression = str(arguments["expression"])
        tree = ast.parse(expression, mode="eval")
        value = self._evaluate(tree.body, depth=0)
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("result is outside the supported numeric range")
        return format(value, ".15g")

    def _evaluate(self, node: ast.AST, *, depth: int) -> float:
        if depth > 16:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary_ops:
            left = self._evaluate(node.left, depth=depth + 1)
            right = self._evaluate(node.right, depth=depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent is outside the supported range")
            return self._binary_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_ops:
            return self._unary_ops[type(node.op)](self._evaluate(node.operand, depth=depth + 1))
        raise ValueError("expression contains an unsupported operation")

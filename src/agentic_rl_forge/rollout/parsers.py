from __future__ import annotations

import re
from typing import Any, Protocol

import orjson

from agentic_rl_forge.contracts import ActionKind, AgentAction, ToolCall, new_id


class ResponseParser(Protocol):
    def parse(
        self,
        text: str,
        native_tool_calls: list[dict[str, Any]] | None = None,
    ) -> AgentAction: ...


class SearchR1Parser:
    _think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    _search_pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
    _answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def parse(
        self,
        text: str,
        native_tool_calls: list[dict[str, Any]] | None = None,
    ) -> AgentAction:
        if native_tool_calls:
            return NativeToolParser(fallback=self).parse(text, native_tool_calls)
        reasoning = "\n".join(match.strip() for match in self._think_pattern.findall(text))
        search = self._search_pattern.search(text)
        answer = self._answer_pattern.search(text)
        if search is not None and (answer is None or search.start() < answer.start()):
            query = search.group(1).strip()
            if not query:
                return AgentAction(kind=ActionKind.INVALID, reasoning=reasoning, raw_text=text)
            return AgentAction(
                kind=ActionKind.TOOL,
                reasoning=reasoning,
                tool_calls=(ToolCall(name="search", arguments={"query": query}),),
                raw_text=text,
            )
        if answer is not None:
            final_answer = answer.group(1).strip()
            if final_answer:
                return AgentAction(
                    kind=ActionKind.FINAL,
                    reasoning=reasoning,
                    final_answer=final_answer,
                    raw_text=text,
                )
        return AgentAction(kind=ActionKind.INVALID, reasoning=reasoning, raw_text=text)


class NativeToolParser:
    def __init__(self, *, fallback: ResponseParser | None = None) -> None:
        self._fallback = fallback

    def parse(
        self,
        text: str,
        native_tool_calls: list[dict[str, Any]] | None = None,
    ) -> AgentAction:
        if not native_tool_calls:
            if self._fallback is not None:
                return self._fallback.parse(text)
            stripped = text.strip()
            if stripped:
                return AgentAction(
                    kind=ActionKind.FINAL,
                    final_answer=stripped,
                    raw_text=text,
                )
            return AgentAction(kind=ActionKind.INVALID, raw_text=text)
        calls: list[ToolCall] = []
        try:
            for raw_call in native_tool_calls:
                function = raw_call.get("function", {})
                name = str(function["name"])
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = orjson.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must decode to an object")
                calls.append(
                    ToolCall(
                        call_id=str(raw_call.get("id") or new_id("call")),
                        name=name,
                        arguments=arguments,
                    )
                )
        except (KeyError, TypeError, orjson.JSONDecodeError):
            return AgentAction(kind=ActionKind.INVALID, reasoning=text, raw_text=text)
        return AgentAction(
            kind=ActionKind.TOOL,
            reasoning=text,
            tool_calls=tuple(calls),
            raw_text=text,
        )

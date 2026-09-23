from __future__ import annotations

import hashlib
from collections.abc import Iterable

import orjson

from agentic_rl_forge.contracts import (
    Message,
    MessageRole,
    SideEffect,
    TaskSpec,
    ToolSpec,
    VerifierSpec,
)

SEARCH_R1_SYSTEM_PROMPT = """Answer the question by reasoning and searching when needed.
Use <think>...</think> for reasoning, <search>...</search> for a search query, and
<answer>...</answer> for the final answer. Search results will be returned inside tool
messages. Do not place a final answer before you have sufficient evidence."""


def search_tool_spec(*, timeout_s: float = 30.0) -> ToolSpec:
    return ToolSpec(
        name="search",
        description="Retrieve the most relevant passages for a natural-language query.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={"type": "array"},
        timeout_s=timeout_s,
        side_effect=SideEffect.READ,
        tags=frozenset({"search", "search-r1"}),
    )


def build_search_r1_task(
    *,
    task_id: str,
    question: str,
    answers: str | list[str],
    max_searches: int = 4,
    metadata: dict[str, object] | None = None,
) -> TaskSpec:
    if max_searches < 1:
        raise ValueError("max_searches must be positive")
    return TaskSpec(
        task_id=task_id,
        messages=(
            Message(role=MessageRole.SYSTEM, content=SEARCH_R1_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=question),
        ),
        tools=(search_tool_spec(),),
        verifier=VerifierSpec(
            kind="exact_match",
            config={"answer": answers},
            private=True,
        ),
        max_steps=max_searches + 1,
        metadata=metadata or {},
    )


def iter_search_r1_tasks(
    records: Iterable[dict[str, object]],
    *,
    dataset_name: str,
    question_field: str = "question",
    answer_field: str = "answer",
) -> Iterable[TaskSpec]:
    for index, record in enumerate(records):
        question = record.get(question_field)
        answers = record.get(answer_field)
        if not isinstance(question, str):
            raise ValueError(f"record {index} does not contain a string question")
        if not isinstance(answers, str | list):
            raise ValueError(f"record {index} does not contain an answer")
        if isinstance(answers, list) and not all(isinstance(item, str) for item in answers):
            raise ValueError(f"record {index} contains a non-string answer")
        source_id = record.get("id", index)
        task_digest = hashlib.sha256(
            orjson.dumps(
                {"dataset": dataset_name, "id": source_id, "question": question},
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest()[:16]
        yield build_search_r1_task(
            task_id=f"{dataset_name}-{task_digest}",
            question=question,
            answers=answers,
            metadata={"dataset": dataset_name, "source_id": source_id},
        )

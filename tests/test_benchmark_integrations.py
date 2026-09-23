import json

import httpx
import pytest

from agentic_rl_forge.contracts import ToolCall
from agentic_rl_forge.environments import RemoteToolEnvironment
from agentic_rl_forge.integrations import (
    WebArenaTaskRecord,
    build_search_r1_task,
    iter_search_r1_tasks,
)
from agentic_rl_forge.services import BM25Index, Document


def test_search_r1_task_builder_and_dataset_ids_are_deterministic() -> None:
    task = build_search_r1_task(
        task_id="nq-1",
        question="What is the capital of France?",
        answers=["Paris"],
    )
    records = [{"id": "1", "question": "What is the capital of France?", "answer": "Paris"}]
    first = tuple(iter_search_r1_tasks(records, dataset_name="nq"))
    second = tuple(iter_search_r1_tasks(records, dataset_name="nq"))

    assert task.tools[0].name == "search"
    assert task.max_steps == 5
    assert first[0].task_id == second[0].task_id
    assert first[0].metadata["dataset"] == "nq"


def test_webarena_record_builds_stateful_browser_tools() -> None:
    task = WebArenaTaskRecord(
        task_id=7,
        intent="Find the latest order and report its status.",
        sites=("shopping_admin",),
        start_url="http://shopping.local/admin",
        eval={"eval_type": "string_match"},
    ).to_task_spec(max_steps=20)

    assert task.task_id == "webarena-7"
    assert task.verifier.kind == "webarena_remote"
    assert len(task.tools) >= 6
    click = next(tool for tool in task.tools if tool.name == "browser.click")
    assert not click.idempotent
    assert task.metadata["sites"] == ["shopping_admin"]


def test_bm25_retriever_ranks_relevant_document_first() -> None:
    index = BM25Index(
        (
            Document("fr", "Paris is the capital city of France."),
            Document("de", "Berlin is the capital city of Germany."),
            Document("misc", "A recipe for bread and soup."),
        )
    )

    results = index.search("France capital", top_k=2)

    assert results[0]["document"]["id"] == "fr"
    assert results[0]["score"] > 0


@pytest.mark.asyncio
async def test_remote_environment_lifecycle_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "session-1"})
        if request.url.path == "/sessions/session-1/execute":
            payload = json.loads(request.content)
            call = payload["calls"][0]
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "call_id": call["call_id"],
                            "name": call["name"],
                            "content": "[]",
                            "ok": True,
                        }
                    ]
                },
            )
        if request.url.path == "/sessions/session-1/snapshots":
            return httpx.Response(
                200,
                json={
                    "snapshot_id": "snapshot-1",
                    "environment_id": "session-1",
                    "state_digest": "abc",
                    "restorable": True,
                },
            )
        if request.url.path == "/sessions/session-1/restore":
            return httpx.Response(204)
        if request.method == "DELETE" and request.url.path == "/sessions/session-1":
            return httpx.Response(204)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    environment = RemoteToolEnvironment(
        base_url="http://environment.local",
        version="remote-v1",
        client=client,
    )
    task = build_search_r1_task(task_id="q", question="Question", answers="Answer")

    session_id = await environment.create_session(task)
    result = await environment.execute(
        session_id,
        (ToolCall(call_id="call-1", name="search", arguments={"query": "q"}),),
    )
    snapshot = await environment.snapshot(session_id)
    await environment.restore(session_id, snapshot)
    await environment.close_session(session_id)

    assert result[0].ok
    assert snapshot.snapshot_id == "snapshot-1"
    assert environment.version == "remote-v1"
    await client.aclose()

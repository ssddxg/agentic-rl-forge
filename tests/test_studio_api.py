from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentic_rl_forge.studio.api import StudioServices
from agentic_rl_forge.studio.app import create_studio_app
from agentic_rl_forge.studio.indexes import StudioIndexManager
from agentic_rl_forge.studio.ingestion import DocumentIngestor
from agentic_rl_forge.studio.models import ChatAnswer, Citation, SearchHit
from agentic_rl_forge.studio.paths import StudioPaths
from agentic_rl_forge.studio.repository import StudioRepository
from agentic_rl_forge.studio.security import LocalSessionGuard


class _FakeChatClient:
    def __init__(self, api_key: str | None, seen_keys: list[str | None]) -> None:
        self._api_key = api_key
        self._seen_keys = seen_keys
        self.closed = False

    async def chat(
        self,
        question: str,
        hits: Sequence[SearchHit],
        *,
        system_prompt: str | None = None,
    ) -> ChatAnswer:
        del question, system_prompt
        self._seen_keys.append(self._api_key)
        hit = hits[0]
        return ChatAnswer(
            content="巴黎是法国的首都。[1]",
            citations=(
                Citation(
                    index=1,
                    document_id=hit.document_id,
                    source_id=hit.source_id,
                    source_name=hit.source_name,
                    excerpt=hit.contents,
                ),
            ),
            model="mock-model",
        )

    async def test_connection(self) -> str:
        self._seen_keys.append(self._api_key)
        return "mock-model"

    async def close(self) -> None:
        self.closed = True


def _services(
    tmp_path: Path,
    *,
    max_file_bytes: int = 1024 * 1024,
    max_total_bytes: int = 2 * 1024 * 1024 * 1024,
) -> tuple[StudioServices, list[str | None]]:
    paths = StudioPaths(tmp_path / "studio").ensure()
    repository = StudioRepository(paths.database)
    ingestor = DocumentIngestor(
        paths,
        repository,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    indexes = StudioIndexManager(paths, repository)
    seen_keys: list[str | None] = []

    def factory(_: Any, api_key: str | None) -> _FakeChatClient:
        return _FakeChatClient(api_key, seen_keys)

    return (
        StudioServices(
            repository=repository,
            ingestor=ingestor,
            indexes=indexes,
            chat_client_factory=factory,
            session_guard=LocalSessionGuard(),
        ),
        seen_keys,
    )


async def _bootstrap(client: httpx.AsyncClient, *, prefix: str = "/api/v1") -> str:
    response = await client.get(f"{prefix}/bootstrap")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.cookies.get("arf_studio_session")
    return str(response.json()["security"]["csrfToken"])


async def _wait_for_job(client: httpx.AsyncClient, job_id: str) -> dict[str, object]:
    for _ in range(200):
        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload  # type: ignore[no-any-return]
        await asyncio.sleep(0.01)
    pytest.fail("Studio ingestion job did not finish")


@pytest.mark.asyncio
async def test_studio_api_real_crud_upload_search_answer_and_delete(tmp_path: Path) -> None:
    services, seen_keys = _services(tmp_path)
    app = create_studio_app(services=services)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client,
    ):
        csrf = await _bootstrap(client, prefix="/api")
        headers = {"origin": "http://localhost", "x-csrf-token": csrf}

        created = await client.post(
            "/api/v1/knowledge-bases",
            json={"name": "旅行资料", "description": "可搜索的本地资料"},
            headers=headers,
        )
        assert created.status_code == 201
        knowledge_base_id = created.json()["id"]
        assert created.json()["name"] == "旅行资料"

        listed = await client.get("/api/v1/knowledge-bases")
        assert [item["id"] for item in listed.json()["items"]] == [knowledge_base_id]
        updated = await client.patch(
            f"/api/v1/knowledge-bases/{knowledge_base_id}",
            json={"description": ""},
            headers=headers,
        )
        assert updated.json()["description"] == ""

        rejected = await client.post(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/sources",
            files={"file": ("malware.exe", b"unsafe", "application/octet-stream")},
            headers=headers,
        )
        assert rejected.status_code == 415
        assert rejected.json()["error"]["code"] == "unsupported_file_type"

        uploaded = await client.post(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/sources",
            files={
                "file": (
                    r"C:\fakepath\guide.md",
                    "巴黎是法国的首都。游客可以乘坐地铁。".encode(),
                    "text/markdown",
                )
            },
            headers=headers,
        )
        assert uploaded.status_code == 202
        assert uploaded.json()["source"]["originalName"] == "guide.md"
        source_id = uploaded.json()["source"]["id"]
        job_id = uploaded.json()["job"]["id"]
        finished = await _wait_for_job(client, job_id)
        assert finished["status"] == "succeeded"
        assert finished["kind"] == "ingest"

        sources = await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/sources")
        assert sources.json()["items"][0]["status"] == "ready"
        assert sources.json()["items"][0]["documentCount"] == 1

        duplicate = await client.post(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/sources",
            files={
                "file": (
                    "renamed.md",
                    "巴黎是法国的首都。游客可以乘坐地铁。".encode(),
                    "text/markdown",
                )
            },
            headers=headers,
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "duplicate_source"

        found = await client.post(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/search",
            json={"query": "法国首都", "topK": 3, "mode": "hybrid_character"},
            headers=headers,
        )
        assert found.status_code == 200
        assert found.json()["items"][0]["sourceId"] == source_id
        assert found.json()["items"][0]["scoringMethod"] == "hybrid_character"

        await asyncio.sleep(0.01)
        reindexed_source = await client.post(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/sources/{source_id}/reindex",
            headers=headers,
        )
        assert reindexed_source.status_code == 200
        source_reindex_job = reindexed_source.json()["job"]
        assert source_reindex_job["kind"] == "reindex"
        assert (await _wait_for_job(client, source_reindex_job["id"]))["status"] == "succeeded"

        await asyncio.sleep(0.01)
        reindexed_all = await client.post(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/reindex",
            headers=headers,
        )
        assert reindexed_all.status_code == 202
        full_reindex_job = reindexed_all.json()["jobs"][0]
        assert (await _wait_for_job(client, full_reindex_job["id"]))["status"] == "succeeded"
        jobs = await client.get(
            "/api/v1/jobs",
            params={"knowledgeBaseId": knowledge_base_id, "limit": 10},
        )
        assert {item["kind"] for item in jobs.json()["items"]} == {"ingest", "reindex"}

        saved = await client.put(
            "/api/v1/model-settings",
            json={
                "baseUrl": "http://127.0.0.1:11434/v1",
                "model": "local-model",
                "apiKey": "top-secret-key",
            },
            headers=headers,
        )
        assert saved.status_code == 200
        assert saved.json()["apiKeyConfigured"] is True
        assert "top-secret-key" not in saved.text

        connection = await client.post("/api/v1/model-settings/test", headers=headers)
        assert connection.json() == {
            "ok": True,
            "model": "mock-model",
            "message": "连接成功。",
        }
        answered = await client.post(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/answer",
            json={"question": "法国的首都在哪里?", "topK": 3},
            headers=headers,
        )
        assert answered.status_code == 200
        assert answered.json()["content"].endswith("[1]")
        assert answered.json()["citations"][0]["sourceName"] == "guide.md"
        assert answered.json()["searchHits"][0]["sourceId"] == source_id
        assert seen_keys == ["top-secret-key", "top-secret-key"]

        preserved = await client.put(
            "/api/v1/model-settings",
            json={"baseUrl": "http://localhost:11434/v1", "model": "next-model"},
            headers=headers,
        )
        assert preserved.json()["apiKeyConfigured"] is True
        cleared = await client.put(
            "/api/v1/model-settings",
            json={
                "baseUrl": "http://localhost:11434/v1",
                "model": "next-model",
                "clearApiKey": True,
            },
            headers=headers,
        )
        assert cleared.json()["apiKeyConfigured"] is False
        assert services.repository.get_model_api_key() is None

        deleted_source = await client.delete(
            f"/api/v1/knowledge-bases/{knowledge_base_id}/sources/{source_id}",
            headers=headers,
        )
        assert deleted_source.status_code == 204
        deleted_kb = await client.delete(
            f"/api/v1/knowledge-bases/{knowledge_base_id}",
            headers=headers,
        )
        assert deleted_kb.status_code == 204
        assert (await client.get("/api/v1/knowledge-bases")).json() == {"items": []}


@pytest.mark.asyncio
async def test_studio_app_restart_recovers_interrupted_work_through_api(tmp_path: Path) -> None:
    paths = StudioPaths(tmp_path / "studio").ensure()
    repository = StudioRepository(paths.database)
    knowledge_base = repository.create_knowledge_base("异常恢复")
    pending_source = repository.create_source(
        knowledge_base.id,
        original_name="queued.md",
        stored_name="queued.md",
        media_type="text/markdown",
        size_bytes=10,
    )
    processing_source = repository.create_source(
        knowledge_base.id,
        original_name="running.md",
        stored_name="running.md",
        media_type="text/markdown",
        size_bytes=10,
    )
    repository.update_source_status(processing_source.id, "processing")
    queued_job = repository.create_job(
        knowledge_base.id,
        kind="ingest",
        source_id=pending_source.id,
    )
    running_job = repository.create_job(
        knowledge_base.id,
        kind="reindex",
        source_id=processing_source.id,
    )
    repository.update_job(running_job.id, "running")
    repository.close()

    app = create_studio_app(paths.root)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
        ) as client,
    ):
        sources_response = await client.get(f"/api/v1/knowledge-bases/{knowledge_base.id}/sources")
        jobs_response = await client.get(
            "/api/v1/jobs",
            params={"knowledgeBaseId": knowledge_base.id},
        )

    assert sources_response.status_code == 200
    assert jobs_response.status_code == 200
    expected_error = "上次运行中断, 请重新处理。"
    sources = {item["id"]: item for item in sources_response.json()["items"]}
    jobs = {item["id"]: item for item in jobs_response.json()["items"]}
    assert sources[pending_source.id]["status"] == "failed"
    assert sources[pending_source.id]["error"] == expected_error
    assert sources[processing_source.id]["status"] == "failed"
    assert sources[processing_source.id]["error"] == expected_error
    assert jobs[queued_job.id]["status"] == "failed"
    assert jobs[queued_job.id]["error"] == expected_error
    assert jobs[running_job.id]["status"] == "failed"
    assert jobs[running_job.id]["error"] == expected_error

    reopened = StudioRepository(paths.database)
    try:
        assert reopened.require_source(pending_source.id).status == "failed"
        persisted_job = reopened.get_job(running_job.id)
        assert persisted_job is not None
        assert persisted_job.status == "failed"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_studio_api_security_limits_and_redacted_validation(tmp_path: Path) -> None:
    services, _ = _services(tmp_path, max_file_bytes=32, max_total_bytes=32)
    app = create_studio_app(services=services)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            missing_csrf = await client.post(
                "/api/v1/knowledge-bases",
                json={"name": "Blocked"},
            )
            assert missing_csrf.status_code == 403
            assert missing_csrf.json()["error"]["code"] == "csrf_failed"
            assert missing_csrf.headers["x-request-id"]

            csrf = await _bootstrap(client)
            headers = {"origin": "http://localhost", "x-csrf-token": csrf}
            hostile = await client.post(
                "/api/v1/knowledge-bases",
                json={"name": "Blocked"},
                headers={**headers, "origin": "https://attacker.example"},
            )
            assert hostile.status_code == 403
            assert hostile.json()["error"]["code"] == "origin_not_allowed"

            wrong_local_port = await client.post(
                "/api/v1/knowledge-bases",
                json={"name": "Blocked"},
                headers={**headers, "origin": "http://localhost:9999"},
            )
            assert wrong_local_port.status_code == 403
            assert wrong_local_port.json()["error"]["code"] == "origin_not_allowed"

            secret = "must-never-appear"
            invalid = await client.put(
                "/api/v1/model-settings",
                json={
                    "baseUrl": "http://localhost:11434",
                    "model": "local",
                    "apiKey": secret,
                    "unexpected": True,
                },
                headers=headers,
            )
            assert invalid.status_code == 422
            assert invalid.json()["error"]["code"] == "validation_error"
            assert secret not in invalid.text

            created = await client.post(
                "/api/v1/knowledge-bases",
                json={"name": "Limits"},
                headers=headers,
            )
            knowledge_base_id = created.json()["id"]
            too_large = await client.post(
                f"/api/v1/knowledge-bases/{knowledge_base_id}/sources",
                files={"file": ("large.txt", b"x" * 33, "text/plain")},
                headers=headers,
            )
            assert too_large.status_code == 413
            assert too_large.json()["error"]["code"] == "file_too_large"

            within_limit = await client.post(
                f"/api/v1/knowledge-bases/{knowledge_base_id}/sources",
                files={"file": ("first.txt", b"a" * 20, "text/plain")},
                headers=headers,
            )
            assert within_limit.status_code == 202
            storage_limit = await client.post(
                f"/api/v1/knowledge-bases/{knowledge_base_id}/sources",
                files={"file": ("second.txt", b"b" * 20, "text/plain")},
                headers=headers,
            )
            assert storage_limit.status_code == 507
            assert storage_limit.json()["error"]["code"] == "storage_limit"

            body_limit = await client.post(
                f"/api/v1/knowledge-bases/{knowledge_base_id}/sources",
                content=b"x" * (1024 * 1024 + 33),
                headers={**headers, "content-type": "multipart/form-data; boundary=x"},
            )
            assert body_limit.status_code == 413
            assert body_limit.json()["error"]["code"] == "request_too_large"

            missing = await client.get("/api/v1/does-not-exist")
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "not_found"

        async with httpx.AsyncClient(transport=transport, base_url="http://evil.example") as evil:
            rejected_host = await evil.get("/api/v1/bootstrap")
            assert rejected_host.status_code == 400
            assert rejected_host.json()["error"]["code"] == "host_not_allowed"


@pytest.mark.asyncio
async def test_network_mode_requires_request_origin_to_match_host(tmp_path: Path) -> None:
    services, _ = _services(tmp_path)
    app = create_studio_app(services=services, allow_network=True)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://192.0.2.10:8765",
        ) as client,
    ):
        csrf = await _bootstrap(client)
        accepted = await client.post(
            "/api/v1/knowledge-bases",
            json={"name": "LAN docs"},
            headers={
                "origin": "http://192.0.2.10:8765",
                "x-csrf-token": csrf,
            },
        )
        assert accepted.status_code == 201

        rejected = await client.post(
            "/api/v1/knowledge-bases",
            json={"name": "Wrong origin"},
            headers={
                "origin": "http://192.0.2.11:8765",
                "x-csrf-token": csrf,
            },
        )
        assert rejected.status_code == 403
        assert rejected.json()["error"]["code"] == "origin_not_allowed"


@pytest.mark.asyncio
async def test_local_mode_accepts_ipv6_loopback(tmp_path: Path) -> None:
    services, _ = _services(tmp_path)
    app = create_studio_app(services=services)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://[::1]:8765",
        ) as client,
    ):
        bootstrap = await client.get("/api/v1/bootstrap")

    assert bootstrap.status_code == 200

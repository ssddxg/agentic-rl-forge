from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from agentic_rl_forge.studio import create_studio_app
from agentic_rl_forge.studio import rl as rl_module
from agentic_rl_forge.studio.rl import (
    OfflineRunCreate,
    RLJobRunner,
    RLRunStatus,
    RLWorkspace,
)
from agentic_rl_forge.studio.security import UploadBodyLimitMiddleware

_QA_PAYLOAD = b'{"id":"fr","question":"Capital of France?","answer":"Paris"}\n'
_CORPUS_PAYLOAD = b'{"id":"fr","contents":"Paris is the capital of France."}\n'


async def _csrf_headers(client: httpx.AsyncClient) -> dict[str, str]:
    response = await client.get("/api/v1/bootstrap")
    assert response.status_code == 200
    return {
        "origin": "http://localhost",
        "x-csrf-token": str(response.json()["security"]["csrfToken"]),
    }


@pytest.mark.asyncio
async def test_rl_studio_runs_real_trajectory_and_offline_pipeline(tmp_path: Path) -> None:
    app = create_studio_app(tmp_path / "studio")
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        headers = await _csrf_headers(client)

        overview = await client.get("/api/v1/rl/overview")
        assert overview.status_code == 200
        assert overview.json()["app"]["name"] == "AgenticRLForge"
        assert overview.json()["diagnostics"]["core"]["status"] == "pass"
        assert {item["id"] for item in overview.json()["algorithms"]} >= {
            "grpo",
            "nash-md",
            "mcts-prm",
        }

        demo = await client.post("/api/v1/rl/trajectory-demo", headers=headers)
        assert demo.status_code == 200
        assert demo.json()["status"] == "succeeded"
        assert demo.json()["summary"]["stepCount"] == 2
        assert demo.json()["summary"]["totalReward"] > 0
        assert demo.json()["steps"][0]["action"] == "tool"
        assert demo.json()["steps"][0]["toolResults"][0]["ok"] is True
        assert "Paris" in demo.json()["steps"][0]["toolResults"][0]["content"]
        assert demo.json()["steps"][1]["finalAnswer"] == "Paris"
        assert {signal["name"] for signal in demo.json()["finalRewardSignals"]} >= {"exact_match"}

        datasets = await client.get("/api/v1/rl/datasets")
        assert datasets.status_code == 200
        assert datasets.json()["items"][0]["id"] == "sample"

        uploaded_dataset = await client.post(
            "/api/v1/rl/datasets",
            data={"name": "首都问答验证集"},
            files={
                "qaFile": (
                    "qa.jsonl",
                    _QA_PAYLOAD,
                    "application/x-ndjson",
                ),
                "corpusFile": (
                    "corpus.jsonl",
                    _CORPUS_PAYLOAD,
                    "application/x-ndjson",
                ),
            },
            headers=headers,
        )
        assert uploaded_dataset.status_code == 201
        dataset_id = uploaded_dataset.json()["id"]
        assert uploaded_dataset.json()["taskCount"] == 1
        assert uploaded_dataset.json()["documentCount"] == 1

        duplicate_dataset = await client.post(
            "/api/v1/rl/datasets",
            data={"name": "同内容重复上传"},
            files={
                "qaFile": ("qa.jsonl", _QA_PAYLOAD, "application/x-ndjson"),
                "corpusFile": ("corpus.jsonl", _CORPUS_PAYLOAD, "application/x-ndjson"),
            },
            headers=headers,
        )
        assert duplicate_dataset.status_code == 409
        assert duplicate_dataset.json()["error"]["code"] == "duplicate_rl_dataset"

        created = await client.post(
            "/api/v1/rl/offline-runs",
            json={
                "rolloutsPerTask": 2,
                "seed": 7,
                "maxConcurrency": 2,
                "datasetId": dataset_id,
            },
            headers=headers,
        )
        assert created.status_code == 202
        run_id = created.json()["id"]
        assert created.json()["status"] == "queued"
        assert created.json()["config"]["datasetId"] == dataset_id

        finished: dict[str, object] | None = None
        for _ in range(400):
            current = await client.get(f"/api/v1/rl/offline-runs/{run_id}")
            assert current.status_code == 200
            payload = current.json()
            if payload["status"] in {"succeeded", "failed"}:
                finished = payload
                break
            await asyncio.sleep(0.01)

        assert finished is not None
        assert finished["status"] == "succeeded"
        result = finished["result"]
        assert isinstance(result, dict)
        assert result["trajectory_count"] == 2
        assert result["accepted_trajectory_count"] >= 1
        assert result["trainer_batch_valid"] is True
        artifacts = result["artifacts"]
        assert isinstance(artifacts, dict)
        assert set(artifacts) == {
            "benchmark_report",
            "database",
            "filtered_trajectories",
            "shards",
            "trainer_store",
        }
        artifact_exists = await asyncio.gather(
            *(asyncio.to_thread(Path(str(path)).exists) for path in artifacts.values())
        )
        assert all(artifact_exists)

        listed = await client.get("/api/v1/rl/offline-runs")
        assert listed.status_code == 200
        assert listed.json()["items"][0]["id"] == run_id


def test_rl_workspace_recovers_interrupted_runs(tmp_path: Path) -> None:
    workspace = RLWorkspace(tmp_path / "rl").ensure()
    record = workspace.create_offline_run(OfflineRunCreate())

    assert workspace.recover_interrupted() == 1
    recovered = workspace.get_run(record.id)

    assert recovered is not None
    assert recovered.status is RLRunStatus.FAILED
    assert recovered.error == "上次运行被中断, 请重新启动这项验证。"
    assert workspace.recover_interrupted() == 0


@pytest.mark.asyncio
async def test_rl_api_enforces_csrf_and_returns_safe_upload_errors(tmp_path: Path) -> None:
    data_dir = tmp_path / "private-studio-data"
    app = create_studio_app(data_dir)
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        missing_csrf = await client.post("/api/v1/rl/trajectory-demo")
        assert missing_csrf.status_code == 403
        assert missing_csrf.json()["error"]["code"] == "csrf_failed"

        headers = await _csrf_headers(client)
        wrong_origin = await client.post(
            "/api/v1/rl/trajectory-demo",
            headers={**headers, "origin": "https://attacker.example"},
        )
        assert wrong_origin.status_code == 403
        assert wrong_origin.json()["error"]["code"] == "origin_not_allowed"

        wrong_extension = await client.post(
            "/api/v1/rl/datasets",
            data={"name": "错误扩展名"},
            files={
                "qaFile": ("qa.txt", _QA_PAYLOAD, "text/plain"),
                "corpusFile": ("corpus.jsonl", _CORPUS_PAYLOAD, "application/x-ndjson"),
            },
            headers=headers,
        )
        assert wrong_extension.status_code == 415
        assert wrong_extension.json()["error"]["code"] == "invalid_dataset_file"

        malformed = await client.post(
            "/api/v1/rl/datasets",
            data={"name": "损坏数据"},
            files={
                "qaFile": ("../../qa.jsonl", b"{not-json}\n", "application/x-ndjson"),
                "corpusFile": (
                    "..\\..\\corpus.jsonl",
                    _CORPUS_PAYLOAD,
                    "application/x-ndjson",
                ),
            },
            headers=headers,
        )
        assert malformed.status_code == 422
        error = malformed.json()["error"]
        assert error["code"] == "invalid_rl_dataset"
        assert str(data_dir) not in error["message"]
        assert "qa.jsonl" in error["message"]

        too_large = await client.post(
            "/api/v1/rl/datasets",
            data={"name": "超大数据"},
            files={
                "qaFile": (
                    "qa.jsonl",
                    b"x" * (8 * 1024 * 1024 + 1),
                    "application/x-ndjson",
                ),
                "corpusFile": ("corpus.jsonl", _CORPUS_PAYLOAD, "application/x-ndjson"),
            },
            headers=headers,
        )
        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "dataset_file_too_large"

        unknown_dataset = await client.post(
            "/api/v1/rl/offline-runs",
            json={"datasetId": "data_0000000000000000"},
            headers=headers,
        )
        assert unknown_dataset.status_code == 404
        assert unknown_dataset.json()["error"]["code"] == "rl_dataset_not_found"

        unsafe_run_id = await client.get("/api/v1/rl/offline-runs/not-a-run")
        assert unsafe_run_id.status_code == 404
        assert unsafe_run_id.json()["error"]["code"] == "rl_run_not_found"


def test_rl_workspace_deduplicates_simultaneous_dataset_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = RLWorkspace(tmp_path / "rl").ensure()
    worker_count = 4
    ready = threading.Barrier(worker_count)
    original_loader = rl_module.load_offline_tasks

    def synchronized_loader(path: Path):  # type: ignore[no-untyped-def]
        ready.wait(timeout=10)
        return original_loader(path)

    monkeypatch.setattr(rl_module, "load_offline_tasks", synchronized_loader)

    def upload(index: int):  # type: ignore[no-untyped-def]
        try:
            return workspace.create_dataset(
                f"并发数据集 {index}",
                _QA_PAYLOAD,
                _CORPUS_PAYLOAD,
            )
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = tuple(executor.map(upload, range(worker_count)))

    assert sum(record is not None for record in results) == 1
    assert len(workspace.list_datasets(include_sample=False)) == 1


@pytest.mark.asyncio
async def test_rl_run_fails_cleanly_when_dataset_integrity_is_lost(tmp_path: Path) -> None:
    workspace = RLWorkspace(tmp_path / "rl").ensure()
    dataset = workspace.create_dataset("完整性验证", _QA_PAYLOAD, _CORPUS_PAYLOAD)
    run = workspace.create_offline_run(OfflineRunCreate(datasetId=dataset.id))
    (workspace.datasets / dataset.id / "qa.jsonl").write_bytes(_QA_PAYLOAD + b" ")

    finished = await workspace.execute_offline_run(run.id)

    assert finished.status is RLRunStatus.FAILED
    assert finished.error is not None
    assert "integrity verification" in finished.error
    assert str(workspace.root) not in finished.error
    persisted = workspace.get_run(run.id)
    assert persisted is not None
    assert persisted.status is RLRunStatus.FAILED


@pytest.mark.asyncio
async def test_rl_job_runner_serializes_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = RLWorkspace(tmp_path / "rl").ensure()
    runner = RLJobRunner(workspace)
    active = 0
    peak_active = 0
    completed = 0
    all_completed = asyncio.Event()

    async def fake_execute(run_id: str) -> None:
        nonlocal active, peak_active, completed
        assert run_id in {"first", "second"}
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        completed += 1
        if completed == 2:
            all_completed.set()

    monkeypatch.setattr(workspace, "execute_offline_run", fake_execute)
    runner.enqueue("first")
    runner.enqueue("second")

    await asyncio.wait_for(all_completed.wait(), timeout=2)
    await runner.close()

    assert peak_active == 1


@pytest.mark.asyncio
async def test_rl_job_runner_shutdown_marks_running_and_queued_runs_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = RLWorkspace(tmp_path / "rl").ensure()
    first = workspace.create_offline_run(OfflineRunCreate())
    second = workspace.create_offline_run(OfflineRunCreate())
    pipeline_started = asyncio.Event()

    async def blocked_pipeline(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pipeline_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(rl_module, "run_offline_pipeline", blocked_pipeline)
    runner = RLJobRunner(workspace)
    runner.enqueue(first.id)
    runner.enqueue(second.id)
    await asyncio.wait_for(pipeline_started.wait(), timeout=2)

    await runner.close()

    first_record = workspace.get_run(first.id)
    second_record = workspace.get_run(second.id)
    assert first_record is not None
    assert second_record is not None
    assert first_record.status is RLRunStatus.FAILED
    assert second_record.status is RLRunStatus.FAILED
    assert first_record.error == "应用已停止, 本次运行未完成。"
    assert second_record.error == "应用已停止, 本次运行未完成。"


@pytest.mark.asyncio
async def test_rl_app_restart_recovers_running_run_through_api(tmp_path: Path) -> None:
    data_dir = tmp_path / "studio"
    workspace = RLWorkspace(data_dir / "rl").ensure()
    queued = workspace.create_offline_run(OfflineRunCreate())
    running = workspace._claim_queued_run(queued.id)
    assert running.status is RLRunStatus.RUNNING
    app = create_studio_app(data_dir)
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        response = await client.get(f"/api/v1/rl/offline-runs/{queued.id}")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "上次运行被中断, 请重新启动这项验证。"


@pytest.mark.skipif(os.name != "nt", reason="Windows portable path guard")
@pytest.mark.asyncio
async def test_rl_run_reports_windows_long_path_as_a_terminal_failure(tmp_path: Path) -> None:
    # Keep the input paths below the legacy limit while ensuring the deepest exported artifact
    # would exceed it. The pipeline must fail early and persist a terminal state.
    padding = max(8, min(80, 170 - len(str(tmp_path))))
    workspace = RLWorkspace(tmp_path / ("x" * padding)).ensure()
    run = workspace.create_offline_run(OfflineRunCreate())

    finished = await workspace.execute_offline_run(run.id)

    assert finished.status is RLRunStatus.FAILED
    assert finished.error is not None
    assert "path is too long" in finished.error
    assert str(workspace.root) not in finished.error


@pytest.mark.asyncio
async def test_upload_body_limit_uses_exact_rl_dataset_paths_and_chunk_limits() -> None:
    downstream = FastAPI()

    @downstream.post("/{path:path}")
    async def accepted(path: str) -> dict[str, object]:
        return {"accepted": True, "path": path}

    app = UploadBodyLimitMiddleware(
        downstream,
        maximum_bytes=10,
        rl_dataset_maximum_bytes=20,
    )
    transport = httpx.ASGITransport(app=app)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"x" * 12
        yield b"y" * 9

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        content_length_rejected = await client.post(
            "/api/v1/rl/datasets",
            content=b"x" * 21,
        )
        chunked_rejected = await client.post(
            "/api/rl/datasets",
            content=chunks(),
        )
        near_match_allowed = await client.post(
            "/api/v1/rl/datasets-extra",
            content=b"x" * 21,
        )
        source_rejected = await client.post(
            "/api/v1/knowledge-bases/example/sources",
            content=b"x" * 11,
        )

    assert content_length_rejected.status_code == 413
    assert content_length_rejected.json()["error"]["code"] == "request_too_large"
    assert chunked_rejected.status_code == 413
    assert chunked_rejected.json()["error"]["code"] == "request_too_large"
    assert near_match_allowed.status_code == 200
    assert near_match_allowed.json()["accepted"] is True
    assert source_rejected.status_code == 413

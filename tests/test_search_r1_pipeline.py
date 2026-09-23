from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

import httpx
import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    RunArtifactManifest,
    RunHeartbeat,
    RunLeaseToken,
    RunStatus,
    utc_now,
)
from agentic_rl_forge.pipelines import (
    SearchR1CollectionConfig,
    collect_search_r1,
    inspect_search_r1_plan,
    load_search_r1_collection_config,
)
from agentic_rl_forge.pipelines.search_r1 import _renew_run_lease
from agentic_rl_forge.storage import RunArtifactBundle, SQLiteTrajectoryStore, TrajectoryQuery


class RecordingRunLeaseStore:
    def __init__(self) -> None:
        self.renewed = Event()
        self.renewal_count = 0

    def renew_run_lease(
        self,
        run_id: str,
        token: RunLeaseToken,
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> RunHeartbeat:
        self.renewal_count += 1
        renewed_at = now or utc_now()
        self.renewed.set()
        return RunHeartbeat(
            run_id=run_id,
            owner_id=token.owner_id,
            epoch=token.epoch,
            acquired_at=renewed_at,
            heartbeat_at=renewed_at,
            lease_expires_at=renewed_at + timedelta(seconds=ttl_s),
        )


def collection_config(**updates: object) -> SearchR1CollectionConfig:
    values: dict[str, object] = {
        "name": "mock-search-r1",
        "dataset_name": "mock-qa",
        "model": "mock-model",
        "policy_version": "mock-policy-v1",
        "model_base_url": "http://model.local",
        "retrieval_endpoint": "http://retriever.local/retrieve",
        "rollouts_per_task": 2,
        "max_tasks": 1,
        "max_concurrency": 2,
        "model_max_concurrency": 2,
        "model_max_retries": 0,
        "run_lease_ttl_s": 0.5,
        "heartbeat_interval_s": 0.02,
        "slot_claim_ttl_s": 0.5,
        "slot_claim_renewal_interval_s": 0.01,
    }
    values.update(updates)
    return SearchR1CollectionConfig.model_validate(values)


@pytest.mark.asyncio
async def test_search_r1_collection_persists_complete_mock_run(tmp_path: Path) -> None:
    source = tmp_path / "qa.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "france",
                "question": "What is the capital of France?",
                "answer": "Paris",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model_payloads: list[dict[str, object]] = []

    async def model_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        payload = json.loads(request.content)
        model_payloads.append(payload)
        messages = payload["messages"]
        has_information = any(
            message["role"] == "user" and str(message["content"]).startswith("<information>")
            for message in messages
        )
        content = (
            "<think>The passage identifies the capital.</think><answer>Paris</answer>"
            if has_information
            else "<think>I need evidence.</think><search>capital of France</search>"
        )
        return httpx.Response(
            200,
            json={
                "model": "mock-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                        "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                    }
                ],
                "usage": {"completion_tokens": 2},
            },
        )

    async def retrieval_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/retrieve"
        assert json.loads(request.content)["queries"] == ["capital of France"]
        return httpx.Response(
            200,
            json={
                "result": [
                    [
                        {
                            "document": {
                                "id": "france",
                                "contents": "Paris is the capital of France.",
                            },
                            "score": 1.0,
                        }
                    ]
                ]
            },
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(model_handler)) as model_client,
        httpx.AsyncClient(transport=httpx.MockTransport(retrieval_handler)) as retrieval_client,
    ):
        result = await collect_search_r1(
            source,
            tmp_path / "artifacts",
            collection_config(),
            model_client=model_client,
            retrieval_client=retrieval_client,
        )
        resumed = await collect_search_r1(
            source,
            tmp_path / "artifacts",
            collection_config(),
            model_client=model_client,
            retrieval_client=retrieval_client,
        )
        cached_status = await inspect_search_r1_plan(
            source,
            tmp_path / "artifacts",
            collection_config(),
        )
        fresh_status = await inspect_search_r1_plan(
            source,
            tmp_path / "artifacts",
            collection_config(plan_salt="fresh-sample"),
        )

    assert result.task_count == 1
    assert result.trajectory_count == 2
    assert result.collected_trajectory_count == 2
    assert result.reused_trajectory_count == 0
    assert result.success_count == 2
    assert result.status_counts == {"succeeded": 2}
    assert len(model_payloads) == 4
    assert resumed.run_id != result.run_id
    assert resumed.plan_id == result.plan_id
    assert resumed.collected_trajectory_count == 0
    assert resumed.reused_trajectory_count == 2
    assert resumed.trajectory_count == 2
    assert cached_status.plan_id == result.plan_id
    assert cached_status.reusable_slot_count == 2
    assert cached_status.missing_slot_count == 0
    assert cached_status.conflict_count == 0
    assert fresh_status.plan_id != result.plan_id
    assert fresh_status.reusable_slot_count == 0
    assert fresh_status.missing_slot_count == 2
    assert fresh_status.conflict_count == 0
    assert all("tools" not in payload for payload in model_payloads)
    assert (
        sum(
            any(
                message["role"] == "user" and str(message["content"]).startswith("<information>")
                for message in payload["messages"]
            )
            for payload in model_payloads
        )
        == 2
    )
    artifact_paths = tuple(Path(artifact) for artifact in result.artifacts.values())
    assert all(await asyncio.gather(*(asyncio.to_thread(path.exists) for path in artifact_paths)))
    metrics_text = await asyncio.to_thread(
        Path(result.artifacts["metrics"]).read_text,
        encoding="utf-8",
    )
    assert "arf_trajectories_total" in metrics_text
    artifact_manifest_path = Path(result.artifacts["artifact_manifest"])
    artifact_manifest = RunArtifactManifest.model_validate_json(
        await asyncio.to_thread(artifact_manifest_path.read_bytes)
    )
    artifact_verification = await asyncio.to_thread(
        RunArtifactBundle(tmp_path / "artifacts").verify,
        artifact_manifest,
    )
    assert artifact_verification.valid
    assert artifact_verification.verified_artifact_count == 7
    assert artifact_verification.shard_verification is not None
    assert artifact_verification.shard_verification.complete

    portable_root = tmp_path / "portable-copy"
    await asyncio.to_thread(shutil.copytree, tmp_path / "artifacts", portable_root)
    portable_manifest_path = portable_root / artifact_manifest_path.relative_to(
        tmp_path / "artifacts"
    )
    portable_manifest = RunArtifactManifest.model_validate_json(
        await asyncio.to_thread(portable_manifest_path.read_bytes)
    )
    portable_verification = await asyncio.to_thread(
        RunArtifactBundle(portable_root).verify,
        portable_manifest,
    )
    assert portable_verification.valid
    cli_verification = await asyncio.to_thread(
        CliRunner().invoke,
        app,
        ["run-artifacts-verify", str(portable_manifest_path)],
    )
    assert cli_verification.exit_code == 0
    assert '"valid": true' in cli_verification.stdout

    portable_metrics = portable_root / next(
        item.relative_path for item in portable_manifest.artifacts if item.name == "metrics"
    )
    await asyncio.to_thread(portable_metrics.write_text, "tampered\n", encoding="utf-8")
    tampered_verification = await asyncio.to_thread(
        RunArtifactBundle(portable_root).verify,
        portable_manifest,
    )
    assert not tampered_verification.valid
    assert tampered_verification.mismatched_artifacts == ("metrics",)
    tampered_cli = await asyncio.to_thread(
        CliRunner().invoke,
        app,
        ["run-artifacts-verify", str(portable_manifest_path)],
    )
    assert tampered_cli.exit_code == 1
    assert '"metrics"' in tampered_cli.stdout
    claim_releases = await asyncio.to_thread(
        lambda: tuple(Path(result.artifacts["slot_claims"]).rglob("releases/*.json"))
    )
    assert len(claim_releases) == 2
    claim_renewals = await asyncio.to_thread(
        lambda: tuple(Path(result.artifacts["slot_claims"]).rglob("renewals/**/*.json"))
    )
    assert len(claim_renewals) >= 2

    with SQLiteTrajectoryStore(Path(result.artifacts["database"])) as store:
        run = store.get_run(result.run_id)
        heartbeat = store.get_run_heartbeat(result.run_id)
        trajectories = store.query(TrajectoryQuery(run_id=result.run_id))
    assert run is not None
    assert run.status is RunStatus.COMPLETED
    assert heartbeat is not None
    assert heartbeat.heartbeat_at > heartbeat.acquired_at
    assert heartbeat.released_at is not None
    assert run.config["api_key_configured"] is False
    assert len(trajectories) == 2
    assert all(item.total_observation_tokens > 0 for item in trajectories)

    with SQLiteTrajectoryStore(Path(resumed.artifacts["database"])) as store:
        resumed_trajectories = store.query(TrajectoryQuery(run_id=resumed.run_id))
    assert {item.trajectory_id for item in resumed_trajectories} == {
        item.trajectory_id for item in trajectories
    }


@pytest.mark.asyncio
async def test_run_lease_renewal_uses_persisted_heartbeat_deadline() -> None:
    store = RecordingRunLeaseStore()
    stop = asyncio.Event()
    observed_at = utc_now() - timedelta(seconds=1)
    heartbeat = RunHeartbeat(
        run_id="run-delayed-heartbeat",
        owner_id="worker-delayed-heartbeat",
        epoch=1,
        acquired_at=observed_at,
        heartbeat_at=observed_at,
        lease_expires_at=observed_at + timedelta(seconds=5),
    )
    renewal_task = asyncio.create_task(
        _renew_run_lease(
            store,  # type: ignore[arg-type]
            heartbeat.run_id,
            heartbeat.token,
            initial_heartbeat=heartbeat,
            ttl_s=5.0,
            interval_s=0.5,
            stop=stop,
        )
    )

    renewed_promptly = await asyncio.to_thread(store.renewed.wait, 0.2)
    stop.set()
    await renewal_task

    assert renewed_promptly
    assert store.renewal_count == 1


@pytest.mark.asyncio
async def test_run_lease_renewal_handles_legacy_asyncio_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyAsyncioTimeoutError(Exception):
        pass

    original_wait_for = asyncio.wait_for
    wait_count = 0

    async def raise_first_legacy_timeout(
        awaitable: object,
        timeout: float | None = None,
    ) -> object:
        nonlocal wait_count
        wait_count += 1
        if wait_count == 1:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise LegacyAsyncioTimeoutError
        return await original_wait_for(awaitable, timeout=timeout)  # type: ignore[arg-type]

    monkeypatch.setattr(asyncio, "TimeoutError", LegacyAsyncioTimeoutError)
    monkeypatch.setattr(asyncio, "wait_for", raise_first_legacy_timeout)
    store = RecordingRunLeaseStore()
    stop = asyncio.Event()
    observed_at = utc_now()
    heartbeat = RunHeartbeat(
        run_id="run-legacy-timeout",
        owner_id="worker-legacy-timeout",
        epoch=1,
        acquired_at=observed_at,
        heartbeat_at=observed_at,
        lease_expires_at=observed_at + timedelta(seconds=10),
    )
    renewal_task = asyncio.create_task(
        _renew_run_lease(
            store,  # type: ignore[arg-type]
            heartbeat.run_id,
            heartbeat.token,
            initial_heartbeat=heartbeat,
            ttl_s=10.0,
            interval_s=5.0,
            stop=stop,
        )
    )

    assert await asyncio.to_thread(store.renewed.wait, 0.2)
    stop.set()
    await renewal_task

    assert store.renewal_count == 1


def test_collection_config_loads_yaml_and_rejects_embedded_credentials(tmp_path: Path) -> None:
    config_path = tmp_path / "collection.yaml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "name: test",
                "dataset_name: qa",
                "model: model",
                "policy_version: policy-v1",
                "model_base_url: http://127.0.0.1:8001",
                "retrieval_endpoint: http://127.0.0.1:8000/retrieve",
                "rollouts_per_task: 2",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_search_r1_collection_config(config_path)

    assert loaded.name == "test"
    assert (
        collection_config().plan_config_digest
        == collection_config(
            max_concurrency=7,
            model_timeout_s=10,
            run_lease_ttl_s=20,
            heartbeat_interval_s=5,
            slot_claim_ttl_s=20,
            slot_claim_renewal_interval_s=5,
        ).plan_config_digest
    )
    assert (
        collection_config().plan_config_digest
        != collection_config(temperature=0.5).plan_config_digest
    )
    with pytest.raises(ValueError, match="cannot contain credentials"):
        collection_config(model_base_url="http://token:model@model.local")
    with pytest.raises(ValueError, match="must not exceed half"):
        collection_config(run_lease_ttl_s=10, heartbeat_interval_s=6)
    with pytest.raises(ValueError, match="slot_claim_renewal_interval_s"):
        collection_config(slot_claim_ttl_s=10, slot_claim_renewal_interval_s=6)

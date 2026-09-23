from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agentic_rl_forge.pipelines.offline as offline_module
from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import Message, MessageRole, RunStatus
from agentic_rl_forge.data import corpus_digest, load_corpus_jsonl
from agentic_rl_forge.pipelines.offline import SeededAnswerPolicy, run_offline_pipeline
from agentic_rl_forge.rollout import GenerationRequest
from agentic_rl_forge.storage import SQLiteTrajectoryStore


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "qa.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    data.write_text(
        '{"id":"capital","question":"What is the capital of France?",'
        '"answer":["Paris","City of Paris"]}\n',
        encoding="utf-8",
    )
    corpus.write_text(
        '{"id":"evidence","contents":"PARIS is the capital of France."}\n',
        encoding="utf-8",
    )
    return data, corpus


@pytest.mark.asyncio
async def test_offline_pipeline_handles_negative_seed_and_records_input_digests(
    tmp_path: Path,
) -> None:
    data, corpus = _write_inputs(tmp_path)
    output = tmp_path / "run"

    result = await run_offline_pipeline(
        output,
        data,
        corpus,
        rollouts_per_task=3,
        seed=-1,
        max_concurrency=2,
    )

    expected_qa_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
    expected_corpus_sha256 = corpus_digest(load_corpus_jsonl(corpus))
    assert result.qa_sha256 == expected_qa_sha256
    assert result.corpus_sha256 == expected_corpus_sha256
    assert result.trajectory_count == 3
    assert result.accepted_trajectory_count == 3
    assert result.learning_signal_group_rate == 1.0
    with SQLiteTrajectoryStore(output / "trajectories.db") as store:
        run = store.get_run(result.run_id)
    assert run is not None
    assert run.status is RunStatus.COMPLETED
    assert run.seed == -1
    assert run.config["qa_sha256"] == expected_qa_sha256
    assert run.config["corpus_sha256"] == expected_corpus_sha256


@pytest.mark.asyncio
async def test_offline_pipeline_rejects_rollout_count_that_cannot_meet_uniqueness(
    tmp_path: Path,
) -> None:
    data, corpus = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="between 2 and 4"):
        await run_offline_pipeline(
            tmp_path / "run",
            data,
            corpus,
            rollouts_per_task=5,
        )

    assert not (tmp_path / "run").exists()


def test_offline_pipeline_cli_rejects_corpus_without_answer_evidence(tmp_path: Path) -> None:
    data, corpus = _write_inputs(tmp_path)
    corpus.write_text(
        '{"id":"unrelated","contents":'
        '"France has Parisian museums, but this does not state its capital."}\n',
        encoding="utf-8",
    )
    output = tmp_path / "run"

    result = CliRunner().invoke(
        app,
        ["offline-pipeline", str(data), str(corpus), str(output)],
    )
    rendered = result.stdout + result.stderr

    assert result.exit_code != 0
    assert "no answer-bearing evidence" in rendered
    assert "Traceback" not in rendered
    assert not output.exists()
    assert not list(tmp_path.glob(".arf-*"))


@pytest.mark.asyncio
async def test_seeded_policy_requires_answer_bearing_tool_evidence() -> None:
    policy = SeededAnswerPolicy(
        {"What is the capital of France?": ("Paris",)},
        {"What is the capital of France?": "Berlin"},
    )
    messages = (
        Message(role=MessageRole.USER, content="What is the capital of France?"),
        Message(
            role=MessageRole.TOOL,
            name="search",
            tool_call_id="call-1",
            content=json.dumps([{"id": "unrelated", "score": 1.0, "content": "Saturn has moons."}]),
            metadata={"ok": True},
        ),
    )

    with pytest.raises(ValueError, match="does not support an accepted answer"):
        await policy.generate(messages, (), GenerationRequest(seed=1))


@pytest.mark.asyncio
async def test_seeded_policy_preserves_zero_and_negative_rollout_seed_semantics() -> None:
    question = "What is the capital of France?"
    policy = SeededAnswerPolicy({question: ("Paris",)}, {question: "Berlin"})
    messages = (
        Message(role=MessageRole.USER, content=question),
        Message(
            role=MessageRole.TOOL,
            name="search",
            tool_call_id="call-1",
            content='[{"id":"fr","score":1.0,"content":"Paris is the capital."}]',
            metadata={"ok": True},
        ),
    )

    negative_rollout = await policy.generate(messages, (), GenerationRequest(seed=0))
    zero_rollout = await policy.generate(messages, (), GenerationRequest(seed=1))

    assert negative_rollout.action.final_answer == "Berlin"
    assert zero_rollout.action.final_answer == "Paris"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows path-length regression test")
async def test_offline_pipeline_uses_short_staging_for_deep_windows_output(
    tmp_path: Path,
) -> None:
    data, corpus = _write_inputs(tmp_path)
    padding_length = max(1, 125 - len(str(tmp_path)) - len(str(Path("out"))) - 2)
    output = tmp_path / ("x" * padding_length) / "out"
    assert len(str(output)) >= 125

    result = await run_offline_pipeline(
        output,
        data,
        corpus,
        rollouts_per_task=2,
        max_concurrency=2,
    )

    assert result.trainer_batch_valid is True
    assert output.is_dir()
    assert not list(output.parent.glob(".arf-*"))


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows path-length regression test")
async def test_offline_pipeline_rejects_windows_output_whose_artifacts_are_unreadable(
    tmp_path: Path,
) -> None:
    data, corpus = _write_inputs(tmp_path)
    padding_length = max(1, 150 - len(str(tmp_path)) - len(str(Path("out"))) - 2)
    output = tmp_path / ("x" * padding_length) / "out"
    assert len(str(output)) >= 150

    with pytest.raises(ValueError, match="too long for portable Windows artifacts"):
        await run_offline_pipeline(output, data, corpus, rollouts_per_task=2)

    assert not output.exists()
    assert not output.parent.exists()


@pytest.mark.asyncio
async def test_offline_pipeline_closes_sqlite_when_shard_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, corpus = _write_inputs(tmp_path)
    output = tmp_path / "run"
    staging = tmp_path / ".controlled-staging"
    closed = False
    original_close = offline_module.SQLiteTrajectoryStore.close

    def create_controlled_staging(target: Path) -> Path:
        del target
        staging.mkdir()
        return staging

    def fail_shard_initialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("shard initialization failed")

    def track_close(store: offline_module.SQLiteTrajectoryStore) -> None:
        nonlocal closed
        original_close(store)
        closed = True

    monkeypatch.setattr(offline_module, "_create_staging_directory", create_controlled_staging)
    monkeypatch.setattr(offline_module, "ShardedTrajectoryStore", fail_shard_initialization)
    monkeypatch.setattr(offline_module.SQLiteTrajectoryStore, "close", track_close)

    with pytest.raises(OSError, match="shard initialization failed"):
        await run_offline_pipeline(output, data, corpus, rollouts_per_task=2)

    assert closed is True
    assert not staging.exists()
    assert not output.exists()

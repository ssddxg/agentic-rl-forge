from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tempfile
import time
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import orjson
from pydantic import Field

from agentic_rl_forge import __version__
from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    ContractModel,
    Message,
    MessageRole,
    RunKind,
    RunManifest,
    RunStatus,
    TaskSpec,
    ToolCall,
    ToolSpec,
)
from agentic_rl_forge.data.corpus import corpus_digest, load_corpus_jsonl
from agentic_rl_forge.environments import InMemorySearchTool, LocalToolEnvironment
from agentic_rl_forge.evaluation import BenchmarkAggregator
from agentic_rl_forge.integrations import TrainerBatchExporter, build_search_r1_task
from agentic_rl_forge.rewards import CostReward, ExactMatchOutcome, RewardEngine
from agentic_rl_forge.rollout import (
    AgentLoop,
    GenerationRequest,
    MetricsRolloutCallback,
    PolicyOutput,
    RolloutScheduler,
    ShardedRolloutCallback,
    SignalAwareRolloutFilter,
    SQLiteRolloutCallback,
)
from agentic_rl_forge.services import MetricsRegistry
from agentic_rl_forge.storage import (
    LocalBlobStore,
    ShardedTrajectoryStore,
    SQLiteTrajectoryStore,
    export_trajectories_jsonl,
)

_WINDOWS_LEGACY_PATH_LIMIT = 259
_LONGEST_OFFLINE_ARTIFACT = (
    Path("shards") / "runs" / f"run_{'0' * 32}" / "shards" / f"{'0' * 64}.json"
)


class OfflinePipelineResult(ContractModel):
    run_id: str
    qa_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_count: int = Field(ge=1)
    accepted_trajectory_count: int = Field(ge=1)
    group_pass_rate: float
    learning_signal_group_rate: float
    shard_manifest_id: str
    trainer_batch_id: str
    trainer_batch_valid: bool
    metrics: dict[str, dict[str, float]]
    artifacts: dict[str, str]


class SeededAnswerPolicy:
    """Deterministic policy used to verify the complete offline data path.

    This deliberately alternates accepted and rejected answers. It validates storage,
    evaluation, filtering, and trainer export; it is not intended to measure model quality.
    """

    version = "offline-seeded-policy-v3"

    def __init__(
        self,
        accepted_answers: Mapping[str, tuple[str, ...]],
        incorrect_answers: Mapping[str, str],
    ) -> None:
        self._accepted_answers = dict(accepted_answers)
        self._incorrect_answers = dict(incorrect_answers)

    async def generate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        request: GenerationRequest,
    ) -> PolicyOutput:
        del tools
        question = next(
            message.content for message in reversed(messages) if message.role is MessageRole.USER
        )
        if question not in self._accepted_answers:
            raise ValueError(f"offline policy does not have an answer for question {question!r}")
        if not any(message.role is MessageRole.TOOL for message in messages):
            raw_text = f"<think>Find supporting evidence.</think><search>{question}</search>"
            return PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.TOOL,
                    reasoning="Find supporting evidence.",
                    tool_calls=(ToolCall(name="search", arguments={"query": question}),),
                    raw_text=raw_text,
                ),
                generated_token_count=12,
                policy_logprobs=(-0.1,) * 12,
                finish_reason="tool_calls",
                model="offline-seeded-policy",
            )
        evidence = _tool_evidence_contents(messages)
        supported_answer = _supported_answer(evidence, self._accepted_answers[question])
        if supported_answer is None:
            raise ValueError(
                f"retrieved evidence does not support an accepted answer for {question!r}"
            )
        rollout_seed = (request.seed if request.seed is not None else 1) - 1
        answer = supported_answer if rollout_seed % 2 == 0 else self._incorrect_answers[question]
        raw_text = f"<think>Use the retrieved evidence.</think><answer>{answer}</answer>"
        return PolicyOutput(
            action=AgentAction(
                kind=ActionKind.FINAL,
                reasoning="Use the retrieved evidence.",
                final_answer=answer,
                raw_text=raw_text,
            ),
            generated_token_count=8,
            policy_logprobs=(-0.1,) * 8,
            finish_reason="stop",
            model="offline-seeded-policy",
        )


def _normalized_evidence_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains_supported_answer(content: str, answer: str) -> bool:
    if any(
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uac00" <= character <= "\ud7af"
        or "\uf900" <= character <= "\ufaff"
        for character in answer
    ):
        return answer in content
    return re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", content) is not None


def _supported_answer(contents: Sequence[str], answers: Sequence[str]) -> str | None:
    normalized_contents = tuple(_normalized_evidence_text(content) for content in contents)
    for answer in answers:
        normalized_answer = _normalized_evidence_text(answer)
        if normalized_answer and any(
            _contains_supported_answer(content, normalized_answer)
            for content in normalized_contents
        ):
            return answer
    return None


def _tool_evidence_contents(messages: Sequence[Message]) -> tuple[str, ...]:
    contents: list[str] = []
    for message in messages:
        if message.role is not MessageRole.TOOL:
            continue
        if message.metadata.get("ok") is False:
            raise ValueError("offline policy received a failed search observation")
        try:
            payload = orjson.loads(message.content)
        except orjson.JSONDecodeError as error:
            raise ValueError("offline policy received invalid JSON search evidence") from error
        if not isinstance(payload, list):
            raise ValueError("offline policy expected search evidence to be a JSON array")
        for result in payload:
            if not isinstance(result, dict) or not isinstance(result.get("content"), str):
                raise ValueError("offline policy received a malformed search result")
            contents.append(result["content"])
    return tuple(contents)


def _load_offline_tasks_with_digest(path: Path) -> tuple[tuple[TaskSpec, ...], str]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read QA dataset {path}") from error
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"could not read QA dataset {path}: content is not valid UTF-8") from error
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        task_id = record.get("id")
        question = record.get("question")
        answers = record.get("answer")
        if isinstance(task_id, bool) or not isinstance(task_id, str | int):
            raise ValueError(f"{path}:{line_number} has an invalid id")
        normalized_id = str(task_id).strip()
        if not normalized_id:
            raise ValueError(f"{path}:{line_number} has an empty id")
        if normalized_id in seen_ids:
            raise ValueError(f"{path}:{line_number} repeats id {normalized_id!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{line_number} has an empty or invalid question")
        normalized_question = question.strip()
        if normalized_question in seen_questions:
            raise ValueError(f"{path}:{line_number} repeats question {normalized_question!r}")
        if isinstance(answers, str):
            normalized_answers: str | list[str] = answers.strip()
            if not normalized_answers:
                raise ValueError(f"{path}:{line_number} has an empty answer")
        elif (
            isinstance(answers, list)
            and answers
            and all(isinstance(item, str) and item.strip() for item in answers)
        ):
            normalized_answers = [item.strip() for item in answers]
        else:
            raise ValueError(f"{path}:{line_number} has an invalid answer")
        seen_ids.add(normalized_id)
        seen_questions.add(normalized_question)
        records.append(
            {
                "id": normalized_id,
                "question": normalized_question,
                "answer": normalized_answers,
            }
        )
    if not records:
        raise ValueError(f"QA dataset {path} contains no records")
    tasks = tuple(
        build_search_r1_task(
            task_id=str(record["id"]),
            question=str(record["question"]),
            answers=record["answer"],
            metadata={"dataset": "offline-pipeline"},
        )
        for record in records
    )
    return tasks, hashlib.sha256(payload).hexdigest()


def load_offline_tasks(path: Path) -> tuple[TaskSpec, ...]:
    tasks, _ = _load_offline_tasks_with_digest(path)
    return tasks


def _answer_maps(
    tasks: Sequence[TaskSpec],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    accepted_by_question: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        question = next(
            message.content for message in task.messages if message.role is MessageRole.USER
        )
        configured = task.verifier.config.get("answer")
        answers = [configured] if isinstance(configured, str) else configured
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(item, str) and item for item in answers)
        ):
            raise ValueError(f"task {task.task_id!r} does not have a usable exact-match answer")
        accepted_by_question[question] = tuple(answers)
    answer_pool = tuple(
        dict.fromkeys(
            answer
            for accepted_answers in accepted_by_question.values()
            for answer in accepted_answers
        )
    )
    incorrect: dict[str, str] = {}
    for question, accepted in accepted_by_question.items():
        distractor = next((answer for answer in answer_pool if answer not in accepted), None)
        if distractor is None:
            distractor = f"__incorrect_answer_{len(incorrect) + 1}__"
        incorrect[question] = distractor
    return accepted_by_question, incorrect


def _validate_search_evidence(
    search_tool: InMemorySearchTool,
    accepted_answers: Mapping[str, tuple[str, ...]],
) -> None:
    unsupported: list[str] = []
    for question, answers in accepted_answers.items():
        results = search_tool.search(question)
        contents = tuple(str(result["content"]) for result in results)
        if _supported_answer(contents, answers) is None:
            unsupported.append(question)
    if unsupported:
        questions = ", ".join(repr(question) for question in unsupported[:3])
        remainder = len(unsupported) - 3
        suffix = f" (and {remainder} more)" if remainder > 0 else ""
        raise ValueError(
            "corpus search found no answer-bearing evidence for "
            f"{questions}{suffix}; add relevant corpus content or improve the questions"
        )


async def _run_in_directory(
    output_dir: Path,
    *,
    tasks: tuple[TaskSpec, ...],
    qa_sha256: str,
    corpus_path: Path,
    rollouts_per_task: int,
    seed: int,
    max_concurrency: int,
) -> OfflinePipelineResult:
    database = output_dir / "trajectories.db"
    shards_root = output_dir / "shards"
    filtered_path = output_dir / "filtered-trajectories.jsonl"
    report_path = output_dir / "benchmark-report.json"
    trainer_root = output_dir / "trainer-store"
    metrics = MetricsRegistry()
    accepted_answers, incorrect_answers = _answer_maps(tasks)
    documents = load_corpus_jsonl(corpus_path)
    corpus_sha256 = corpus_digest(documents)
    search_tool = InMemorySearchTool(
        {document.document_id: document.contents for document in documents}
    )
    _validate_search_evidence(search_tool, accepted_answers)
    run_manifest = RunManifest(
        name="offline-pipeline",
        kind=RunKind.ROLLOUT,
        config={
            "rollouts_per_task": rollouts_per_task,
            "seed": seed,
            "task_count": len(tasks),
            "corpus_document_count": len(documents),
            "qa_sha256": qa_sha256,
            "corpus_sha256": corpus_sha256,
        },
        seed=seed,
        policy_version=SeededAnswerPolicy.version,
        environment_version="local-tools-v1",
        package_version=__version__,
        benchmark="offline-pipeline",
    )
    sqlite_store = SQLiteTrajectoryStore(database)
    try:
        sqlite_store.create_run(run_manifest)
        shard_store = ShardedTrajectoryStore(shards_root, run_id=run_manifest.run_id)
        shard_callback = ShardedRolloutCallback(shard_store)

        def loop_factory() -> AgentLoop:
            return AgentLoop(
                policy=SeededAnswerPolicy(accepted_answers, incorrect_answers),
                environment=LocalToolEnvironment((search_tool,)),
                rewards=RewardEngine((ExactMatchOutcome(), CostReward())),
            )

        scheduler = RolloutScheduler(
            loop_factory,
            max_concurrency=max_concurrency,
            callbacks=(
                SQLiteRolloutCallback(sqlite_store, run_id=run_manifest.run_id),
                shard_callback,
                MetricsRolloutCallback(metrics),
            ),
        )
        batch = await scheduler.collect(
            tasks,
            rollouts_per_task=rollouts_per_task,
            seed=seed,
        )
        report = BenchmarkAggregator().aggregate(
            batch.trajectories,
            benchmark="offline-pipeline",
            run_id=run_manifest.run_id,
        )
        report_path.write_bytes(report.canonical_bytes() + b"\n")
        filtered = SignalAwareRolloutFilter(
            expected_group_size=rollouts_per_task,
            min_reward_stddev=0.01,
            min_unique_trajectory_ratio=0.5,
        ).filter(
            batch.trajectories,
            expected_policy_version=batch.policy_version,
        )
        if not filtered.accepted:
            raise ValueError(
                "offline pipeline did not produce learning signal; "
                "use 2 to 4 rollouts per task and verify the corpus evidence"
            )
        export_trajectories_jsonl(filtered.accepted, filtered_path)
        trainer = TrainerBatchExporter(LocalBlobStore(trainer_root))
        trainer_manifest = trainer.export(
            filtered.accepted,
            expected_policy_version=batch.policy_version,
            expected_group_size=rollouts_per_task,
            source_run_id=run_manifest.run_id,
            metadata={"filter": "signal-aware"},
        )
        shard_manifest = shard_callback.manifest
        if shard_manifest is None:
            raise RuntimeError("sharded rollout callback did not finalize a manifest")
        trainer_batch_valid = trainer.verify(trainer_manifest).valid
        if not trainer_batch_valid:
            raise ValueError("offline pipeline produced an invalid trainer batch")
        snapshot = metrics.snapshot()
        result = OfflinePipelineResult(
            run_id=run_manifest.run_id,
            qa_sha256=qa_sha256,
            corpus_sha256=corpus_sha256,
            trajectory_count=len(batch.trajectories),
            accepted_trajectory_count=len(filtered.accepted),
            group_pass_rate=report.metrics["group_pass_rate"].value,
            learning_signal_group_rate=report.metrics["learning_signal_group_rate"].value,
            shard_manifest_id=shard_manifest.manifest_id,
            trainer_batch_id=trainer_manifest.batch_id,
            trainer_batch_valid=trainer_batch_valid,
            metrics={
                "counters": dict(snapshot.counters),
                "gauges": dict(snapshot.gauges),
            },
            artifacts={
                "database": "trajectories.db",
                "filtered_trajectories": "filtered-trajectories.jsonl",
                "benchmark_report": "benchmark-report.json",
                "shards": "shards",
                "trainer_store": "trainer-store",
            },
        )
        sqlite_store.finish_run(run_manifest.run_id)
        return result
    except BaseException:
        with suppress(ValueError):
            sqlite_store.finish_run(run_manifest.run_id, status=RunStatus.FAILED)
        raise
    finally:
        sqlite_store.close()


async def run_offline_pipeline(
    output_dir: Path,
    data_path: Path,
    corpus_path: Path,
    *,
    rollouts_per_task: int = 4,
    seed: int = 0,
    max_concurrency: int = 8,
) -> OfflinePipelineResult:
    if not 2 <= rollouts_per_task <= 4:
        raise ValueError("rollouts_per_task must be between 2 and 4 to create learning signal")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    target, staging = await asyncio.to_thread(_prepare_output_directory, output_dir)
    try:
        tasks, qa_sha256 = await asyncio.to_thread(
            _load_offline_tasks_with_digest,
            data_path,
        )
        result = await _run_in_directory(
            staging,
            tasks=tasks,
            qa_sha256=qa_sha256,
            corpus_path=corpus_path,
            rollouts_per_task=rollouts_per_task,
            seed=seed,
            max_concurrency=max_concurrency,
        )
        await asyncio.to_thread(staging.replace, target)
        return result.model_copy(
            update={
                "artifacts": {
                    name: str(target / relative_path)
                    for name, relative_path in result.artifacts.items()
                }
            }
        )
    except BaseException:
        await asyncio.to_thread(_remove_staging_directory, staging)
        raise


def _prepare_output_directory(output_dir: Path) -> tuple[Path, Path]:
    target = output_dir.resolve()
    _validate_output_path_length(target)
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = _create_staging_directory(target)
    return target, staging


def _validate_output_path_length(target: Path) -> None:
    if os.name != "nt":
        return
    longest_path = target / _LONGEST_OFFLINE_ARTIFACT
    target_units = len(str(target).encode("utf-16-le")) // 2
    longest_units = len(str(longest_path).encode("utf-16-le")) // 2
    if longest_units <= _WINDOWS_LEGACY_PATH_LIMIT:
        return
    relative_units = len(str(_LONGEST_OFFLINE_ARTIFACT).encode("utf-16-le")) // 2
    maximum_target_units = _WINDOWS_LEGACY_PATH_LIMIT - relative_units - 1
    raise ValueError(
        "output path is too long for portable Windows artifacts: "
        f"use an absolute output path of at most {maximum_target_units} UTF-16 characters "
        f"(received {target_units}); choose a shorter directory such as C:\\arf-runs\\run-1"
    )


def _create_staging_directory(target: Path) -> Path:
    candidates = [target.parent]
    if os.name == "nt":
        preferred = [Path.home(), Path(tempfile.gettempdir())]
        ancestors = list(reversed(target.parent.parents))
        candidates = sorted(preferred, key=lambda path: len(str(path))) + ancestors + candidates

    target_device = target.parent.stat().st_dev
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            identity = os.path.normcase(str(resolved))
            if identity in seen or not resolved.is_dir():
                continue
            seen.add(identity)
            if resolved.stat().st_dev != target_device:
                continue
            return Path(tempfile.mkdtemp(prefix=".arf-", dir=resolved))
        except OSError:
            continue
    raise OSError(f"could not create a staging directory on the same volume as {target}")


def _remove_staging_directory(staging: Path) -> None:
    for attempt in range(6):
        try:
            shutil.rmtree(staging)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2**attempt))

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, Literal
from uuid import uuid4

import orjson
from fastapi import APIRouter, File, Form, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from agentic_rl_forge import __version__
from agentic_rl_forge.contracts import ActionKind, AgentAction, ToolCall
from agentic_rl_forge.data import load_corpus_jsonl
from agentic_rl_forge.environments import InMemorySearchTool, LocalToolEnvironment
from agentic_rl_forge.integrations import build_search_r1_task
from agentic_rl_forge.pipelines import load_offline_tasks, run_offline_pipeline
from agentic_rl_forge.quality import DoctorProfile, run_doctor
from agentic_rl_forge.rewards import (
    CostReward,
    ExactMatchOutcome,
    RewardEngine,
    ToolExecutionReward,
)
from agentic_rl_forge.rollout import AgentLoop, PolicyOutput, ScriptedPolicy

from .api import StudioAPIError, _services

LOGGER = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^rl_[0-9a-f]{16}$")
_DATASET_ID = re.compile(r"^(?:sample|data_[0-9a-f]{16})$")
MAX_DATASET_FILE_BYTES = 8 * 1024 * 1024
_QA_SAMPLE = (
    b'{"id":"capital-france","question":"What is the capital of France?",'
    b'"answer":"Paris"}\n'
    b'{"id":"capital-germany","question":"What is the capital of Germany?",'
    b'"answer":"Berlin"}\n'
)
_CORPUS_SAMPLE = (
    b'{"id":"france","contents":"Paris is the capital and largest city of France."}\n'
    b'{"id":"germany","contents":"Berlin is the capital and largest city of Germany."}\n'
    b'{"id":"lyon","contents":"Lyon is a major city in France but is not the '
    b'national capital."}\n'
    b'{"id":"munich","contents":"Munich is the capital of Bavaria but not the '
    b'capital of Germany."}\n'
)


class RLRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OfflineRunCreate(_StrictModel):
    rollouts_per_task: int = Field(default=4, alias="rolloutsPerTask", ge=2, le=4)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    max_concurrency: int = Field(default=4, alias="maxConcurrency", ge=1, le=32)
    dataset_id: str = Field(
        default="sample",
        alias="datasetId",
        pattern=r"^(?:sample|data_[0-9a-f]{16})$",
    )


class RLDatasetRecord(_StrictModel):
    id: str = Field(pattern=r"^(?:sample|data_[0-9a-f]{16})$")
    name: str = Field(min_length=1, max_length=120)
    task_count: int = Field(alias="taskCount", ge=1)
    document_count: int = Field(alias="documentCount", ge=1)
    qa_sha256: str = Field(alias="qaSha256", pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(alias="corpusSha256", pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(alias="sizeBytes", ge=1)
    built_in: bool = Field(default=False, alias="builtIn")
    created_at: datetime = Field(alias="createdAt")


class RLRunRecord(_StrictModel):
    id: str = Field(pattern=r"^rl_[0-9a-f]{16}$")
    kind: Literal["offline_pipeline"] = "offline_pipeline"
    title: str = Field(min_length=1, max_length=120)
    status: RLRunStatus
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    config: OfflineRunCreate
    result: dict[str, Any] | None = None
    error: str | None = Field(default=None, max_length=2_000)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sample_dataset() -> RLDatasetRecord:
    return RLDatasetRecord(
        id="sample",
        name="内置 Search-R1 入门数据",
        taskCount=2,
        documentCount=4,
        qaSha256=hashlib.sha256(_QA_SAMPLE).hexdigest(),
        corpusSha256=hashlib.sha256(_CORPUS_SAMPLE).hexdigest(),
        sizeBytes=len(_QA_SAMPLE) + len(_CORPUS_SAMPLE),
        builtIn=True,
        createdAt=datetime(2026, 9, 23, tzinfo=timezone.utc),
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_error_text(error: Exception, *private_paths: Path) -> str:
    """Keep actionable details without disclosing absolute Studio data paths."""

    message = str(error).strip() or type(error).__name__
    for path in sorted(private_paths, key=lambda item: len(str(item)), reverse=True):
        for value in sorted({str(path), path.as_posix()}, key=len, reverse=True):
            if value:
                message = message.replace(value, path.name)
    return message


class RLWorkspace:
    """Persist and execute laptop-safe Agent RL workflows for Studio."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.runs = self.root / "runs"
        self.datasets = self.root / "datasets"
        self._lock = RLock()

    def ensure(self) -> RLWorkspace:
        self.runs.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        return self

    def recover_interrupted(self) -> int:
        recovered = 0
        with self._lock:
            for metadata in self.runs.glob("rl_*/run.json"):
                try:
                    record = self._read_record(metadata)
                except (OSError, ValueError):
                    LOGGER.warning("Ignoring unreadable RL run metadata at %s", metadata)
                    continue
                if record.status not in {RLRunStatus.QUEUED, RLRunStatus.RUNNING}:
                    continue
                self._save_record(
                    record.model_copy(
                        update={
                            "status": RLRunStatus.FAILED,
                            "updated_at": _utc_now(),
                            "error": "上次运行被中断, 请重新启动这项验证。",
                        }
                    )
                )
                recovered += 1
        return recovered

    def create_offline_run(self, config: OfflineRunCreate) -> RLRunRecord:
        if self.get_dataset(config.dataset_id) is None:
            raise ValueError("RL dataset does not exist")
        now = _utc_now()
        record = RLRunRecord(
            id=f"rl_{uuid4().hex[:16]}",
            title="本地离线 RL 数据管线",
            status=RLRunStatus.QUEUED,
            createdAt=now,
            updatedAt=now,
            config=config,
        )
        with self._lock:
            run_dir = self._run_dir(record.id)
            run_dir.mkdir(parents=False, exist_ok=False)
            self._save_record(record)
        return record

    def create_dataset(
        self,
        name: str,
        qa_payload: bytes,
        corpus_payload: bytes,
    ) -> RLDatasetRecord:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("dataset name cannot be empty")
        if len(normalized_name) > 120:
            raise ValueError("dataset name is too long")
        if not qa_payload or not corpus_payload:
            raise ValueError("dataset files cannot be empty")
        dataset_id = f"data_{uuid4().hex[:16]}"
        staging = self.datasets / f".{dataset_id}.{uuid4().hex}.tmp"
        destination = self._dataset_dir(dataset_id)
        try:
            staging.mkdir(parents=False, exist_ok=False)
            qa_path = staging / "qa.jsonl"
            corpus_path = staging / "corpus.jsonl"
            qa_path.write_bytes(qa_payload)
            corpus_path.write_bytes(corpus_payload)
            try:
                tasks = load_offline_tasks(qa_path)
                documents = load_corpus_jsonl(corpus_path)
            except ValueError as error:
                raise ValueError(_safe_error_text(error, qa_path, corpus_path, staging)) from None
            record = RLDatasetRecord(
                id=dataset_id,
                name=normalized_name,
                taskCount=len(tasks),
                documentCount=len(documents),
                qaSha256=hashlib.sha256(qa_payload).hexdigest(),
                corpusSha256=hashlib.sha256(corpus_payload).hexdigest(),
                sizeBytes=len(qa_payload) + len(corpus_payload),
                createdAt=_utc_now(),
            )
            with self._lock:
                for existing in self.list_datasets(include_sample=False):
                    if (
                        existing.qa_sha256 == record.qa_sha256
                        and existing.corpus_sha256 == record.corpus_sha256
                    ):
                        raise FileExistsError("the same RL dataset is already present")
                _write_atomic(
                    staging / "dataset.json",
                    orjson.dumps(
                        record.model_dump(mode="json", by_alias=True),
                        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
                    )
                    + b"\n",
                )
                staging.replace(destination)
            return record
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def list_datasets(self, *, include_sample: bool = True) -> tuple[RLDatasetRecord, ...]:
        with self._lock:
            records: list[RLDatasetRecord] = []
            for metadata in self.datasets.glob("data_*/dataset.json"):
                try:
                    records.append(self._read_dataset(metadata))
                except (OSError, ValueError):
                    LOGGER.warning("Ignoring unreadable RL dataset metadata at %s", metadata)
            ordered = sorted(
                records,
                key=lambda item: (item.created_at, item.id),
                reverse=True,
            )
            if include_sample:
                return (_sample_dataset(), *ordered)
            return tuple(ordered)

    def get_dataset(self, dataset_id: str) -> RLDatasetRecord | None:
        if dataset_id == "sample":
            return _sample_dataset()
        if not _DATASET_ID.fullmatch(dataset_id):
            return None
        metadata = self._dataset_dir(dataset_id) / "dataset.json"
        if not metadata.is_file():
            return None
        with self._lock:
            try:
                return self._read_dataset(metadata)
            except (OSError, ValueError):
                return None

    def list_runs(self) -> tuple[RLRunRecord, ...]:
        with self._lock:
            records: list[RLRunRecord] = []
            for metadata in self.runs.glob("rl_*/run.json"):
                try:
                    records.append(self._read_record(metadata))
                except (OSError, ValueError):
                    LOGGER.warning("Ignoring unreadable RL run metadata at %s", metadata)
            return tuple(sorted(records, key=lambda item: (item.created_at, item.id), reverse=True))

    def get_run(self, run_id: str) -> RLRunRecord | None:
        if not _RUN_ID.fullmatch(run_id):
            return None
        metadata = self._run_dir(run_id) / "run.json"
        if not metadata.is_file():
            return None
        with self._lock:
            try:
                return self._read_record(metadata)
            except (OSError, ValueError):
                return None

    def fail_run_if_active(self, run_id: str, message: str) -> RLRunRecord | None:
        record = self.get_run(run_id)
        if record is None or record.status not in {RLRunStatus.QUEUED, RLRunStatus.RUNNING}:
            return record
        return self._update_record(
            record,
            status=RLRunStatus.FAILED,
            error=message,
            result=None,
        )

    async def execute_offline_run(self, run_id: str) -> RLRunRecord:
        record = await asyncio.to_thread(self._claim_queued_run, run_id)
        run_dir = self._run_dir(run_id)
        inputs = run_dir / "inputs"
        qa_path = inputs / "qa.jsonl"
        corpus_path = inputs / "corpus.jsonl"
        try:
            await asyncio.to_thread(inputs.mkdir, parents=False, exist_ok=False)
            qa_payload, corpus_payload = await asyncio.to_thread(
                self._dataset_payloads,
                record.config.dataset_id,
            )
            await asyncio.to_thread(_write_atomic, qa_path, qa_payload)
            await asyncio.to_thread(_write_atomic, corpus_path, corpus_payload)
            result = await run_offline_pipeline(
                self.root / run_id,
                qa_path,
                corpus_path,
                rollouts_per_task=record.config.rollouts_per_task,
                seed=record.config.seed,
                max_concurrency=record.config.max_concurrency,
            )
        except asyncio.CancelledError:
            self._update_record(
                record,
                status=RLRunStatus.FAILED,
                error="应用已停止, 本次运行未完成。",
                result=None,
            )
            raise
        except Exception as error:
            LOGGER.exception("Local offline RL pipeline failed for %s", run_id)
            return self._update_record(
                record,
                status=RLRunStatus.FAILED,
                error=(
                    "本地管线运行失败: "
                    f"{_safe_error_text(error, self.root, run_dir, inputs)[:1_900]}"
                ),
                result=None,
            )
        return self._update_record(
            record,
            status=RLRunStatus.SUCCEEDED,
            error=None,
            result=result.model_dump(mode="json"),
        )

    def _claim_queued_run(self, run_id: str) -> RLRunRecord:
        """Atomically claim a persisted run before any asynchronous preparation starts."""

        with self._lock:
            metadata = self._run_dir(run_id) / "run.json"
            if not metadata.is_file():
                raise ValueError("RL run does not exist")
            try:
                record = self._read_record(metadata)
            except (OSError, ValueError) as error:
                raise ValueError("RL run metadata is unreadable") from error
            if record.status is not RLRunStatus.QUEUED:
                raise ValueError("RL run is not queued")
            updated = record.model_copy(
                update={
                    "status": RLRunStatus.RUNNING,
                    "updated_at": _utc_now(),
                    "error": None,
                    "result": None,
                }
            )
            self._save_record(updated)
            return updated

    def overview(self) -> dict[str, object]:
        core = run_doctor(DoctorProfile.CORE)
        training = run_doctor(DoctorProfile.TRAINING)
        records = self.list_runs()
        datasets = self.list_datasets()
        active = sum(
            record.status in {RLRunStatus.QUEUED, RLRunStatus.RUNNING} for record in records
        )
        succeeded = sum(record.status is RLRunStatus.SUCCEEDED for record in records)
        training_ready = training.status.value == "pass"
        return {
            "app": {"name": "AgenticRLForge", "version": __version__},
            "summary": {
                "totalRuns": len(records),
                "activeRuns": active,
                "succeededRuns": succeeded,
                "datasetCount": len(datasets),
                "trainingRuntimeReady": training_ready,
            },
            "diagnostics": {
                "core": _doctor_payload(core),
                "training": _doctor_payload(training),
            },
            "algorithms": [
                {
                    "id": "grpo",
                    "name": "GRPO",
                    "state": "ready",
                    "description": "分组轨迹优势计算与训练批次校验。",
                },
                {
                    "id": "nash-md",
                    "name": "Nash-MD",
                    "state": "ready",
                    "description": "策略混合与无静态参考模型的自博弈工具。",
                },
                {
                    "id": "mcts-prm",
                    "name": "PRM + MCTS",
                    "state": "ready",
                    "description": "过程奖励引导的测试时搜索与计算预算控制。",
                },
                {
                    "id": "verl",
                    "name": "verl 分布式训练",
                    "state": "ready" if training_ready else "setup_required",
                    "description": (
                        "训练运行时已就绪。"
                        if training_ready
                        else "需要另行安装 PyTorch 与 verl, 普通电脑仍可使用本地验证。"
                    ),
                },
            ],
            "workflows": [
                {
                    "id": "trajectory-demo",
                    "name": "智能体轨迹演示",
                    "available": True,
                    "description": "真实执行搜索动作、环境反馈与奖励计算。",
                },
                {
                    "id": "offline-pipeline",
                    "name": "离线 RL 数据管线",
                    "available": True,
                    "description": "生成分组轨迹、评测、筛选并导出训练批次。",
                },
                {
                    "id": "gpu-training",
                    "name": "GPU 权重训练",
                    "available": training_ready,
                    "description": "依赖外部模型服务、PyTorch 与 verl 训练环境。",
                },
            ],
        }

    def _update_record(
        self,
        record: RLRunRecord,
        *,
        status: RLRunStatus,
        error: str | None,
        result: dict[str, Any] | None,
    ) -> RLRunRecord:
        updated = record.model_copy(
            update={
                "status": status,
                "updated_at": _utc_now(),
                "error": error,
                "result": result,
            }
        )
        with self._lock:
            self._save_record(updated)
        return updated

    def _run_dir(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("invalid RL run id")
        return self.runs / run_id

    def _dataset_dir(self, dataset_id: str) -> Path:
        if not _DATASET_ID.fullmatch(dataset_id) or dataset_id == "sample":
            raise ValueError("invalid RL dataset id")
        return self.datasets / dataset_id

    def _dataset_payloads(self, dataset_id: str) -> tuple[bytes, bytes]:
        if dataset_id == "sample":
            return _QA_SAMPLE, _CORPUS_SAMPLE
        record = self.get_dataset(dataset_id)
        if record is None:
            raise ValueError("RL dataset does not exist")
        directory = self._dataset_dir(record.id)
        qa_payload = (directory / "qa.jsonl").read_bytes()
        corpus_payload = (directory / "corpus.jsonl").read_bytes()
        if hashlib.sha256(qa_payload).hexdigest() != record.qa_sha256:
            raise ValueError("RL dataset QA file failed integrity verification")
        if hashlib.sha256(corpus_payload).hexdigest() != record.corpus_sha256:
            raise ValueError("RL dataset corpus file failed integrity verification")
        return qa_payload, corpus_payload

    def _save_record(self, record: RLRunRecord) -> None:
        metadata = self._run_dir(record.id) / "run.json"
        payload = orjson.dumps(
            record.model_dump(mode="json", by_alias=True),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        _write_atomic(metadata, payload + b"\n")

    @staticmethod
    def _read_record(path: Path) -> RLRunRecord:
        return RLRunRecord.model_validate(orjson.loads(path.read_bytes()))

    @staticmethod
    def _read_dataset(path: Path) -> RLDatasetRecord:
        return RLDatasetRecord.model_validate(orjson.loads(path.read_bytes()))


class RLJobRunner:
    def __init__(self, workspace: RLWorkspace) -> None:
        self._workspace = workspace
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._execution_slot = asyncio.Semaphore(1)
        self._closing = False

    def enqueue(self, run_id: str) -> None:
        if self._closing:
            raise RuntimeError("Studio is shutting down")
        if run_id in self._tasks:
            raise RuntimeError("RL run is already active")
        task = asyncio.create_task(self._run(run_id), name=f"studio-rl-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda completed: self._finished(run_id, completed))

    async def close(self) -> None:
        self._closing = True
        pending = tuple(self._tasks.items())
        for _, task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*(task for _, task in pending), return_exceptions=True)
            for run_id, _ in pending:
                await asyncio.to_thread(
                    self._workspace.fail_run_if_active,
                    run_id,
                    "应用已停止, 本次运行未完成。",
                )

    async def _run(self, run_id: str) -> None:
        async with self._execution_slot:
            await self._workspace.execute_offline_run(run_id)

    def _finished(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(run_id, None)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            LOGGER.exception("Unexpected RL workflow failure for %s", run_id)


def _doctor_payload(report: Any) -> dict[str, object]:
    return {
        "profile": report.profile.value,
        "status": report.status.value,
        "pythonVersion": report.python_version,
        "checks": [
            {
                "name": check.name,
                "status": check.status.value,
                "purpose": check.purpose,
                "detail": check.detail,
                "remediation": check.remediation,
                "required": check.required,
            }
            for check in report.checks
        ],
    }


def _run_payload(record: RLRunRecord) -> dict[str, object]:
    return record.model_dump(mode="json", by_alias=True)


def _dataset_payload(record: RLDatasetRecord) -> dict[str, object]:
    return record.model_dump(mode="json", by_alias=True)


async def _read_dataset_upload(upload: UploadFile, label: str) -> bytes:
    filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if Path(filename).suffix.lower() != ".jsonl":
        raise StudioAPIError(415, "invalid_dataset_file", f"{label} 必须是 .jsonl 文件。")
    if upload.size is not None and upload.size > MAX_DATASET_FILE_BYTES:
        raise StudioAPIError(413, "dataset_file_too_large", f"{label} 不能超过 8 MB。")
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_DATASET_FILE_BYTES:
            raise StudioAPIError(413, "dataset_file_too_large", f"{label} 不能超过 8 MB。")
        chunks.append(chunk)
    if not chunks:
        raise StudioAPIError(422, "empty_dataset_file", f"{label} 不能为空。")
    return b"".join(chunks)


def _rl_workspace(request: Request) -> RLWorkspace:
    workspace = getattr(_services(request), "rl_workspace", None)
    if not isinstance(workspace, RLWorkspace):
        raise StudioAPIError(
            503,
            "rl_workspace_unavailable",
            "强化学习工作台尚未初始化, 请重新启动应用。",
        )
    return workspace


async def run_trajectory_demo() -> dict[str, object]:
    search = InMemorySearchTool(
        {
            "france": "Paris is the capital and largest city of France.",
            "germany": "Berlin is the capital and largest city of Germany.",
        }
    )
    task = build_search_r1_task(
        task_id="studio-trajectory-demo",
        question="What is the capital of France?",
        answers="Paris",
    ).model_copy(update={"tools": (search.spec,)})
    policy = ScriptedPolicy(
        (
            PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.TOOL,
                    reasoning="先调用搜索工具获取可核验的证据。",
                    tool_calls=(ToolCall(name="search", arguments={"query": "capital of France"}),),
                    raw_text="<think>Find evidence.</think><search>capital of France</search>",
                ),
                generated_token_count=12,
                model="studio-scripted-policy",
            ),
            PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.FINAL,
                    reasoning="检索结果直接支持答案。",
                    final_answer="Paris",
                    raw_text="<think>Use evidence.</think><answer>Paris</answer>",
                ),
                generated_token_count=10,
                model="studio-scripted-policy",
            ),
        ),
        version="studio-demo-policy-v1",
    )
    trajectory = await AgentLoop(
        policy=policy,
        environment=LocalToolEnvironment((search,)),
        rewards=RewardEngine((ExactMatchOutcome(), ToolExecutionReward(), CostReward())),
    ).run(task, group_id="studio-demo-group", seed=0)
    return {
        "trajectoryId": trajectory.trajectory_id,
        "taskId": trajectory.task_id,
        "status": trajectory.status.value,
        "policyVersion": trajectory.policy_version,
        "environmentVersion": trajectory.environment_version,
        "summary": {
            "stepCount": len(trajectory.steps),
            "totalReward": trajectory.total_reward,
            "generatedTokens": trajectory.total_generated_tokens,
            "observationTokens": trajectory.total_observation_tokens,
        },
        "steps": [
            {
                "index": step.index,
                "action": step.action.kind.value,
                "reasoning": step.action.reasoning,
                "toolCalls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in step.action.tool_calls
                ],
                "toolResults": [
                    {"name": result.name, "ok": result.ok, "content": result.content}
                    for result in step.tool_results
                ],
                "finalAnswer": step.action.final_answer,
                "reward": step.rewards.total,
                "rewardSignals": [
                    {
                        "name": signal.name,
                        "source": signal.source.value,
                        "value": signal.value,
                        "weightedValue": signal.weighted_value,
                    }
                    for signal in step.rewards.signals
                ],
                "generatedTokens": step.generated_token_count,
                "observationTokens": step.observation_token_count,
            }
            for step in trajectory.steps
        ],
        "finalRewardSignals": [
            {
                "name": signal.name,
                "source": signal.source.value,
                "value": signal.value,
                "weightedValue": signal.weighted_value,
            }
            for signal in trajectory.final_reward.signals
        ],
    }


router = APIRouter(prefix="/rl", tags=["reinforcement-learning"])


@router.get("/overview")
async def rl_overview(request: Request) -> dict[str, object]:
    return await asyncio.to_thread(_rl_workspace(request).overview)


@router.post("/trajectory-demo")
async def trajectory_demo(request: Request) -> dict[str, object]:
    _rl_workspace(request)
    return await run_trajectory_demo()


@router.get("/datasets")
async def list_rl_datasets(request: Request) -> dict[str, object]:
    records = await asyncio.to_thread(_rl_workspace(request).list_datasets)
    return {"items": [_dataset_payload(record) for record in records]}


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
async def create_rl_dataset(
    request: Request,
    name: Annotated[str, Form(min_length=1, max_length=120)],
    qa_file: Annotated[UploadFile, File(alias="qaFile")],
    corpus_file: Annotated[UploadFile, File(alias="corpusFile")],
) -> dict[str, object]:
    workspace = _rl_workspace(request)
    try:
        # These are local spooled files. Sequential reads make cleanup deterministic if one
        # upload is invalid instead of leaving the other gather branch reading a closed file.
        qa_payload = await _read_dataset_upload(qa_file, "问答数据")
        corpus_payload = await _read_dataset_upload(corpus_file, "检索语料")
        try:
            record = await asyncio.to_thread(
                workspace.create_dataset,
                name,
                qa_payload,
                corpus_payload,
            )
        except FileExistsError:
            raise StudioAPIError(409, "duplicate_rl_dataset", "相同的数据集已经存在。") from None
        except ValueError as error:
            raise StudioAPIError(
                422,
                "invalid_rl_dataset",
                f"数据集校验失败: {str(error)[:1_500]}",
            ) from None
    finally:
        await qa_file.close()
        await corpus_file.close()
    return _dataset_payload(record)


@router.get("/offline-runs")
async def list_offline_runs(request: Request) -> dict[str, object]:
    records = await asyncio.to_thread(_rl_workspace(request).list_runs)
    return {"items": [_run_payload(record) for record in records]}


@router.get("/offline-runs/{run_id}")
async def get_offline_run(request: Request, run_id: str) -> dict[str, object]:
    record = await asyncio.to_thread(_rl_workspace(request).get_run, run_id)
    if record is None:
        raise StudioAPIError(404, "rl_run_not_found", "找不到这次强化学习运行。")
    return _run_payload(record)


@router.post("/offline-runs", status_code=status.HTTP_202_ACCEPTED)
async def create_offline_run(
    request: Request,
    payload: OfflineRunCreate,
) -> dict[str, object]:
    services = _services(request)
    workspace = _rl_workspace(request)
    runner = getattr(services, "rl_job_runner", None)
    if not isinstance(runner, RLJobRunner):
        raise StudioAPIError(
            503,
            "rl_runner_unavailable",
            "强化学习任务运行器尚未初始化, 请重新启动应用。",
        )
    try:
        record = await asyncio.to_thread(workspace.create_offline_run, payload)
    except ValueError:
        raise StudioAPIError(404, "rl_dataset_not_found", "找不到所选的数据集。") from None
    try:
        runner.enqueue(record.id)
    except Exception:
        LOGGER.exception("Could not enqueue RL workflow %s", record.id)
        await asyncio.to_thread(
            workspace.fail_run_if_active,
            record.id,
            "任务无法启动, 请重新运行。",
        )
        raise StudioAPIError(500, "rl_run_start_failed", "任务无法启动, 请重试。") from None
    return _run_payload(record)


__all__ = [
    "MAX_DATASET_FILE_BYTES",
    "OfflineRunCreate",
    "RLDatasetRecord",
    "RLJobRunner",
    "RLRunRecord",
    "RLRunStatus",
    "RLWorkspace",
    "router",
    "run_trajectory_demo",
]

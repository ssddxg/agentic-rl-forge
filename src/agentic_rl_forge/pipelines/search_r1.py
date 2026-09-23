from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import Field, model_validator

from agentic_rl_forge import __version__
from agentic_rl_forge.contracts import (
    ContractModel,
    RolloutSlot,
    RunHeartbeat,
    RunKind,
    RunLeaseToken,
    RunManifest,
    RunStatus,
    TaskSpec,
    TrajectoryStatus,
    new_id,
    utc_now,
)
from agentic_rl_forge.environments import HTTPRetrievalTool, LocalToolEnvironment
from agentic_rl_forge.evaluation import BenchmarkAggregator
from agentic_rl_forge.integrations import iter_search_r1_tasks
from agentic_rl_forge.rewards import (
    CompletionGuardReward,
    CostReward,
    ExactMatchOutcome,
    InvalidActionReward,
    RepeatedActionPenalty,
    RewardEngine,
    ToolExecutionReward,
)
from agentic_rl_forge.rollout import (
    AgentLoop,
    MetricsRolloutCallback,
    OpenAICompatiblePolicy,
    RolloutConfig,
    RolloutPlanBuilder,
    RolloutScheduler,
    SearchR1Parser,
    ShardedRolloutCallback,
    SQLiteRolloutCallback,
    validate_planned_trajectory,
)
from agentic_rl_forge.services import MetricsRegistry
from agentic_rl_forge.storage import (
    ClaimStoreConsistency,
    LocalBlobStore,
    RunArtifactBundle,
    RunLeaseConflictError,
    ShardedTrajectoryStore,
    SlotClaimCoordinator,
    SQLiteTrajectoryStore,
)


class SearchR1CollectionConfig(ContractModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    model_base_url: str = Field(min_length=1)
    retrieval_endpoint: str = Field(min_length=1)
    rollouts_per_task: int = Field(default=5, ge=2)
    task_offset: int = Field(default=0, ge=0)
    max_tasks: int | None = Field(default=None, ge=1)
    seed: int = 0
    max_steps: int = Field(default=5, ge=2)
    max_tokens_per_step: int = Field(default=500, ge=1)
    max_concurrency: int = Field(default=32, ge=1)
    model_max_concurrency: int = Field(default=32, ge=1)
    model_timeout_s: float = Field(default=120.0, gt=0)
    model_max_retries: int = Field(default=3, ge=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    retrieval_top_k: int = Field(default=3, ge=1)
    retrieval_timeout_s: float = Field(default=30.0, gt=0)
    repeated_action_limit: int = Field(default=3, ge=1)
    success_reward: float = 1.0
    failure_reward: float = 0.0
    tool_failure_penalty: float = Field(default=-0.1, le=0.0)
    invalid_action_penalty: float = Field(default=-0.25, le=0.0)
    repeated_action_penalty: float = Field(default=-0.2, le=0.0)
    truncated_penalty: float = Field(default=-0.1, le=0.0)
    error_penalty: float = Field(default=-0.5, le=0.0)
    per_generated_token: float = Field(default=-0.00001, le=0.0)
    per_tool_call: float = Field(default=-0.002, le=0.0)
    plan_salt: str = Field(default="", max_length=128)
    run_lease_ttl_s: float = Field(default=60.0, gt=0)
    heartbeat_interval_s: float = Field(default=15.0, gt=0)
    enable_slot_claims: bool = True
    slot_claim_ttl_s: float = Field(default=900.0, gt=0)
    slot_claim_renewal_interval_s: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def validate_endpoints(self) -> SearchR1CollectionConfig:
        for name in ("model_base_url", "retrieval_endpoint"):
            value = getattr(self, name)
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{name} must be an absolute HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{name} cannot contain credentials, a query, or a fragment")
        if self.heartbeat_interval_s > self.run_lease_ttl_s / 2:
            raise ValueError("heartbeat_interval_s must not exceed half of run_lease_ttl_s")
        if self.slot_claim_renewal_interval_s > self.slot_claim_ttl_s / 2:
            raise ValueError(
                "slot_claim_renewal_interval_s must not exceed half of slot_claim_ttl_s"
            )
        return self

    @property
    def plan_config_digest(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={
                "heartbeat_interval_s",
                "enable_slot_claims",
                "max_concurrency",
                "model_max_concurrency",
                "model_max_retries",
                "model_timeout_s",
                "retrieval_timeout_s",
                "run_lease_ttl_s",
                "slot_claim_ttl_s",
                "slot_claim_renewal_interval_s",
            },
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class SearchR1CollectionResult(ContractModel):
    run_id: str
    plan_id: str
    policy_version: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    trajectory_count: int = Field(ge=1)
    collected_trajectory_count: int = Field(ge=0)
    reused_trajectory_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    status_counts: dict[str, int]
    shard_manifest_id: str
    artifacts: dict[str, str]

    @model_validator(mode="after")
    def validate_counts(self) -> SearchR1CollectionResult:
        if self.collected_trajectory_count + self.reused_trajectory_count != self.trajectory_count:
            raise ValueError("collected and reused counts must equal trajectory_count")
        if self.success_count > self.trajectory_count:
            raise ValueError("success_count cannot exceed trajectory_count")
        if sum(self.status_counts.values()) != self.trajectory_count:
            raise ValueError("status counts must equal trajectory_count")
        return self


class SearchR1PlanStatus(ContractModel):
    plan_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    total_slot_count: int = Field(ge=1)
    reusable_slot_count: int = Field(ge=0)
    missing_slot_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    reusable_slots: tuple[RolloutSlot, ...]
    missing_slots: tuple[RolloutSlot, ...]
    conflicting_trajectory_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> SearchR1PlanStatus:
        if self.reusable_slot_count != len(self.reusable_slots):
            raise ValueError("reusable slot count does not match reusable_slots")
        if self.missing_slot_count != len(self.missing_slots):
            raise ValueError("missing slot count does not match missing_slots")
        if self.conflict_count != len(self.conflicting_trajectory_ids):
            raise ValueError("conflict count does not match conflicting_trajectory_ids")
        if (
            self.reusable_slot_count + self.missing_slot_count + self.conflict_count
            != self.total_slot_count
        ):
            raise ValueError("plan status categories must cover every slot")
        return self


def load_search_r1_collection_config(path: Path) -> SearchR1CollectionConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("collection config is not valid YAML") from error
    if not isinstance(payload, dict):
        raise ValueError("collection config must contain a YAML object")
    return SearchR1CollectionConfig.model_validate(payload)


async def inspect_search_r1_plan(
    source: Path,
    output_dir: Path,
    config: SearchR1CollectionConfig,
) -> SearchR1PlanStatus:
    source_path = await asyncio.to_thread(source.resolve)
    output_root = await asyncio.to_thread(output_dir.resolve)
    source_bytes = await asyncio.to_thread(source_path.read_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    tasks = _load_tasks(source_bytes, config)
    plan = RolloutPlanBuilder().build(
        tasks,
        policy_version=config.policy_version,
        source_sha256=source_sha256,
        config_digest=config.plan_config_digest,
        rollouts_per_task=config.rollouts_per_task,
        seed=config.seed,
    )
    database_path = output_root / "trajectories.db"
    reusable: list[RolloutSlot] = []
    missing: list[RolloutSlot] = []
    conflicts: list[str] = []
    database_exists = await asyncio.to_thread(database_path.is_file)
    store = SQLiteTrajectoryStore(database_path) if database_exists else None
    try:
        for slot in plan.slots:
            trajectory = store.get(slot.trajectory_id) if store is not None else None
            if trajectory is None:
                missing.append(slot)
                continue
            try:
                validate_planned_trajectory(plan, trajectory)
            except ValueError:
                conflicts.append(trajectory.trajectory_id)
            else:
                reusable.append(slot)
    finally:
        if store is not None:
            await asyncio.to_thread(store.close)
    return SearchR1PlanStatus(
        plan_id=plan.plan_id,
        source_sha256=source_sha256,
        task_count=len(tasks),
        total_slot_count=len(plan.slots),
        reusable_slot_count=len(reusable),
        missing_slot_count=len(missing),
        conflict_count=len(conflicts),
        reusable_slots=tuple(reusable),
        missing_slots=tuple(missing),
        conflicting_trajectory_ids=tuple(conflicts),
    )


async def collect_search_r1(
    source: Path,
    output_dir: Path,
    config: SearchR1CollectionConfig,
    *,
    api_key: str | None = None,
    model_client: httpx.AsyncClient | None = None,
    retrieval_client: httpx.AsyncClient | None = None,
    slot_claim_coordinator: SlotClaimCoordinator | None = None,
) -> SearchR1CollectionResult:
    output_root = await asyncio.to_thread(output_dir.resolve)
    source_path = await asyncio.to_thread(source.resolve)
    await asyncio.to_thread(output_root.mkdir, parents=True, exist_ok=True)
    source_bytes = await asyncio.to_thread(source_path.read_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    tasks = _load_tasks(source_bytes, config)
    database_path = output_root / "trajectories.db"
    shards_path = output_root / "shards"
    sqlite_store = SQLiteTrajectoryStore(database_path)
    plan = RolloutPlanBuilder().build(
        tasks,
        policy_version=config.policy_version,
        source_sha256=source_sha256,
        config_digest=config.plan_config_digest,
        rollouts_per_task=config.rollouts_per_task,
        seed=config.seed,
    )
    existing_trajectories = tuple(
        trajectory
        for slot in plan.slots
        if (trajectory := sqlite_store.get(slot.trajectory_id)) is not None
    )
    root_artifacts = await asyncio.to_thread(LocalBlobStore, output_root)
    local_claim_path: Path | None = None
    if slot_claim_coordinator is None and config.enable_slot_claims:
        slot_claim_coordinator = SlotClaimCoordinator(
            root_artifacts,
            consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
            prefix="coordination/slot-claims",
        )
        local_claim_path = output_root / "coordination" / "slot-claims"
    plan_key = f"plans/{plan.plan_id}.json"
    await asyncio.to_thread(
        root_artifacts.put_if_absent,
        plan_key,
        plan.canonical_bytes() + b"\n",
    )
    run_manifest = RunManifest(
        name=config.name,
        kind=RunKind.ROLLOUT,
        config={
            **config.model_dump(mode="json", exclude_none=True),
            "source": str(source_path),
            "source_sha256": source_sha256,
            "selected_task_count": len(tasks),
            "api_key_configured": api_key is not None,
            "rollout_plan_id": plan.plan_id,
            "reused_trajectory_count": len(existing_trajectories),
        },
        seed=config.seed,
        policy_version=config.policy_version,
        environment_version="local-tools-v1",
        benchmark=config.dataset_name,
        package_version=__version__,
    )
    sqlite_store.create_run(run_manifest)
    run_owner_id = new_id("worker")
    lease = await asyncio.to_thread(
        sqlite_store.acquire_run_lease,
        run_manifest.run_id,
        owner_id=run_owner_id,
        ttl_s=config.run_lease_ttl_s,
    )
    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _renew_run_lease(
            sqlite_store,
            run_manifest.run_id,
            lease.token,
            initial_heartbeat=lease,
            ttl_s=config.run_lease_ttl_s,
            interval_s=config.heartbeat_interval_s,
            stop=heartbeat_stop,
        )
    )
    run_path = output_root / "runs" / run_manifest.run_id
    artifact_store = await asyncio.to_thread(LocalBlobStore, run_path)
    shard_store = ShardedTrajectoryStore(shards_path, run_id=run_manifest.run_id)
    shard_callback = ShardedRolloutCallback(
        shard_store,
        expected_policy_version=config.policy_version,
    )
    metrics = MetricsRegistry()
    search_tool = HTTPRetrievalTool(
        config.retrieval_endpoint,
        top_k=config.retrieval_top_k,
        timeout_s=config.retrieval_timeout_s,
        client=retrieval_client,
    )
    policy = OpenAICompatiblePolicy(
        base_url=config.model_base_url,
        model=config.model,
        version=config.policy_version,
        parser=SearchR1Parser(),
        api_key=api_key,
        timeout_s=config.model_timeout_s,
        max_concurrency=config.model_max_concurrency,
        max_retries=config.model_max_retries,
        native_tools=False,
        client=model_client,
    )
    rewards = RewardEngine(
        (
            ExactMatchOutcome(
                success_reward=config.success_reward,
                failure_reward=config.failure_reward,
            ),
            ToolExecutionReward(failure_penalty=config.tool_failure_penalty),
            CostReward(
                per_generated_token=config.per_generated_token,
                per_tool_call=config.per_tool_call,
            ),
            InvalidActionReward(penalty=config.invalid_action_penalty),
            RepeatedActionPenalty(
                repeat_threshold=max(1, config.repeated_action_limit - 1),
                penalty=config.repeated_action_penalty,
            ),
            CompletionGuardReward(
                truncated_penalty=config.truncated_penalty,
                error_penalty=config.error_penalty,
            ),
        )
    )

    def loop_factory() -> AgentLoop:
        return AgentLoop(
            policy=policy,
            environment=LocalToolEnvironment((search_tool,)),
            rewards=rewards,
            config=RolloutConfig(
                max_steps=config.max_steps,
                max_tokens_per_step=config.max_tokens_per_step,
                temperature=config.temperature,
                top_p=config.top_p,
                repeated_action_limit=config.repeated_action_limit,
            ),
        )

    scheduler = RolloutScheduler(
        loop_factory,
        max_concurrency=config.max_concurrency,
        callbacks=(
            SQLiteRolloutCallback(
                sqlite_store,
                run_id=run_manifest.run_id,
                lease=lease.token,
            ),
            shard_callback,
            MetricsRolloutCallback(metrics),
        ),
        slot_claims=slot_claim_coordinator,
        claim_owner_id=run_owner_id if slot_claim_coordinator is not None else None,
        claim_ttl_s=config.slot_claim_ttl_s,
        claim_renewal_interval_s=(
            config.slot_claim_renewal_interval_s if slot_claim_coordinator is not None else None
        ),
    )
    run_finished = False
    try:
        collection_task = asyncio.create_task(
            scheduler.collect(
                tasks,
                rollouts_per_task=config.rollouts_per_task,
                seed=config.seed,
                plan=plan,
                existing_trajectories=existing_trajectories,
            )
        )
        done, _ = await asyncio.wait(
            (collection_task, heartbeat_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            collection_task.cancel()
            await asyncio.gather(collection_task, return_exceptions=True)
            await heartbeat_task
            raise RuntimeError("run heartbeat stopped unexpectedly")
        batch = await collection_task
        report = BenchmarkAggregator().aggregate(
            batch.trajectories,
            benchmark=config.dataset_name,
            run_id=run_manifest.run_id,
        )
        shard_manifest = shard_callback.manifest
        if shard_manifest is None:
            raise RuntimeError("sharded rollout callback did not finalize a manifest")
        trajectory_payload = b"".join(
            trajectory.canonical_bytes() + b"\n" for trajectory in batch.trajectories
        )
        await asyncio.to_thread(
            artifact_store.put_if_absent,
            "trajectories.jsonl",
            trajectory_payload,
        )
        await asyncio.to_thread(
            artifact_store.put_if_absent,
            "benchmark-report.json",
            report.canonical_bytes() + b"\n",
        )
        await asyncio.to_thread(
            artifact_store.put_if_absent,
            "metrics.prom",
            metrics.render_prometheus().encode("utf-8"),
        )
        status_counts = dict(
            sorted(Counter(item.status.value for item in batch.trajectories).items())
        )
        await asyncio.to_thread(
            sqlite_store.renew_run_lease,
            run_manifest.run_id,
            lease.token,
            ttl_s=config.run_lease_ttl_s,
        )
        heartbeat_stop.set()
        await heartbeat_task
        finished_manifest = await asyncio.to_thread(
            sqlite_store.finish_run,
            run_manifest.run_id,
            metadata={
                "trajectory_count": len(batch.trajectories),
                "collected_trajectory_count": len(batch.trajectories) - len(existing_trajectories),
                "reused_trajectory_count": len(existing_trajectories),
                "status_counts": status_counts,
                "shard_manifest_id": shard_manifest.manifest_id,
            },
            lease=lease.token,
        )
        run_finished = True
        await asyncio.to_thread(
            artifact_store.put_if_absent,
            "run-manifest.json",
            finished_manifest.canonical_bytes() + b"\n",
        )
        artifacts = {
            "database": str(database_path),
            "shards": str(shards_path),
            "rollout_plan": str(output_root / plan_key),
            "run_manifest": str(run_path / "run-manifest.json"),
            "trajectories": str(run_path / "trajectories.jsonl"),
            "benchmark_report": str(run_path / "benchmark-report.json"),
            "metrics": str(run_path / "metrics.prom"),
            "summary": str(run_path / "summary.json"),
            "artifact_manifest": str(run_path / "artifact-manifest.json"),
        }
        if local_claim_path is not None:
            artifacts["slot_claims"] = str(local_claim_path)
        result = SearchR1CollectionResult(
            run_id=run_manifest.run_id,
            plan_id=plan.plan_id,
            policy_version=batch.policy_version,
            source_sha256=source_sha256,
            task_count=len(tasks),
            trajectory_count=len(batch.trajectories),
            collected_trajectory_count=len(batch.trajectories) - len(existing_trajectories),
            reused_trajectory_count=len(existing_trajectories),
            success_count=sum(
                item.status is TrajectoryStatus.SUCCEEDED for item in batch.trajectories
            ),
            status_counts=status_counts,
            shard_manifest_id=shard_manifest.manifest_id,
            artifacts=artifacts,
        )
        await asyncio.to_thread(
            artifact_store.put_if_absent,
            "summary.json",
            result.canonical_bytes() + b"\n",
        )
        if finished_manifest.completed_at is None:
            raise RuntimeError("terminal run manifest is missing completion time")
        artifact_bundle = RunArtifactBundle(output_root)
        artifact_manifest = await asyncio.to_thread(
            artifact_bundle.build,
            run_id=run_manifest.run_id,
            plan_id=plan.plan_id,
            shard_manifest_id=shard_manifest.manifest_id,
            created_at=finished_manifest.completed_at,
        )
        artifact_verification = await asyncio.to_thread(
            artifact_bundle.verify,
            artifact_manifest,
        )
        if not artifact_verification.valid:
            raise RuntimeError("run artifact manifest failed post-write verification")
        return result
    except BaseException as error:
        heartbeat_stop.set()
        if not heartbeat_task.done():
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if not run_finished:
            with suppress(RunLeaseConflictError):
                await asyncio.to_thread(
                    sqlite_store.finish_run,
                    run_manifest.run_id,
                    status=RunStatus.FAILED,
                    metadata={"error_type": type(error).__name__},
                    lease=lease.token,
                )
        raise
    finally:
        heartbeat_stop.set()
        if not heartbeat_task.done():
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        await policy.close()
        await search_tool.close()
        await asyncio.to_thread(sqlite_store.close)


async def _renew_run_lease(
    store: SQLiteTrajectoryStore,
    run_id: str,
    token: RunLeaseToken,
    *,
    initial_heartbeat: RunHeartbeat,
    ttl_s: float,
    interval_s: float,
    stop: asyncio.Event,
) -> None:
    renewal_interval = timedelta(seconds=interval_s)
    next_renewal_at = initial_heartbeat.heartbeat_at + renewal_interval
    while True:
        if stop.is_set():
            return
        delay_s = min(
            interval_s,
            max(0.0, (next_renewal_at - utc_now()).total_seconds()),
        )
        if delay_s > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay_s)
            except asyncio.TimeoutError:
                pass
            else:
                return
        if stop.is_set():
            return
        heartbeat = await asyncio.to_thread(
            store.renew_run_lease,
            run_id,
            token,
            ttl_s=ttl_s,
        )
        next_renewal_at = heartbeat.heartbeat_at + renewal_interval


def _load_tasks(
    source_bytes: bytes,
    config: SearchR1CollectionConfig,
) -> tuple[TaskSpec, ...]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(source_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"source line {line_number} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"source line {line_number} must contain a JSON object")
        records.append(payload)
    stop = None if config.max_tasks is None else config.task_offset + config.max_tasks
    selected = records[config.task_offset : stop]
    if not selected:
        raise ValueError("task selection is empty")
    return tuple(
        task.model_copy(update={"max_steps": config.max_steps})
        for task in iter_search_r1_tasks(selected, dataset_name=config.dataset_name)
    )

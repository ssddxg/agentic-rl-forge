from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Annotated

import orjson
import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from agentic_rl_forge import __version__
from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    CheckpointArtifact,
    CheckpointManifest,
    DataOrigin,
    DatasetCollectionManifest,
    RolloutPlan,
    RunArtifactGcPlan,
    RunArtifactManifest,
    RunArtifactMirrorBatchHealth,
    RunArtifactMirrorBatchInspection,
    RunArtifactMirrorBatchMemberState,
    RunArtifactMirrorBatchPlan,
    RunArtifactMirrorBatchResolution,
    RunArtifactMirrorBatchResolutionKind,
    RunArtifactMirrorPlan,
    RunKind,
    RunLivenessState,
    RunManifest,
    RunStatus,
    SignedDatasetCollectionManifest,
    SignedRunArtifactArchiveAttestation,
    SlotClaimState,
    ToolCall,
    TrajectoryStatus,
)
from agentic_rl_forge.data import (
    CorpusInspection,
    DatasetManifestBuilder,
    Ed25519ManifestSigner,
    PRMDatasetBuilder,
    RunArtifactArchiveAttestor,
    build_text_corpus,
    export_prm_jsonl,
    inspect_corpus,
)
from agentic_rl_forge.environments import InMemorySearchTool, LocalToolEnvironment
from agentic_rl_forge.evaluation import BenchmarkAggregator, BenchmarkComparator
from agentic_rl_forge.experiments import (
    ExperimentAnalysis,
    ExperimentAnalyzer,
    ExperimentArtifactKind,
    ExperimentExecutionError,
    ExperimentIndexBuilder,
    ExperimentObjective,
    ExperimentObjectiveDirection,
    ExperimentOperationsIndex,
    ExperimentPlan,
    ExperimentPlanError,
    ExperimentPromoter,
    ExperimentPromotionAcquirer,
    ExperimentPromotionAcquisitionError,
    ExperimentPromotionAcquisitionPolicy,
    ExperimentPromotionAliasAction,
    ExperimentPromotionAliasPolicy,
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestor,
    ExperimentPromotionArchiveError,
    ExperimentPromotionError,
    ExperimentPromotionGovernancePolicy,
    ExperimentPromotionPolicy,
    ExperimentPromotionRegistry,
    ExperimentPromotionRegistryError,
    ExperimentPromotionRemoteFetcher,
    ExperimentPromotionRemoteFetchError,
    ExperimentPromotionRemotePolicy,
    ExperimentPromotionRemoteReaderRouter,
    ExperimentPromotionStage,
    ExperimentRankingSpec,
    ExperimentRunner,
    ExperimentTrialState,
    SignedExperimentPromotionArchiveAttestation,
    build_experiment_plan,
    render_experiment_csv,
    render_experiment_html,
    render_experiment_markdown,
)
from agentic_rl_forge.integrations import (
    TrainerBatchExporter,
    export_jsonl,
    iter_search_r1_tasks,
    task_to_verl_record,
)
from agentic_rl_forge.pipelines import (
    collect_search_r1,
    inspect_search_r1_plan,
    load_search_r1_collection_config,
    run_offline_pipeline,
)
from agentic_rl_forge.quality import DoctorProfile, ReleaseAuditor, run_doctor
from agentic_rl_forge.rewards import (
    CostReward,
    ExactMatchOutcome,
    RewardEngine,
    ToolExecutionReward,
)
from agentic_rl_forge.rollout import (
    AgentLoop,
    PolicyOutput,
    ScriptedPolicy,
    SignalAwareRolloutFilter,
)
from agentic_rl_forge.services import (
    BM25Index,
    ReloadableRetriever,
    create_retriever_app,
    load_jsonl_documents,
)
from agentic_rl_forge.storage import (
    BlobConflictError,
    CheckpointRegistry,
    ClaimStoreConsistency,
    ConditionalBlobStore,
    LocalBlobStore,
    RunArtifactArchive,
    RunArtifactArchiveError,
    RunArtifactBundle,
    RunArtifactMirror,
    RunArtifactMirrorError,
    RunArtifactTransport,
    RunArtifactTransportError,
    RunReconciliationConflictError,
    ShardedTrajectoryStore,
    SlotClaimCoordinator,
    SQLiteTrajectoryStore,
    TrajectoryQuery,
    create_s3_blob_store,
    export_trajectories_jsonl,
    load_trajectories_jsonl,
)

app = typer.Typer(no_args_is_help=True, invoke_without_command=True, add_completion=False)
console = Console()


@app.callback()
def main(
    version_flag: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the installed package version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """AgenticRLForge local Studio and agent-RL command line."""
    if version_flag:
        console.print(__version__)
        raise typer.Exit()


def _echo_machine_json(value: object) -> None:
    """Emit locale-independent JSON that remains UTF-8 decodable when redirected."""
    if isinstance(value, bytes | bytearray | memoryview):
        value = orjson.loads(value)
    typer.echo(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _resolve_run_artifact_root(
    manifest_path: Path,
    manifest: RunArtifactManifest,
    root: Path | None,
) -> Path:
    if root is None:
        if manifest_path.parent.parent.name != "runs":
            raise typer.BadParameter(
                "cannot infer collection root; provide --root",
                param_hint="--root",
            )
        root = manifest_path.parents[2]
    expected_path = root / RunArtifactBundle.manifest_relative_path(manifest.run_id)
    if expected_path.resolve() != manifest_path.resolve():
        raise typer.BadParameter("manifest path does not match its run under collection root")
    return root


def _resolve_archive_digest(
    archive: Path,
    *,
    expected_sha256: str | None,
    checksum: Path | None,
    fallback: str | None = None,
) -> str:
    if expected_sha256 is not None and checksum is not None:
        raise typer.BadParameter(
            "use either --expected-sha256 or --checksum, not both",
            param_hint="--expected-sha256",
        )
    if expected_sha256 is not None:
        return RunArtifactArchive.normalize_digest(expected_sha256)
    if checksum is not None:
        return RunArtifactArchive.read_checksum(checksum)
    adjacent = RunArtifactArchive.checksum_path(archive)
    if adjacent.is_file():
        return RunArtifactArchive.read_checksum(adjacent)
    if fallback is not None:
        return RunArtifactArchive.normalize_digest(fallback)
    return RunArtifactArchive.read_checksum(adjacent)


def _load_archive_attestation(path: Path) -> SignedRunArtifactArchiveAttestation:
    try:
        return SignedRunArtifactArchiveAttestation.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid run artifact attestation: {error}") from error


def _resolve_experiment_promotion_archive_digest(
    archive: Path,
    *,
    expected_sha256: str | None,
    checksum: Path | None,
    fallback: str | None = None,
) -> str:
    if expected_sha256 is not None and checksum is not None:
        raise typer.BadParameter(
            "use either --expected-sha256 or --checksum, not both",
            param_hint="--expected-sha256",
        )
    if expected_sha256 is not None:
        return ExperimentPromotionArchive.normalize_digest(expected_sha256)
    if checksum is not None:
        return ExperimentPromotionArchive.read_checksum(checksum)
    adjacent = ExperimentPromotionArchive.checksum_path(archive)
    if adjacent.is_file():
        return ExperimentPromotionArchive.read_checksum(adjacent)
    if fallback is not None:
        return ExperimentPromotionArchive.normalize_digest(fallback)
    return ExperimentPromotionArchive.read_checksum(adjacent)


def _load_experiment_promotion_archive_attestation(
    path: Path,
) -> SignedExperimentPromotionArchiveAttestation:
    try:
        payload = path.read_bytes()
        signed = SignedExperimentPromotionArchiveAttestation.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid experiment promotion attestation: {error}") from error
    if payload != signed.canonical_bytes() + b"\n":
        raise typer.BadParameter("experiment promotion attestation is not canonical")
    return signed


def _load_gc_plan(path: Path) -> RunArtifactGcPlan:
    try:
        payload = path.read_bytes()
        plan = RunArtifactGcPlan.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid run artifact GC plan: {error}") from error
    if payload != plan.canonical_bytes() + b"\n":
        raise typer.BadParameter("run artifact GC plan is not canonical")
    return plan


def _load_mirror_plan(path: Path) -> RunArtifactMirrorPlan:
    try:
        payload = path.read_bytes()
        plan = RunArtifactMirrorPlan.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid run artifact mirror plan: {error}") from error
    if payload != plan.canonical_bytes() + b"\n":
        raise typer.BadParameter("run artifact mirror plan is not canonical")
    return plan


def _load_mirror_batch_plan(path: Path) -> RunArtifactMirrorBatchPlan:
    try:
        payload = path.read_bytes()
        plan = RunArtifactMirrorBatchPlan.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid run artifact mirror batch plan: {error}") from error
    if payload != plan.canonical_bytes() + b"\n":
        raise typer.BadParameter("run artifact mirror batch plan is not canonical")
    return plan


def _load_experiment_plan(path: Path) -> ExperimentPlan:
    try:
        payload = path.read_bytes()
        plan = ExperimentPlan.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid experiment plan: {error}") from error
    if payload != plan.canonical_bytes() + b"\n":
        raise typer.BadParameter("experiment plan is not canonical")
    return plan


def _load_experiment_index(path: Path) -> ExperimentOperationsIndex:
    try:
        payload = path.read_bytes()
        index = ExperimentOperationsIndex.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid experiment index: {error}") from error
    if payload != index.canonical_bytes() + b"\n":
        raise typer.BadParameter("experiment index is not canonical")
    return index


def _load_experiment_analysis(path: Path) -> ExperimentAnalysis:
    try:
        payload = path.read_bytes()
        analysis = ExperimentAnalysis.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid experiment analysis: {error}") from error
    if payload != analysis.canonical_bytes() + b"\n":
        raise typer.BadParameter("experiment analysis is not canonical")
    return analysis


def _parse_experiment_objectives(values: list[str] | None) -> tuple[ExperimentObjective, ...]:
    if not values:
        raise typer.BadParameter(
            "provide at least one METRIC_ID:DIRECTION[:WEIGHT] value",
            param_hint="--objective",
        )
    objectives = []
    for value in values:
        parts = value.rsplit(":", maxsplit=2)
        if len(parts) not in {2, 3}:
            raise typer.BadParameter(
                f"invalid experiment objective {value!r}",
                param_hint="--objective",
            )
        metric_id, direction_value = parts[:2]
        try:
            direction = ExperimentObjectiveDirection(direction_value)
            weight = float(parts[2]) if len(parts) == 3 else 1.0
            objectives.append(
                ExperimentObjective(
                    metric_id=metric_id,
                    direction=direction,
                    weight=weight,
                )
            )
        except ValueError as error:
            raise typer.BadParameter(
                f"invalid experiment objective {value!r}: {error}",
                param_hint="--objective",
            ) from error
    return tuple(sorted(objectives, key=lambda item: item.metric_id))


def _trusted_public_keys(paths: list[Path] | None) -> tuple[str, ...]:
    if not paths:
        raise typer.BadParameter(
            "provide at least one trusted --public-key",
            param_hint="--public-key",
        )
    values = []
    for path in paths:
        try:
            value = path.read_text(encoding="ascii").strip()
            Ed25519ManifestSigner.public_key_id(value)
        except (OSError, UnicodeError, ValueError) as error:
            raise typer.BadParameter(f"invalid Ed25519 public key {path}: {error}") from error
        values.append(value)
    return tuple(values)


def _create_archive_transport_store(
    *,
    store_root: Path | None,
    s3_bucket: str | None,
    s3_prefix: str,
    s3_endpoint_url: str | None,
    s3_region: str | None,
) -> ConditionalBlobStore:
    if (store_root is None) == (s3_bucket is None):
        raise typer.BadParameter(
            "provide exactly one of --store-root or --s3-bucket",
            param_hint="--store-root",
        )
    if store_root is not None:
        if s3_prefix or s3_endpoint_url is not None or s3_region is not None:
            raise typer.BadParameter("S3 options require --s3-bucket")
        return LocalBlobStore(store_root)
    assert s3_bucket is not None
    return create_s3_blob_store(
        bucket=s3_bucket,
        prefix=s3_prefix,
        endpoint_url=s3_endpoint_url,
        region_name=s3_region,
    )


def _create_mirror_stores(
    *,
    source_store_root: Path | None,
    source_s3_bucket: str | None,
    source_s3_prefix: str,
    source_s3_endpoint_url: str | None,
    source_s3_region: str | None,
    destination_store_root: Path | None,
    destination_s3_bucket: str | None,
    destination_s3_prefix: str,
    destination_s3_endpoint_url: str | None,
    destination_s3_region: str | None,
) -> tuple[ConditionalBlobStore, ConditionalBlobStore]:
    source = _create_archive_transport_store(
        store_root=source_store_root,
        s3_bucket=source_s3_bucket,
        s3_prefix=source_s3_prefix,
        s3_endpoint_url=source_s3_endpoint_url,
        s3_region=source_s3_region,
    )
    destination = _create_archive_transport_store(
        store_root=destination_store_root,
        s3_bucket=destination_s3_bucket,
        s3_prefix=destination_s3_prefix,
        s3_endpoint_url=destination_s3_endpoint_url,
        s3_region=destination_s3_region,
    )
    return source, destination


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(__version__)


@app.command()
def doctor(
    project: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, resolve_path=True),
    ] = None,
    profile: Annotated[DoctorProfile, typer.Option(case_sensitive=False)] = DoctorProfile.CORE,
    strict: Annotated[
        bool,
        typer.Option(help="Exit with a non-zero status when a required check fails."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a machine-readable diagnostic report."),
    ] = False,
) -> None:
    """Inspect local prerequisites without changing the environment."""
    project_path = project
    if project_path is None and (Path.cwd() / "pyproject.toml").is_file():
        project_path = Path.cwd()
    report = run_doctor(profile, project_path)
    if json_output:
        _echo_machine_json(report.canonical_bytes())
        if report.exit_code(strict):
            raise typer.Exit(code=1)
        return
    table = Table(title="AgenticRLForge environment")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Purpose")
    table.add_row("package", report.package_version, "core runtime")
    table.add_row("profile", report.profile.value, "diagnostic profile")
    for check in report.checks:
        detail = check.purpose
        if check.remediation and check.status.value != "pass":
            detail = f"{detail} — {check.remediation}"
        table.add_row(check.name, check.status.value, detail)
    console.print(table)
    console.print(f"overall status: {report.status.value}")
    if report.exit_code(strict):
        raise typer.Exit(code=1)


@app.command()
def demo(
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Run a complete local search-agent rollout without a model server."""
    from agentic_rl_forge.integrations import build_search_r1_task

    search = InMemorySearchTool(
        {
            "france": "Paris is the capital and largest city of France.",
            "germany": "Berlin is the capital and largest city of Germany.",
        }
    )
    task = build_search_r1_task(
        task_id="local-demo",
        question="What is the capital of France?",
        answers="Paris",
    ).model_copy(update={"tools": (search.spec,)})
    policy = ScriptedPolicy(
        (
            PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.TOOL,
                    reasoning="I should retrieve a supporting passage.",
                    tool_calls=(ToolCall(name="search", arguments={"query": "capital of France"}),),
                    raw_text="<think>I should retrieve evidence.</think>"
                    "<search>capital of France</search>",
                ),
                generated_token_count=12,
            ),
            PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.FINAL,
                    reasoning="The retrieved passage directly supports the answer.",
                    final_answer="Paris",
                    raw_text="<think>The passage is explicit.</think><answer>Paris</answer>",
                ),
                generated_token_count=10,
            ),
        ),
        version="local-demo-policy",
    )
    loop = AgentLoop(
        policy=policy,
        environment=LocalToolEnvironment((search,)),
        rewards=RewardEngine((ExactMatchOutcome(), ToolExecutionReward(), CostReward())),
    )
    trajectory = asyncio.run(loop.run(task, group_id="local-demo-group", seed=0))
    if output is not None:
        export_trajectories_jsonl((trajectory,), output)
    console.print(
        JSON.from_data(
            {
                "trajectory_id": trajectory.trajectory_id,
                "status": trajectory.status.value,
                "steps": len(trajectory.steps),
                "reward": trajectory.total_reward,
                "generated_tokens": trajectory.total_generated_tokens,
                "observation_tokens": trajectory.total_observation_tokens,
                "response_mask": [
                    token for step in trajectory.steps for token in step.generated_token_mask
                ],
            }
        )
    )


@app.command("offline-pipeline")
def offline_pipeline_command(
    data: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    corpus: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(resolve_path=True)],
    rollouts_per_task: Annotated[int, typer.Option(min=2, max=4)] = 4,
    seed: Annotated[int, typer.Option()] = 0,
    max_concurrency: Annotated[int, typer.Option(min=1, max=256)] = 8,
) -> None:
    """Verify the full offline Agent RL data path with a local QA set and corpus."""
    try:
        result = asyncio.run(
            run_offline_pipeline(
                output,
                data,
                corpus,
                rollouts_per_task=rollouts_per_task,
                seed=seed,
                max_concurrency=max_concurrency,
            )
        )
    except (FileExistsError, OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_machine_json(result.canonical_bytes())


@app.command("collect-search-r1")
def collect_search_r1_command(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, resolve_path=True),
    ],
    api_key_env: Annotated[str | None, typer.Option("--api-key-env")] = "OPENAI_API_KEY",
) -> None:
    """Collect durable Search-R1 rollouts from model and retrieval endpoints."""
    try:
        config = load_search_r1_collection_config(config_path)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--config") from error
    api_key = os.environ.get(api_key_env) if api_key_env else None
    result = asyncio.run(
        collect_search_r1(
            source,
            output_dir,
            config,
            api_key=api_key,
        )
    )
    console.print(JSON.from_data(result.model_dump(mode="json")))


@app.command("search-r1-plan-status")
def search_r1_plan_status(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, resolve_path=True),
    ],
) -> None:
    """List reusable, missing, and conflicting slots without contacting services."""
    try:
        config = load_search_r1_collection_config(config_path)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--config") from error
    status = asyncio.run(inspect_search_r1_plan(source, output_dir, config))
    console.print(JSON.from_data(status.model_dump(mode="json")))
    if status.conflict_count:
        raise typer.Exit(code=1)


@app.command("experiment-plan")
def experiment_plan_command(
    matrix_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Expand a deterministic experiment matrix without running commands."""
    try:
        plan = build_experiment_plan(matrix_path)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                plan.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (BlobConflictError, ExperimentPlanError, OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(plan.model_dump(mode="json")))


@app.command("experiment-run")
def experiment_run_command(
    plan_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
    state_dir: Annotated[
        Path,
        typer.Option("--state-dir", file_okay=False, resolve_path=True),
    ] = Path("artifacts/experiment-state"),
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_plan_id: Annotated[
        str | None,
        typer.Option("--confirm-plan-id"),
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression"),
    ] = False,
    fail_on_failed: Annotated[bool, typer.Option("--fail-on-failed")] = False,
    fail_on_incomplete: Annotated[bool, typer.Option("--fail-on-incomplete")] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Inspect or execute a confirmed local experiment plan with resumable evidence."""
    plan = _load_experiment_plan(plan_path)
    if execute and confirm_plan_id is None:
        raise typer.BadParameter(
            "--execute requires --confirm-plan-id",
            param_hint="--confirm-plan-id",
        )
    if not execute and confirm_plan_id is not None:
        raise typer.BadParameter(
            "--confirm-plan-id requires --execute",
            param_hint="--confirm-plan-id",
        )
    try:
        runner = ExperimentRunner(root, state_dir)
        report = (
            runner.run(
                plan,
                confirm_plan_id=confirm_plan_id or "",
                max_workers=workers,
            )
            if execute
            else runner.report(plan)
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                report.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        ExperimentExecutionError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(report.model_dump(mode="json")))
    failed = report.state_counts[ExperimentTrialState.FAILED.value]
    incomplete = report.state_counts[ExperimentTrialState.INCOMPLETE.value]
    if (
        (fail_on_regression and report.failed_gate_count)
        or (fail_on_failed and failed)
        or (fail_on_incomplete and incomplete)
    ):
        raise typer.Exit(code=1)


@app.command("experiment-index")
def experiment_index_command(
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
    state_dir: Annotated[
        Path,
        typer.Option("--state-dir", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts/experiment-state"),
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
    csv_output: Annotated[
        Path | None,
        typer.Option("--csv-output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_issues: Annotated[bool, typer.Option("--fail-on-issues")] = False,
) -> None:
    """Discover and validate local experiment plans, reports, parameters, and metrics."""
    try:
        index = ExperimentIndexBuilder(root, state_dir).build()
        if output is not None and csv_output is not None and output == csv_output:
            raise ValueError("experiment index outputs must use distinct paths")
        payloads: tuple[tuple[Path | None, bytes], ...] = (
            (output, index.canonical_bytes() + b"\n"),
            (csv_output, render_experiment_csv(index).encode("utf-8")),
        )
        for destination, payload in payloads:
            if destination is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(destination.parent).put_if_absent(destination.name, payload)
            destination.chmod(0o644)
    except (BlobConflictError, OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(index.model_dump(mode="json")))
    if fail_on_issues and index.issues:
        raise typer.Exit(code=1)


@app.command("experiment-analyze")
def experiment_analyze_command(
    index_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    objective: Annotated[
        list[str] | None,
        typer.Option("--objective"),
    ] = None,
    experiment_name: Annotated[
        list[str] | None,
        typer.Option("--experiment-name"),
    ] = None,
    plan_id: Annotated[list[str] | None, typer.Option("--plan-id")] = None,
    include_regressions: Annotated[
        bool,
        typer.Option("--include-regressions"),
    ] = False,
    baseline_plan_id: Annotated[
        str | None,
        typer.Option("--baseline-plan-id"),
    ] = None,
    baseline_trial_id: Annotated[
        str | None,
        typer.Option("--baseline-trial-id"),
    ] = None,
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", min=1, max=50_000),
    ] = 10_000,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
    markdown_output: Annotated[
        Path | None,
        typer.Option("--markdown-output", dir_okay=False, resolve_path=True),
    ] = None,
    html_output: Annotated[
        Path | None,
        typer.Option("--html-output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_issues: Annotated[bool, typer.Option("--fail-on-issues")] = False,
    fail_on_ineligible: Annotated[
        bool,
        typer.Option("--fail-on-ineligible"),
    ] = False,
) -> None:
    """Rank indexed trials, compare a baseline, and export deterministic dashboards."""
    index = _load_experiment_index(index_path)
    objectives = _parse_experiment_objectives(objective)
    eligible_states = (
        (ExperimentTrialState.COMPLETE, ExperimentTrialState.REGRESSION)
        if include_regressions
        else (ExperimentTrialState.COMPLETE,)
    )
    try:
        spec = ExperimentRankingSpec(
            objectives=objectives,
            eligible_states=eligible_states,
            experiment_names=tuple(sorted(set(experiment_name or []))),
            plan_ids=tuple(sorted(set(plan_id or []))),
            baseline_plan_id=baseline_plan_id,
            baseline_trial_id=baseline_trial_id,
            max_candidates=max_candidates,
        )
        analysis = ExperimentAnalyzer().analyze(index, spec)
        destinations = [item for item in (output, markdown_output, html_output) if item is not None]
        if len(destinations) != len(set(destinations)):
            raise ValueError("experiment analysis outputs must use distinct paths")
        payloads: tuple[tuple[Path | None, bytes], ...] = (
            (output, analysis.canonical_bytes() + b"\n"),
            (markdown_output, render_experiment_markdown(index, analysis).encode("utf-8")),
            (html_output, render_experiment_html(index, analysis).encode("utf-8")),
        )
        for destination, payload in payloads:
            if destination is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(destination.parent).put_if_absent(destination.name, payload)
            destination.chmod(0o644)
    except (BlobConflictError, OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(analysis.model_dump(mode="json")))
    if (fail_on_issues and index.issues) or (fail_on_ineligible and analysis.ineligible_count):
        raise typer.Exit(code=1)


@app.command("experiment-promote")
def experiment_promote_command(
    index_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    analysis_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    promotion_name: Annotated[str, typer.Option("--name")],
    plan_id: Annotated[str, typer.Option("--plan-id")],
    trial_id: Annotated[str, typer.Option("--trial-id")],
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
    state_dir: Annotated[
        Path,
        typer.Option("--state-dir", exists=True, file_okay=False, resolve_path=True),
    ] = Path("artifacts/experiment-state"),
    promotion_dir: Annotated[
        Path,
        typer.Option("--promotion-dir", file_okay=False, resolve_path=True),
    ] = Path("artifacts/experiment-promotions"),
    required_artifact_kind: Annotated[
        list[str] | None,
        typer.Option("--required-artifact-kind"),
    ] = None,
    allow_index_issues: Annotated[bool, typer.Option("--allow-index-issues")] = False,
    allow_regression: Annotated[bool, typer.Option("--allow-regression")] = False,
    require_pareto_front: Annotated[
        bool,
        typer.Option("--require-pareto-front"),
    ] = False,
    maximum_rank: Annotated[int | None, typer.Option("--maximum-rank", min=1)] = None,
    allow_multiple_checkpoints: Annotated[
        bool,
        typer.Option("--allow-multiple-checkpoints"),
    ] = False,
    allow_unverified_checkpoint: Annotated[
        bool,
        typer.Option("--allow-unverified-checkpoint"),
    ] = False,
    skip_dataset_lineage: Annotated[
        bool,
        typer.Option("--skip-dataset-lineage"),
    ] = False,
    minimum_approvals: Annotated[
        int,
        typer.Option("--minimum-approvals", min=1, max=16),
    ] = 1,
    operator_may_approve: Annotated[
        bool,
        typer.Option("--operator-may-approve"),
    ] = False,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_preview_id: Annotated[
        str | None,
        typer.Option("--confirm-preview-id"),
    ] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    approver: Annotated[list[str] | None, typer.Option("--approver")] = None,
    preview_output: Annotated[
        Path | None,
        typer.Option("--preview-output", dir_okay=False, resolve_path=True),
    ] = None,
    record_output: Annotated[
        Path | None,
        typer.Option("--record-output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_ineligible: Annotated[
        bool,
        typer.Option("--fail-on-ineligible"),
    ] = False,
) -> None:
    """Preview or confirm one immutable experiment promotion and model card."""
    index = _load_experiment_index(index_path)
    analysis = _load_experiment_analysis(analysis_path)
    if execute:
        missing = [
            name
            for name, value in (
                ("--confirm-preview-id", confirm_preview_id),
                ("--operator", operator),
                ("--reason", reason),
                ("--approver", approver),
            )
            if not value
        ]
        if missing:
            raise typer.BadParameter(
                "--execute requires " + ", ".join(missing),
                param_hint="--execute",
            )
    elif any(value is not None for value in (confirm_preview_id, operator, reason, approver)):
        raise typer.BadParameter(
            "confirmation, operator, reason, and approvers require --execute",
            param_hint="--execute",
        )
    if record_output is not None and not execute:
        raise typer.BadParameter("--record-output requires --execute", param_hint="--record-output")
    if preview_output is not None and preview_output == record_output:
        raise typer.BadParameter(
            "preview and record outputs must use distinct paths",
            param_hint="--record-output",
        )
    try:
        required_kinds = (
            tuple(
                sorted(
                    {ExperimentArtifactKind(value) for value in required_artifact_kind},
                    key=lambda item: item.value,
                )
            )
            if required_artifact_kind
            else ExperimentPromotionPolicy().required_artifact_kinds
        )
        policy = ExperimentPromotionPolicy(
            required_artifact_kinds=required_kinds,
            require_clean_index=not allow_index_issues,
            allow_regression=allow_regression,
            require_pareto_front=require_pareto_front,
            maximum_rank=maximum_rank,
            require_single_checkpoint=not allow_multiple_checkpoints,
            require_fully_verified_checkpoint=not allow_unverified_checkpoint,
            require_dataset_lineage=not skip_dataset_lineage,
            minimum_approvals=minimum_approvals,
            operator_may_approve=operator_may_approve,
        )
        promoter = ExperimentPromoter(root, state_dir, promotion_dir)
        preview = promoter.preview(
            index,
            analysis,
            promotion_name=promotion_name,
            plan_id=plan_id,
            trial_id=trial_id,
            policy=policy,
        )
        if preview_output is not None:
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(preview_output.parent).put_if_absent(
                preview_output.name,
                preview.canonical_bytes() + b"\n",
            )
            preview_output.chmod(0o644)
        if execute:
            record = promoter.promote(
                index,
                analysis,
                preview,
                confirm_preview_id=confirm_preview_id or "",
                operator=operator or "",
                reason=reason or "",
                approvers=tuple(approver or ()),
            )
            if record_output is not None:
                record_output.parent.mkdir(parents=True, exist_ok=True)
                LocalBlobStore(record_output.parent).put_if_absent(
                    record_output.name,
                    record.canonical_bytes() + b"\n",
                )
                record_output.chmod(0o644)
            console.print(JSON.from_data(record.model_dump(mode="json")))
        else:
            console.print(JSON.from_data(preview.model_dump(mode="json")))
    except (
        BlobConflictError,
        ExperimentPromotionError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    if not execute and fail_on_ineligible and not preview.eligible:
        raise typer.Exit(code=1)


@app.command("experiment-promotion-pack")
def experiment_promotion_pack(
    promotion_directory: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, resolve_path=True),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
) -> None:
    """Pack one verified promotion decision into canonical metadata bytes."""
    try:
        receipt = ExperimentPromotionArchive(promotion_directory).pack(output)
    except (BlobConflictError, ExperimentPromotionArchiveError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(
        JSON.from_data(
            {
                **receipt.model_dump(mode="json"),
                "archive_path": str(output),
                "checksum_path": str(ExperimentPromotionArchive.checksum_path(output)),
            }
        )
    )


@app.command("experiment-promotion-inspect")
def experiment_promotion_inspect(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    attestation: Annotated[
        Path | None,
        typer.Option("--attestation", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Inspect canonical promotion metadata and optionally require a trusted signature."""
    if public_key and attestation is None:
        raise typer.BadParameter(
            "--public-key requires --attestation",
            param_hint="--attestation",
        )
    signed = (
        _load_experiment_promotion_archive_attestation(attestation)
        if attestation is not None
        else None
    )
    trusted_keys = _trusted_public_keys(public_key) if signed is not None else ()
    try:
        expected = _resolve_experiment_promotion_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
            fallback=(signed.attestation.receipt.content_digest if signed is not None else None),
        )
        receipt = ExperimentPromotionArchive.inspect(archive, expected_sha256=expected)
    except ExperimentPromotionArchiveError as error:
        raise typer.BadParameter(str(error)) from error
    payload: dict[str, object] = {
        **receipt.model_dump(mode="json"),
        "archive_path": str(archive),
    }
    if signed is not None:
        verification = ExperimentPromotionArchiveAttestor.verify(
            signed,
            receipt,
            trusted_public_keys=trusted_keys,
        )
        payload["attestation_verification"] = verification.model_dump(mode="json")
        console.print(JSON.from_data(payload))
        if not verification.valid:
            raise typer.Exit(code=1)
        return
    console.print(JSON.from_data(payload))


@app.command("experiment-promotion-unpack")
def experiment_promotion_unpack(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    destination: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    attestation: Annotated[
        Path | None,
        typer.Option("--attestation", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Safely extract promotion metadata, optionally gated by publisher trust."""
    if public_key and attestation is None:
        raise typer.BadParameter(
            "--public-key requires --attestation",
            param_hint="--attestation",
        )
    signed = (
        _load_experiment_promotion_archive_attestation(attestation)
        if attestation is not None
        else None
    )
    trusted_keys = _trusted_public_keys(public_key) if signed is not None else ()
    verification = None
    try:
        expected = _resolve_experiment_promotion_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
            fallback=(signed.attestation.receipt.content_digest if signed is not None else None),
        )
        if signed is not None:
            inspected = ExperimentPromotionArchive.inspect(archive, expected_sha256=expected)
            verification = ExperimentPromotionArchiveAttestor.verify(
                signed,
                inspected,
                trusted_public_keys=trusted_keys,
            )
            if not verification.valid:
                console.print(JSON.from_data(verification.model_dump(mode="json")))
                raise typer.Exit(code=1)
        receipt = ExperimentPromotionArchive.unpack(
            archive,
            destination,
            expected_sha256=expected,
        )
    except ExperimentPromotionArchiveError as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        **receipt.model_dump(mode="json"),
        "archive_path": str(archive),
        "destination": str(destination),
    }
    if verification is not None:
        payload["attestation_verification"] = verification.model_dump(mode="json")
    console.print(JSON.from_data(payload))


@app.command("experiment-promotion-sign")
def experiment_promotion_sign(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    private_key: Annotated[
        Path,
        typer.Option("--private-key", exists=True, dir_okay=False, resolve_path=True),
    ],
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Sign the exact receipt of a canonical promotion metadata archive."""
    try:
        expected = _resolve_experiment_promotion_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
        )
        receipt = ExperimentPromotionArchive.inspect(archive, expected_sha256=expected)
        signer = Ed25519ManifestSigner.from_private_key_base64(
            private_key.read_text(encoding="ascii").strip()
        )
        if output.exists():
            signed = _load_experiment_promotion_archive_attestation(output)
            verification = ExperimentPromotionArchiveAttestor.verify(
                signed,
                receipt,
                trusted_public_keys=(signer.public_key_base64,),
            )
            if not verification.valid:
                raise ExperimentPromotionArchiveError(
                    "attestation output already exists with different content"
                )
        else:
            signed = ExperimentPromotionArchiveAttestor(signer).sign(receipt)
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                signed.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        ExperimentPromotionArchiveError,
        OSError,
        UnicodeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(
        JSON.from_data(
            {
                "output": str(output),
                "attestation_id": signed.attestation.attestation_id,
                "archive_id": signed.attestation.receipt.archive_id,
                "promotion_id": signed.attestation.receipt.promotion_id,
                "signer_key_id": signed.attestation.signer_key_id,
                "signed_at": signed.attestation.signed_at.isoformat(),
            }
        )
    )


@app.command("experiment-promotion-signature-verify")
def experiment_promotion_signature_verify(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    attestation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Verify a promotion archive receipt against trusted publisher keys."""
    signed = _load_experiment_promotion_archive_attestation(attestation)
    trusted_keys = _trusted_public_keys(public_key)
    try:
        expected = _resolve_experiment_promotion_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
            fallback=signed.attestation.receipt.content_digest,
        )
        receipt = ExperimentPromotionArchive.inspect(archive, expected_sha256=expected)
    except ExperimentPromotionArchiveError as error:
        raise typer.BadParameter(str(error)) from error
    verification = ExperimentPromotionArchiveAttestor.verify(
        signed,
        receipt,
        trusted_public_keys=trusted_keys,
    )
    console.print(JSON.from_data(verification.model_dump(mode="json")))
    if not verification.valid:
        raise typer.Exit(code=1)


@app.command("experiment-promotion-acquire")
def experiment_promotion_acquire(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    attestation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    destination_root: Annotated[
        Path,
        typer.Argument(file_okay=False, resolve_path=True),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ],
    state_dir: Annotated[
        Path,
        typer.Option("--state-dir", exists=True, file_okay=False, resolve_path=True),
    ],
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    remote_record: Annotated[
        list[Path] | None,
        typer.Option("--remote-record", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    allow_unresolved_remote: Annotated[
        bool,
        typer.Option("--allow-unresolved-remote"),
    ] = False,
    allow_invalid_native_contracts: Annotated[
        bool,
        typer.Option("--allow-invalid-native-contracts"),
    ] = False,
    allow_missing_checkpoint_payloads: Annotated[
        bool,
        typer.Option("--allow-missing-checkpoint-payloads"),
    ] = False,
    allow_incomplete_checkpoint_ancestry: Annotated[
        bool,
        typer.Option("--allow-incomplete-checkpoint-ancestry"),
    ] = False,
    allow_missing_dataset_lineage: Annotated[
        bool,
        typer.Option("--allow-missing-dataset-lineage"),
    ] = False,
    max_materialized_bytes: Annotated[
        int,
        typer.Option("--max-materialized-bytes", min=1),
    ] = 1 << 40,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_plan_id: Annotated[
        str | None,
        typer.Option("--confirm-plan-id"),
    ] = None,
    plan_output: Annotated[
        Path | None,
        typer.Option("--plan-output", dir_okay=False, resolve_path=True),
    ] = None,
    record_output: Annotated[
        Path | None,
        typer.Option("--record-output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_ineligible: Annotated[
        bool,
        typer.Option("--fail-on-ineligible"),
    ] = False,
) -> None:
    """Verify and materialize one signed promotion artifact graph."""
    if execute and confirm_plan_id is None:
        raise typer.BadParameter(
            "--execute requires --confirm-plan-id",
            param_hint="--confirm-plan-id",
        )
    if not execute and confirm_plan_id is not None:
        raise typer.BadParameter(
            "--confirm-plan-id requires --execute",
            param_hint="--execute",
        )
    if record_output is not None and not execute:
        raise typer.BadParameter(
            "--record-output requires --execute",
            param_hint="--record-output",
        )
    if plan_output is not None and plan_output == record_output:
        raise typer.BadParameter(
            "plan and record outputs must use distinct paths",
            param_hint="--record-output",
        )
    signed = _load_experiment_promotion_archive_attestation(attestation)
    trusted_keys = _trusted_public_keys(public_key)
    try:
        policy = ExperimentPromotionAcquisitionPolicy(
            allow_unresolved_remote=allow_unresolved_remote,
            require_native_contracts=not allow_invalid_native_contracts,
            require_checkpoint_payloads=not allow_missing_checkpoint_payloads,
            require_checkpoint_ancestry=not allow_incomplete_checkpoint_ancestry,
            require_dataset_lineage=not allow_missing_dataset_lineage,
            max_materialized_bytes=max_materialized_bytes,
        )
        acquirer = ExperimentPromotionAcquirer(
            root,
            state_dir,
            remote_records=remote_record or (),
        )
        plan = acquirer.preview(
            archive,
            signed,
            destination_root,
            trusted_public_keys=trusted_keys,
            policy=policy,
        )
        if plan_output is not None:
            plan_output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(plan_output.parent).put_if_absent(
                plan_output.name,
                plan.canonical_bytes() + b"\n",
            )
            plan_output.chmod(0o644)
        if execute:
            record = acquirer.execute(
                plan,
                archive,
                signed,
                trusted_public_keys=trusted_keys,
                confirm_plan_id=confirm_plan_id or "",
            )
            if record_output is not None:
                record_output.parent.mkdir(parents=True, exist_ok=True)
                LocalBlobStore(record_output.parent).put_if_absent(
                    record_output.name,
                    record.canonical_bytes() + b"\n",
                )
                record_output.chmod(0o644)
            console.print(JSON.from_data(record.model_dump(mode="json")))
        else:
            console.print(JSON.from_data(plan.model_dump(mode="json")))
    except (
        BlobConflictError,
        ExperimentPromotionAcquisitionError,
        ExperimentPromotionArchiveError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    if not execute and fail_on_ineligible and not plan.eligible:
        raise typer.Exit(code=1)


@app.command("experiment-promotion-fetch-remote")
def experiment_promotion_fetch_remote(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    attestation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    cache_root: Annotated[
        Path,
        typer.Argument(file_okay=False, resolve_path=True),
    ],
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    https_allow_authority: Annotated[
        list[str] | None,
        typer.Option("--https-allow-authority"),
    ] = None,
    s3_allow_bucket: Annotated[
        list[str] | None,
        typer.Option("--s3-allow-bucket"),
    ] = None,
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
    chunk_size_bytes: Annotated[
        int,
        typer.Option("--chunk-size-bytes", min=64 * 1024, max=64 * 1024 * 1024),
    ] = 8 * 1024 * 1024,
    max_artifact_bytes: Annotated[
        int,
        typer.Option("--max-artifact-bytes", min=1),
    ] = 1 << 39,
    max_total_bytes: Annotated[
        int,
        typer.Option("--max-total-bytes", min=1),
    ] = 1 << 40,
    allow_missing_source_validator: Annotated[
        bool,
        typer.Option("--allow-missing-source-validator"),
    ] = False,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_plan_id: Annotated[
        str | None,
        typer.Option("--confirm-plan-id"),
    ] = None,
    plan_output: Annotated[
        Path | None,
        typer.Option("--plan-output", dir_okay=False, resolve_path=True),
    ] = None,
    record_output: Annotated[
        Path | None,
        typer.Option("--record-output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_ineligible: Annotated[
        bool,
        typer.Option("--fail-on-ineligible"),
    ] = False,
) -> None:
    """Plan or execute explicitly authorized remote promotion artifact fetches."""
    if execute and confirm_plan_id is None:
        raise typer.BadParameter(
            "--execute requires --confirm-plan-id",
            param_hint="--confirm-plan-id",
        )
    if not execute and confirm_plan_id is not None:
        raise typer.BadParameter(
            "--confirm-plan-id requires --execute",
            param_hint="--execute",
        )
    if record_output is not None and not execute:
        raise typer.BadParameter(
            "--record-output requires --execute",
            param_hint="--record-output",
        )
    if plan_output is not None and plan_output == record_output:
        raise typer.BadParameter(
            "plan and record outputs must use distinct paths",
            param_hint="--record-output",
        )
    signed = _load_experiment_promotion_archive_attestation(attestation)
    trusted_keys = _trusted_public_keys(public_key)
    try:
        authorities = tuple(
            sorted({item.strip().casefold() for item in https_allow_authority or []})
        )
        buckets = tuple(sorted({item.strip().casefold() for item in s3_allow_bucket or []}))
        policy = ExperimentPromotionRemotePolicy(
            allowed_https_authorities=authorities,
            allowed_s3_buckets=buckets,
            chunk_size_bytes=chunk_size_bytes,
            max_artifact_bytes=max_artifact_bytes,
            max_total_bytes=max_total_bytes,
            require_source_validator=not allow_missing_source_validator,
        )
        with ExperimentPromotionRemoteReaderRouter(
            allowed_https_authorities=authorities,
            allowed_s3_buckets=buckets,
            s3_endpoint_url=s3_endpoint_url,
            s3_region_name=s3_region,
        ) as reader:
            fetcher = ExperimentPromotionRemoteFetcher(reader)
            plan = fetcher.preview(
                archive,
                signed,
                cache_root,
                trusted_public_keys=trusted_keys,
                policy=policy,
            )
            if plan_output is not None:
                plan_output.parent.mkdir(parents=True, exist_ok=True)
                LocalBlobStore(plan_output.parent).put_if_absent(
                    plan_output.name,
                    plan.canonical_bytes() + b"\n",
                )
                plan_output.chmod(0o644)
            if execute:
                record = fetcher.execute(
                    plan,
                    archive,
                    signed,
                    trusted_public_keys=trusted_keys,
                    confirm_plan_id=confirm_plan_id or "",
                )
                if record_output is not None:
                    record_output.parent.mkdir(parents=True, exist_ok=True)
                    LocalBlobStore(record_output.parent).put_if_absent(
                        record_output.name,
                        record.canonical_bytes() + b"\n",
                    )
                    record_output.chmod(0o644)
                console.print(JSON.from_data(record.model_dump(mode="json")))
            else:
                console.print(JSON.from_data(plan.model_dump(mode="json")))
    except (
        BlobConflictError,
        ExperimentPromotionArchiveError,
        ExperimentPromotionRemoteFetchError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    if not execute and fail_on_ineligible and not plan.eligible:
        raise typer.Exit(code=1)


@app.command("experiment-promotion-lifecycle")
def experiment_promotion_lifecycle(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    attestation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    registry_dir: Annotated[
        Path,
        typer.Option("--registry-dir", file_okay=False, resolve_path=True),
    ],
    target_stage: Annotated[ExperimentPromotionStage, typer.Option("--target-stage")],
    operator: Annotated[str, typer.Option("--operator")],
    reason: Annotated[str, typer.Option("--reason")],
    authorizer: Annotated[list[str] | None, typer.Option("--authorizer")] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    minimum_authorizers: Annotated[
        int,
        typer.Option("--minimum-authorizers", min=1, max=16),
    ] = 1,
    operator_may_authorize: Annotated[
        bool,
        typer.Option("--operator-may-authorize"),
    ] = False,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_preview_id: Annotated[
        str | None,
        typer.Option("--confirm-preview-id"),
    ] = None,
    preview_output: Annotated[
        Path | None,
        typer.Option("--preview-output", dir_okay=False, resolve_path=True),
    ] = None,
    event_output: Annotated[
        Path | None,
        typer.Option("--event-output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_ineligible: Annotated[
        bool,
        typer.Option("--fail-on-ineligible"),
    ] = False,
) -> None:
    """Preview or append one authenticated promotion lifecycle transition."""
    if not authorizer:
        raise typer.BadParameter(
            "provide at least one deployment --authorizer",
            param_hint="--authorizer",
        )
    if execute and confirm_preview_id is None:
        raise typer.BadParameter(
            "--execute requires --confirm-preview-id",
            param_hint="--confirm-preview-id",
        )
    if not execute and confirm_preview_id is not None:
        raise typer.BadParameter(
            "--confirm-preview-id requires --execute",
            param_hint="--execute",
        )
    if event_output is not None and not execute:
        raise typer.BadParameter("--event-output requires --execute", param_hint="--event-output")
    if preview_output is not None and preview_output == event_output:
        raise typer.BadParameter(
            "preview and event outputs must use distinct paths",
            param_hint="--event-output",
        )
    signed = _load_experiment_promotion_archive_attestation(attestation)
    trusted_keys = _trusted_public_keys(public_key)
    try:
        policy = ExperimentPromotionGovernancePolicy(
            minimum_authorizers=minimum_authorizers,
            operator_may_authorize=operator_may_authorize,
        )
        registry = ExperimentPromotionRegistry(LocalBlobStore(registry_dir))
        preview = registry.preview_lifecycle(
            archive,
            signed,
            trusted_public_keys=trusted_keys,
            target_stage=target_stage,
            operator=operator,
            reason=reason,
            authorizers=tuple(authorizer),
            policy=policy,
        )
        if preview_output is not None:
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(preview_output.parent).put_if_absent(
                preview_output.name,
                preview.canonical_bytes() + b"\n",
            )
            preview_output.chmod(0o644)
        if execute:
            event = registry.execute_lifecycle(
                preview,
                archive,
                signed,
                trusted_public_keys=trusted_keys,
                confirm_preview_id=confirm_preview_id or "",
            )
            if event_output is not None:
                event_output.parent.mkdir(parents=True, exist_ok=True)
                LocalBlobStore(event_output.parent).put_if_absent(
                    event_output.name,
                    event.canonical_bytes() + b"\n",
                )
                event_output.chmod(0o644)
            console.print(JSON.from_data(event.model_dump(mode="json")))
        else:
            console.print(JSON.from_data(preview.model_dump(mode="json")))
    except (
        BlobConflictError,
        ExperimentPromotionArchiveError,
        ExperimentPromotionRegistryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    if not execute and fail_on_ineligible and not preview.eligible:
        raise typer.Exit(code=1)


@app.command("experiment-promotion-alias")
def experiment_promotion_alias(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    attestation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    registry_dir: Annotated[
        Path,
        typer.Option("--registry-dir", file_okay=False, resolve_path=True),
    ],
    environment: Annotated[str, typer.Option("--environment")],
    action: Annotated[ExperimentPromotionAliasAction, typer.Option("--action")],
    operator: Annotated[str, typer.Option("--operator")],
    reason: Annotated[str, typer.Option("--reason")],
    authorizer: Annotated[list[str] | None, typer.Option("--authorizer")] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    allowed_stage: Annotated[
        list[str] | None,
        typer.Option("--allowed-stage"),
    ] = None,
    minimum_authorizers: Annotated[
        int,
        typer.Option("--minimum-authorizers", min=1, max=16),
    ] = 1,
    operator_may_authorize: Annotated[
        bool,
        typer.Option("--operator-may-authorize"),
    ] = False,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_preview_id: Annotated[
        str | None,
        typer.Option("--confirm-preview-id"),
    ] = None,
    preview_output: Annotated[
        Path | None,
        typer.Option("--preview-output", dir_okay=False, resolve_path=True),
    ] = None,
    event_output: Annotated[
        Path | None,
        typer.Option("--event-output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_ineligible: Annotated[
        bool,
        typer.Option("--fail-on-ineligible"),
    ] = False,
) -> None:
    """Preview or append one CAS-protected environment alias decision."""
    if not authorizer:
        raise typer.BadParameter(
            "provide at least one deployment --authorizer",
            param_hint="--authorizer",
        )
    if execute and confirm_preview_id is None:
        raise typer.BadParameter(
            "--execute requires --confirm-preview-id",
            param_hint="--confirm-preview-id",
        )
    if not execute and confirm_preview_id is not None:
        raise typer.BadParameter(
            "--confirm-preview-id requires --execute",
            param_hint="--execute",
        )
    if event_output is not None and not execute:
        raise typer.BadParameter("--event-output requires --execute", param_hint="--event-output")
    if preview_output is not None and preview_output == event_output:
        raise typer.BadParameter(
            "preview and event outputs must use distinct paths",
            param_hint="--event-output",
        )
    signed = _load_experiment_promotion_archive_attestation(attestation)
    trusted_keys = _trusted_public_keys(public_key)
    try:
        stages = (
            tuple(
                sorted(
                    {ExperimentPromotionStage(value) for value in allowed_stage},
                    key=lambda item: item.value,
                )
            )
            if allowed_stage
            else (ExperimentPromotionStage.PRODUCTION,)
        )
        policy = ExperimentPromotionAliasPolicy(
            allowed_stages=stages,
            governance=ExperimentPromotionGovernancePolicy(
                minimum_authorizers=minimum_authorizers,
                operator_may_authorize=operator_may_authorize,
            ),
        )
        registry = ExperimentPromotionRegistry(LocalBlobStore(registry_dir))
        preview = registry.preview_alias(
            archive,
            signed,
            trusted_public_keys=trusted_keys,
            environment=environment,
            action=action,
            operator=operator,
            reason=reason,
            authorizers=tuple(authorizer),
            policy=policy,
        )
        if preview_output is not None:
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(preview_output.parent).put_if_absent(
                preview_output.name,
                preview.canonical_bytes() + b"\n",
            )
            preview_output.chmod(0o644)
        if execute:
            event = registry.execute_alias(
                preview,
                archive,
                signed,
                trusted_public_keys=trusted_keys,
                confirm_preview_id=confirm_preview_id or "",
            )
            if event_output is not None:
                event_output.parent.mkdir(parents=True, exist_ok=True)
                LocalBlobStore(event_output.parent).put_if_absent(
                    event_output.name,
                    event.canonical_bytes() + b"\n",
                )
                event_output.chmod(0o644)
            console.print(JSON.from_data(event.model_dump(mode="json")))
        else:
            console.print(JSON.from_data(preview.model_dump(mode="json")))
    except (
        BlobConflictError,
        ExperimentPromotionArchiveError,
        ExperimentPromotionRegistryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    if not execute and fail_on_ineligible and not preview.eligible:
        raise typer.Exit(code=1)


@app.command("experiment-promotion-registry-status")
def experiment_promotion_registry_status(
    registry_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, resolve_path=True),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_issues: Annotated[bool, typer.Option("--fail-on-issues")] = False,
) -> None:
    """Inspect deterministic lifecycle, aliases, rollbacks, and registry damage."""
    try:
        status = ExperimentPromotionRegistry(LocalBlobStore(registry_dir)).status()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                status.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (BlobConflictError, OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(status.model_dump(mode="json")))
    if fail_on_issues and not status.valid:
        raise typer.Exit(code=1)


@app.command("run-liveness")
def run_liveness(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    stale_after_s: Annotated[float, typer.Option("--stale-after", min=0.001)] = 300.0,
    only_stale: Annotated[bool, typer.Option("--only-stale")] = False,
    fail_on_stale: Annotated[bool, typer.Option("--fail-on-stale")] = False,
) -> None:
    """Report active, stale, and terminal runs without changing run status."""
    with SQLiteTrajectoryStore(database) as store:
        entries = store.list_run_liveness(stale_after_s=stale_after_s)
    stale_count = sum(item.state is RunLivenessState.STALE for item in entries)
    selected = (
        tuple(item for item in entries if item.state is RunLivenessState.STALE)
        if only_stale
        else entries
    )
    console.print(
        JSON.from_data(
            {
                "run_count": len(entries),
                "stale_count": stale_count,
                "runs": [item.model_dump(mode="json") for item in selected],
            }
        )
    )
    if fail_on_stale and stale_count:
        raise typer.Exit(code=1)


@app.command("run-reconcile")
def run_reconcile(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    run_id: Annotated[str, typer.Argument()],
    stale_after_s: Annotated[float, typer.Option("--stale-after", min=0.001)] = 300.0,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_state_digest: Annotated[
        str | None,
        typer.Option("--confirm-state-digest"),
    ] = None,
    operator_id: Annotated[str | None, typer.Option("--operator")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Preview or explicitly reconcile one stale run with immutable evidence."""
    if output is not None and not execute:
        raise typer.BadParameter("--output requires --execute", param_hint="--output")
    with SQLiteTrajectoryStore(database) as store:
        if not execute:
            preview = store.preview_run_reconciliation(
                run_id,
                stale_after_s=stale_after_s,
            )
            console.print(JSON.from_data(orjson.loads(preview.canonical_bytes())))
            return
        if confirm_state_digest is None:
            raise typer.BadParameter(
                "copy state_digest from a prior preview",
                param_hint="--confirm-state-digest",
            )
        if operator_id is None or not operator_id.strip():
            raise typer.BadParameter("operator identity is required", param_hint="--operator")
        if reason is None or len(reason.strip()) < 8:
            raise typer.BadParameter(
                "reason must contain at least 8 characters",
                param_hint="--reason",
            )
        existing = store.get_run_reconciliation(run_id)
        if existing is not None:
            if (
                existing.preview.state_digest != confirm_state_digest
                or existing.operator_id != operator_id.strip()
                or existing.reason != reason.strip()
            ):
                raise typer.BadParameter("run already has another reconciliation record")
            record = existing
        else:
            preview = store.preview_run_reconciliation(
                run_id,
                stale_after_s=stale_after_s,
            )
            if preview.state_digest != confirm_state_digest:
                raise typer.BadParameter(
                    "run state changed or confirmation digest does not match preview",
                    param_hint="--confirm-state-digest",
                )
            if not preview.eligible:
                raise typer.BadParameter("run is not eligible for stale reconciliation")
            try:
                record = store.reconcile_stale_run(
                    preview,
                    operator_id=operator_id,
                    reason=reason,
                )
            except (RunReconciliationConflictError, ValueError) as error:
                raise typer.BadParameter(str(error)) from error
    if output is not None:
        try:
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                record.canonical_bytes() + b"\n",
            )
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--output") from error
    console.print(JSON.from_data(orjson.loads(record.canonical_bytes())))


@app.command("slot-claim-status")
def slot_claim_status(
    plan_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    blob_root: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, resolve_path=True),
    ],
    claim_prefix: Annotated[str, typer.Option("--claim-prefix")] = "coordination/slot-claims",
    fail_on_active: Annotated[bool, typer.Option("--fail-on-active")] = False,
    fail_on_expired: Annotated[bool, typer.Option("--fail-on-expired")] = False,
) -> None:
    """Inspect current planned-slot claims without modifying coordination state."""
    plan = RolloutPlan.model_validate_json(plan_path.read_bytes())
    try:
        coordinator = SlotClaimCoordinator(
            LocalBlobStore(blob_root),
            consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
            prefix=claim_prefix,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--claim-prefix") from error
    report = coordinator.inspect_plan(plan)
    console.print(JSON.from_data(report.model_dump(mode="json")))
    active_count = report.state_counts[SlotClaimState.ACTIVE]
    expired_count = report.state_counts[SlotClaimState.EXPIRED]
    if (fail_on_active and active_count) or (fail_on_expired and expired_count):
        raise typer.Exit(code=1)


@app.command("trajectory-import")
def trajectory_import(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    database: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    run_name: Annotated[str, typer.Option("--run-name")] = "trajectory-import",
) -> None:
    """Import validated trajectory JSONL into the persistent store."""
    trajectories = load_trajectories_jsonl(source)
    policies = {item.policy_version for item in trajectories}
    environments = {item.environment_version for item in trajectories}
    manifest = RunManifest(
        name=run_name,
        kind=RunKind.ROLLOUT,
        config={"source": str(source), "trajectory_count": len(trajectories)},
        policy_version=next(iter(policies)) if len(policies) == 1 else None,
        environment_version=(next(iter(environments)) if len(environments) == 1 else None),
        package_version=__version__,
    )
    with SQLiteTrajectoryStore(database) as store:
        store.create_run(manifest)
        try:
            inserted = store.put_many(trajectories, run_id=manifest.run_id)
        except Exception:
            store.finish_run(manifest.run_id, status=RunStatus.FAILED)
            raise
        store.finish_run(
            manifest.run_id,
            metadata={"inserted": inserted, "attached": len(trajectories)},
        )
    console.print(
        f"attached {len(trajectories)} trajectories to {manifest.run_id}; "
        f"inserted {inserted} new records"
    )


@app.command("shard-import")
def shard_import(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    root: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    """Import trajectory JSONL into recoverable per-trajectory shards."""
    trajectories = load_trajectories_jsonl(source)
    store = ShardedTrajectoryStore(root, run_id=run_id)
    inserted = sum(store.put(trajectory) for trajectory in trajectories)
    manifest = store.finalize()
    verification = store.verify(manifest)
    console.print(
        JSON.from_data(
            {
                "inserted": inserted,
                "manifest": manifest.model_dump(mode="json"),
                "verification": verification.model_dump(mode="json"),
            }
        )
    )


@app.command("shard-finalize")
def shard_finalize(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    run_id: Annotated[str, typer.Option("--run-id")],
    expected_policy_version: Annotated[str | None, typer.Option()] = None,
    allow_mixed_policy: Annotated[bool, typer.Option()] = False,
) -> None:
    """Recover existing shards and create an immutable run manifest."""
    store = ShardedTrajectoryStore(root, run_id=run_id)
    manifest = store.finalize(
        expected_policy_version=expected_policy_version,
        require_single_policy=not allow_mixed_policy,
    )
    console.print(JSON.from_data(manifest.model_dump(mode="json")))


@app.command("shard-verify")
def shard_verify(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    run_id: Annotated[str, typer.Option("--run-id")],
    manifest_id: Annotated[str, typer.Option("--manifest-id")],
) -> None:
    """Verify every trajectory shard referenced by a manifest."""
    store = ShardedTrajectoryStore(root, run_id=run_id)
    manifest = store.get_manifest(manifest_id)
    if manifest is None:
        raise typer.BadParameter(f"manifest {manifest_id!r} was not found")
    console.print(JSON.from_data(store.verify(manifest).model_dump(mode="json")))


@app.command("shard-export")
def shard_export(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    run_id: Annotated[str, typer.Option("--run-id")],
    manifest_id: Annotated[str, typer.Option("--manifest-id")],
) -> None:
    """Compact a verified shard manifest into deterministic JSONL."""
    store = ShardedTrajectoryStore(root, run_id=run_id)
    manifest = store.get_manifest(manifest_id)
    if manifest is None:
        raise typer.BadParameter(f"manifest {manifest_id!r} was not found")
    count = store.export_jsonl(manifest, output)
    console.print(f"wrote {count} trajectories to {output}")


@app.command("run-artifacts-verify")
def run_artifacts_verify(
    manifest_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    root: Annotated[
        Path | None,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Verify a portable collection artifact manifest and all trajectory shards."""
    manifest = RunArtifactManifest.model_validate_json(manifest_path.read_bytes())
    root = _resolve_run_artifact_root(manifest_path, manifest, root)
    verification = RunArtifactBundle(root).verify(manifest)
    console.print(JSON.from_data(verification.model_dump(mode="json", exclude_none=True)))
    if not verification.valid:
        raise typer.Exit(code=1)


@app.command("run-artifacts-pack")
def run_artifacts_pack(
    manifest_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    root: Annotated[
        Path | None,
        typer.Option("--root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Create a deterministic, self-contained archive for one verified run."""
    manifest = RunArtifactManifest.model_validate_json(manifest_path.read_bytes())
    collection_root = _resolve_run_artifact_root(manifest_path, manifest, root)
    try:
        receipt = RunArtifactArchive(collection_root).pack(manifest, output)
    except RunArtifactArchiveError as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        **receipt.model_dump(mode="json"),
        "archive_path": str(output),
        "checksum_path": str(RunArtifactArchive.checksum_path(output)),
    }
    console.print(JSON.from_data(payload))


@app.command("run-artifacts-unpack")
def run_artifacts_unpack(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    destination: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    attestation: Annotated[
        Path | None,
        typer.Option("--attestation", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Safely extract and recursively verify one run artifact archive."""
    if public_key and attestation is None:
        raise typer.BadParameter(
            "--public-key requires --attestation",
            param_hint="--attestation",
        )
    signed = _load_archive_attestation(attestation) if attestation is not None else None
    trusted_keys = _trusted_public_keys(public_key) if signed is not None else ()
    verification = None
    try:
        expected = _resolve_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
            fallback=(signed.attestation.receipt.content_digest if signed is not None else None),
        )
        if signed is not None:
            inspected = RunArtifactArchive.inspect(archive, expected_sha256=expected)
            verification = RunArtifactArchiveAttestor.verify(
                signed,
                inspected,
                trusted_public_keys=trusted_keys,
            )
            if not verification.valid:
                console.print(JSON.from_data(verification.model_dump(mode="json")))
                raise typer.Exit(code=1)
        receipt = RunArtifactArchive.unpack(
            archive,
            destination,
            expected_sha256=expected,
        )
    except RunArtifactArchiveError as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        **receipt.model_dump(mode="json"),
        "archive_path": str(archive),
        "destination": str(destination),
    }
    if verification is not None:
        payload["attestation_verification"] = verification.model_dump(mode="json")
    console.print(JSON.from_data(payload))


@app.command("run-artifacts-sign")
def run_artifacts_sign(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    private_key: Annotated[
        Path,
        typer.Option("--private-key", exists=True, dir_okay=False, resolve_path=True),
    ],
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Sign the exact receipt of a canonical run artifact archive."""
    try:
        expected = _resolve_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
        )
        receipt = RunArtifactArchive.inspect(archive, expected_sha256=expected)
        signer = Ed25519ManifestSigner.from_private_key_base64(
            private_key.read_text(encoding="ascii").strip()
        )
        if output.exists():
            signed = _load_archive_attestation(output)
            verification = RunArtifactArchiveAttestor.verify(
                signed,
                receipt,
                trusted_public_keys=(signer.public_key_base64,),
            )
            if not verification.valid or output.read_bytes() != signed.canonical_bytes() + b"\n":
                raise RunArtifactArchiveError(
                    "attestation output already exists with different content"
                )
        else:
            signed = RunArtifactArchiveAttestor(signer).sign(receipt)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                signed.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (BlobConflictError, OSError, RunArtifactArchiveError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(
        JSON.from_data(
            {
                "output": str(output),
                "attestation_id": signed.attestation.attestation_id,
                "archive_id": signed.attestation.receipt.archive_id,
                "signer_key_id": signed.attestation.signer_key_id,
                "signed_at": signed.attestation.signed_at.isoformat(),
            }
        )
    )


@app.command("run-artifacts-signature-verify")
def run_artifacts_signature_verify(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    attestation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Verify an archive against a signed receipt and trusted publisher keys."""
    signed = _load_archive_attestation(attestation)
    trusted_keys = _trusted_public_keys(public_key)
    try:
        expected = _resolve_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
            fallback=signed.attestation.receipt.content_digest,
        )
        receipt = RunArtifactArchive.inspect(archive, expected_sha256=expected)
    except RunArtifactArchiveError as error:
        raise typer.BadParameter(str(error)) from error
    verification = RunArtifactArchiveAttestor.verify(
        signed,
        receipt,
        trusted_public_keys=trusted_keys,
    )
    console.print(JSON.from_data(verification.model_dump(mode="json")))
    if not verification.valid:
        raise typer.Exit(code=1)


@app.command("run-artifacts-publish")
def run_artifacts_publish(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
    attestation: Annotated[
        Path | None,
        typer.Option("--attestation", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    expected_sha256: Annotated[str | None, typer.Option("--expected-sha256")] = None,
    checksum: Annotated[
        Path | None,
        typer.Option("--checksum", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    chunk_size_mib: Annotated[
        int,
        typer.Option("--chunk-size-mib", min=1, max=512),
    ] = 8,
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
) -> None:
    """Publish a verified archive through resumable immutable chunks."""
    signed = _load_archive_attestation(attestation) if attestation is not None else None
    try:
        expected = _resolve_archive_digest(
            archive,
            expected_sha256=expected_sha256,
            checksum=checksum,
            fallback=(signed.attestation.receipt.content_digest if signed is not None else None),
        )
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        result = RunArtifactTransport(
            store,
            chunk_size_bytes=chunk_size_mib * 1024 * 1024,
            max_workers=workers,
        ).publish(
            archive,
            expected_sha256=expected,
            attestation=signed,
        )
    except (
        BlobConflictError,
        OSError,
        RunArtifactArchiveError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(result.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-fetch")
def run_artifacts_fetch(
    archive_id: Annotated[str, typer.Argument()],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    require_attestation: Annotated[bool, typer.Option("--require-attestation")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
) -> None:
    """Fetch, resume, verify, and atomically publish a committed run archive."""
    if require_attestation and not public_key:
        raise typer.BadParameter(
            "--require-attestation needs at least one --public-key",
            param_hint="--public-key",
        )
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        result = RunArtifactTransport(store, max_workers=workers).fetch(
            archive_id,
            output,
            trusted_public_keys=trusted_keys,
            require_attestation=require_attestation or bool(trusted_keys),
        )
    except (
        BlobConflictError,
        OSError,
        RunArtifactArchiveError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    payload = {
        **result.model_dump(mode="json", exclude_none=True),
        "archive_path": str(output),
        "checksum_path": str(RunArtifactArchive.checksum_path(output)),
    }
    attestation_path = RunArtifactTransport.fetched_attestation_path(output)
    if attestation_path.is_file():
        payload["attestation_path"] = str(attestation_path)
    console.print(JSON.from_data(payload))


@app.command("run-artifacts-mirror-plan")
def run_artifacts_mirror_plan(
    archive_id: Annotated[str, typer.Argument()],
    source_store_root: Annotated[
        Path | None,
        typer.Option("--source-store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    source_s3_bucket: Annotated[str | None, typer.Option("--source-s3-bucket")] = None,
    source_s3_prefix: Annotated[str, typer.Option("--source-s3-prefix")] = "",
    source_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--source-s3-endpoint-url"),
    ] = None,
    source_s3_region: Annotated[str | None, typer.Option("--source-s3-region")] = None,
    destination_store_root: Annotated[
        Path | None,
        typer.Option("--destination-store-root", file_okay=False, resolve_path=True),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    require_attestation: Annotated[bool, typer.Option("--require-attestation")] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Preview an exact provider-to-provider committed-release mirror."""
    if require_attestation and not public_key:
        raise typer.BadParameter(
            "--require-attestation needs at least one --public-key",
            param_hint="--public-key",
        )
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        source, destination = _create_mirror_stores(
            source_store_root=source_store_root,
            source_s3_bucket=source_s3_bucket,
            source_s3_prefix=source_s3_prefix,
            source_s3_endpoint_url=source_s3_endpoint_url,
            source_s3_region=source_s3_region,
            destination_store_root=destination_store_root,
            destination_s3_bucket=destination_s3_bucket,
            destination_s3_prefix=destination_s3_prefix,
            destination_s3_endpoint_url=destination_s3_endpoint_url,
            destination_s3_region=destination_s3_region,
        )
        plan = RunArtifactMirror(source, destination).plan(
            archive_id,
            trusted_public_keys=trusted_keys,
            require_attestation=require_attestation or bool(trusted_keys),
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                plan.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactMirrorError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(plan.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-mirror")
def run_artifacts_mirror(
    plan_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    source_store_root: Annotated[
        Path | None,
        typer.Option("--source-store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    source_s3_bucket: Annotated[str | None, typer.Option("--source-s3-bucket")] = None,
    source_s3_prefix: Annotated[str, typer.Option("--source-s3-prefix")] = "",
    source_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--source-s3-endpoint-url"),
    ] = None,
    source_s3_region: Annotated[str | None, typer.Option("--source-s3-region")] = None,
    destination_store_root: Annotated[
        Path | None,
        typer.Option("--destination-store-root", file_okay=False, resolve_path=True),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_plan_id: Annotated[str | None, typer.Option("--confirm-plan-id")] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Validate a mirror plan, or execute it after exact plan-ID confirmation."""
    plan = _load_mirror_plan(plan_path)
    if not execute:
        if any(value is not None for value in (confirm_plan_id, operator, reason)):
            raise typer.BadParameter(
                "confirmation, operator, and reason require --execute",
                param_hint="--execute",
            )
        console.print(JSON.from_data(plan.model_dump(mode="json", exclude_none=True)))
        return
    missing = tuple(
        name
        for name, value in (
            ("--confirm-plan-id", confirm_plan_id),
            ("--operator", operator),
            ("--reason", reason),
        )
        if value is None
    )
    if missing:
        raise typer.BadParameter(
            f"--execute requires {', '.join(missing)}",
            param_hint="--execute",
        )
    if plan.attestation_verification is not None and not public_key:
        raise typer.BadParameter(
            "authenticated mirror plan execution requires at least one --public-key",
            param_hint="--public-key",
        )
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        source, destination = _create_mirror_stores(
            source_store_root=source_store_root,
            source_s3_bucket=source_s3_bucket,
            source_s3_prefix=source_s3_prefix,
            source_s3_endpoint_url=source_s3_endpoint_url,
            source_s3_region=source_s3_region,
            destination_store_root=destination_store_root,
            destination_s3_bucket=destination_s3_bucket,
            destination_s3_prefix=destination_s3_prefix,
            destination_s3_endpoint_url=destination_s3_endpoint_url,
            destination_s3_region=destination_s3_region,
        )
        record = RunArtifactMirror(source, destination).execute(
            plan,
            confirm_plan_id=confirm_plan_id or "",
            operator=operator or "",
            reason=reason or "",
            trusted_public_keys=trusted_keys,
            max_workers=workers,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                record.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactMirrorError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(record.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-mirror-batch-plan")
def run_artifacts_mirror_batch_plan(
    archive_id: Annotated[list[str] | None, typer.Option("--archive-id")] = None,
    all_committed: Annotated[bool, typer.Option("--all-committed")] = False,
    source_store_root: Annotated[
        Path | None,
        typer.Option("--source-store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    source_s3_bucket: Annotated[str | None, typer.Option("--source-s3-bucket")] = None,
    source_s3_prefix: Annotated[str, typer.Option("--source-s3-prefix")] = "",
    source_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--source-s3-endpoint-url"),
    ] = None,
    source_s3_region: Annotated[str | None, typer.Option("--source-s3-region")] = None,
    destination_store_root: Annotated[
        Path | None,
        typer.Option("--destination-store-root", file_okay=False, resolve_path=True),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    require_attestation: Annotated[bool, typer.Option("--require-attestation")] = False,
    max_releases: Annotated[
        int,
        typer.Option("--max-releases", min=1, max=100_000),
    ] = RunArtifactMirror.DEFAULT_MAX_BATCH_RELEASES,
    max_copy_bytes: Annotated[
        int,
        typer.Option("--max-copy-bytes", min=0),
    ] = RunArtifactMirror.DEFAULT_MAX_BATCH_COPY_BYTES,
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Preview an exact multi-release store-to-store mirror batch."""
    selected = tuple(archive_id or ())
    if all_committed == bool(selected):
        raise typer.BadParameter(
            "use either one or more --archive-id values or --all-committed",
            param_hint="--archive-id",
        )
    if require_attestation and not public_key:
        raise typer.BadParameter(
            "--require-attestation needs at least one --public-key",
            param_hint="--public-key",
        )
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        source, destination = _create_mirror_stores(
            source_store_root=source_store_root,
            source_s3_bucket=source_s3_bucket,
            source_s3_prefix=source_s3_prefix,
            source_s3_endpoint_url=source_s3_endpoint_url,
            source_s3_region=source_s3_region,
            destination_store_root=destination_store_root,
            destination_s3_bucket=destination_s3_bucket,
            destination_s3_prefix=destination_s3_prefix,
            destination_s3_endpoint_url=destination_s3_endpoint_url,
            destination_s3_region=destination_s3_region,
        )
        plan = RunArtifactMirror(source, destination).plan_batch(
            archive_ids=selected,
            include_all_committed=all_committed,
            trusted_public_keys=trusted_keys,
            require_attestation=require_attestation or bool(trusted_keys),
            max_release_count=max_releases,
            max_copy_bytes=max_copy_bytes,
            max_workers=workers,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                plan.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactMirrorError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(plan.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-mirror-batch")
def run_artifacts_mirror_batch(
    plan_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    source_store_root: Annotated[
        Path | None,
        typer.Option("--source-store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    source_s3_bucket: Annotated[str | None, typer.Option("--source-s3-bucket")] = None,
    source_s3_prefix: Annotated[str, typer.Option("--source-s3-prefix")] = "",
    source_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--source-s3-endpoint-url"),
    ] = None,
    source_s3_region: Annotated[str | None, typer.Option("--source-s3-region")] = None,
    destination_store_root: Annotated[
        Path | None,
        typer.Option("--destination-store-root", file_okay=False, resolve_path=True),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_plan_id: Annotated[str | None, typer.Option("--confirm-plan-id")] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    release_workers: Annotated[
        int,
        typer.Option("--release-workers", min=1, max=64),
    ] = 1,
    object_workers: Annotated[
        int,
        typer.Option("--object-workers", min=1, max=64),
    ] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Validate a mirror batch plan, or execute its exact release set."""
    plan = _load_mirror_batch_plan(plan_path)
    if not execute:
        if any(value is not None for value in (confirm_plan_id, operator, reason)):
            raise typer.BadParameter(
                "confirmation, operator, and reason require --execute",
                param_hint="--execute",
            )
        console.print(JSON.from_data(plan.model_dump(mode="json", exclude_none=True)))
        return
    missing = tuple(
        name
        for name, value in (
            ("--confirm-plan-id", confirm_plan_id),
            ("--operator", operator),
            ("--reason", reason),
        )
        if value is None
    )
    if missing:
        raise typer.BadParameter(
            f"--execute requires {', '.join(missing)}",
            param_hint="--execute",
        )
    authenticated = any(item.attestation_verification is not None for item in plan.releases)
    if authenticated and not public_key:
        raise typer.BadParameter(
            "authenticated mirror batch execution requires at least one --public-key",
            param_hint="--public-key",
        )
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        source, destination = _create_mirror_stores(
            source_store_root=source_store_root,
            source_s3_bucket=source_s3_bucket,
            source_s3_prefix=source_s3_prefix,
            source_s3_endpoint_url=source_s3_endpoint_url,
            source_s3_region=source_s3_region,
            destination_store_root=destination_store_root,
            destination_s3_bucket=destination_s3_bucket,
            destination_s3_prefix=destination_s3_prefix,
            destination_s3_endpoint_url=destination_s3_endpoint_url,
            destination_s3_region=destination_s3_region,
        )
        record = RunArtifactMirror(source, destination).execute_batch(
            plan,
            confirm_plan_id=confirm_plan_id or "",
            operator=operator or "",
            reason=reason or "",
            trusted_public_keys=trusted_keys,
            release_workers=release_workers,
            object_workers=object_workers,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                record.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactMirrorError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(record.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-mirror-batch-status")
def run_artifacts_mirror_batch_status(
    plan_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    operator: Annotated[str, typer.Option("--operator")],
    reason: Annotated[str, typer.Option("--reason")],
    source_store_root: Annotated[
        Path | None,
        typer.Option("--source-store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    source_s3_bucket: Annotated[str | None, typer.Option("--source-s3-bucket")] = None,
    source_s3_prefix: Annotated[str, typer.Option("--source-s3-prefix")] = "",
    source_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--source-s3-endpoint-url"),
    ] = None,
    source_s3_region: Annotated[str | None, typer.Option("--source-s3-region")] = None,
    destination_store_root: Annotated[
        Path | None,
        typer.Option(
            "--destination-store-root",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_invalid: Annotated[bool, typer.Option("--fail-on-invalid")] = False,
    fail_on_incomplete: Annotated[bool, typer.Option("--fail-on-incomplete")] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Inspect exact persisted and in-flight progress for a mirror batch."""
    plan = _load_mirror_batch_plan(plan_path)
    authenticated = any(item.attestation_verification is not None for item in plan.releases)
    if authenticated and not public_key:
        raise typer.BadParameter(
            "authenticated mirror batch status requires at least one --public-key",
            param_hint="--public-key",
        )
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        source, destination = _create_mirror_stores(
            source_store_root=source_store_root,
            source_s3_bucket=source_s3_bucket,
            source_s3_prefix=source_s3_prefix,
            source_s3_endpoint_url=source_s3_endpoint_url,
            source_s3_region=source_s3_region,
            destination_store_root=destination_store_root,
            destination_s3_bucket=destination_s3_bucket,
            destination_s3_prefix=destination_s3_prefix,
            destination_s3_endpoint_url=destination_s3_endpoint_url,
            destination_s3_region=destination_s3_region,
        )
        status = RunArtifactMirror(source, destination).batch_status(
            plan,
            operator=operator,
            reason=reason,
            trusted_public_keys=trusted_keys,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                status.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactMirrorError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(status.model_dump(mode="json", exclude_none=True)))
    invalid = status.state_counts[RunArtifactMirrorBatchMemberState.INVALID.value]
    incomplete = (
        status.release_count
        - status.state_counts[RunArtifactMirrorBatchMemberState.COMPLETED.value]
    )
    if (fail_on_invalid and invalid) or (fail_on_incomplete and incomplete):
        raise typer.Exit(code=1)


@app.command("run-artifacts-mirror-batch-list")
def run_artifacts_mirror_batch_list(
    destination_store_root: Annotated[
        Path | None,
        typer.Option(
            "--destination-store-root",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    fail_on_unclassified: Annotated[
        bool,
        typer.Option("--fail-on-unclassified"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """List canonical mirror batch evidence from the destination store."""
    try:
        destination = _create_archive_transport_store(
            store_root=destination_store_root,
            s3_bucket=destination_s3_bucket,
            s3_prefix=destination_s3_prefix,
            s3_endpoint_url=destination_s3_endpoint_url,
            s3_region=destination_s3_region,
        )
        ledger = RunArtifactMirror(destination, destination).operations_ledger()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                ledger.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (OSError, RunArtifactMirrorError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(ledger.model_dump(mode="json", exclude_none=True)))
    if fail_on_unclassified and ledger.unclassified_keys:
        raise typer.Exit(code=1)


@app.command("run-artifacts-mirror-batch-inspect")
def run_artifacts_mirror_batch_inspect(
    batch_id: Annotated[str, typer.Argument()],
    source_store_root: Annotated[
        Path | None,
        typer.Option("--source-store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    source_s3_bucket: Annotated[str | None, typer.Option("--source-s3-bucket")] = None,
    source_s3_prefix: Annotated[str, typer.Option("--source-s3-prefix")] = "",
    source_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--source-s3-endpoint-url"),
    ] = None,
    source_s3_region: Annotated[str | None, typer.Option("--source-s3-region")] = None,
    destination_store_root: Annotated[
        Path | None,
        typer.Option(
            "--destination-store-root",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    fail_on_unhealthy: Annotated[bool, typer.Option("--fail-on-unhealthy")] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Inspect one mirror batch by its destination-side batch ID."""
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        source, destination = _create_mirror_stores(
            source_store_root=source_store_root,
            source_s3_bucket=source_s3_bucket,
            source_s3_prefix=source_s3_prefix,
            source_s3_endpoint_url=source_s3_endpoint_url,
            source_s3_region=source_s3_region,
            destination_store_root=destination_store_root,
            destination_s3_bucket=destination_s3_bucket,
            destination_s3_prefix=destination_s3_prefix,
            destination_s3_endpoint_url=destination_s3_endpoint_url,
            destination_s3_region=destination_s3_region,
        )
        inspection = RunArtifactMirror(source, destination).inspect_batch(
            batch_id,
            trusted_public_keys=trusted_keys,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                inspection.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactMirrorError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(inspection.model_dump(mode="json", exclude_none=True)))
    if fail_on_unhealthy and inspection.health not in {
        RunArtifactMirrorBatchHealth.COMPLETE,
        RunArtifactMirrorBatchHealth.RESOLVED,
    }:
        raise typer.Exit(code=1)


@app.command("run-artifacts-mirror-batch-resolve")
def run_artifacts_mirror_batch_resolve(
    batch_id: Annotated[str, typer.Argument()],
    source_store_root: Annotated[
        Path | None,
        typer.Option("--source-store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    source_s3_bucket: Annotated[str | None, typer.Option("--source-s3-bucket")] = None,
    source_s3_prefix: Annotated[str, typer.Option("--source-s3-prefix")] = "",
    source_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--source-s3-endpoint-url"),
    ] = None,
    source_s3_region: Annotated[str | None, typer.Option("--source-s3-region")] = None,
    destination_store_root: Annotated[
        Path | None,
        typer.Option(
            "--destination-store-root",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    destination_s3_bucket: Annotated[
        str | None,
        typer.Option("--destination-s3-bucket"),
    ] = None,
    destination_s3_prefix: Annotated[str, typer.Option("--destination-s3-prefix")] = "",
    destination_s3_endpoint_url: Annotated[
        str | None,
        typer.Option("--destination-s3-endpoint-url"),
    ] = None,
    destination_s3_region: Annotated[
        str | None,
        typer.Option("--destination-s3-region"),
    ] = None,
    public_key: Annotated[
        list[Path] | None,
        typer.Option("--public-key", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    kind: Annotated[
        RunArtifactMirrorBatchResolutionKind | None,
        typer.Option("--kind"),
    ] = None,
    confirm_status_digest: Annotated[
        str | None,
        typer.Option("--confirm-status-digest"),
    ] = None,
    resolver: Annotated[str | None, typer.Option("--resolver")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    replacement_plan_id: Annotated[
        str | None,
        typer.Option("--replacement-plan-id"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Preview or immutably resolve a non-started or blocked mirror batch."""
    trusted_keys = _trusted_public_keys(public_key) if public_key else ()
    try:
        source, destination = _create_mirror_stores(
            source_store_root=source_store_root,
            source_s3_bucket=source_s3_bucket,
            source_s3_prefix=source_s3_prefix,
            source_s3_endpoint_url=source_s3_endpoint_url,
            source_s3_region=source_s3_region,
            destination_store_root=destination_store_root,
            destination_s3_bucket=destination_s3_bucket,
            destination_s3_prefix=destination_s3_prefix,
            destination_s3_endpoint_url=destination_s3_endpoint_url,
            destination_s3_region=destination_s3_region,
        )
        mirror = RunArtifactMirror(source, destination)
        result: RunArtifactMirrorBatchInspection | RunArtifactMirrorBatchResolution
        if not execute:
            if any(
                value is not None
                for value in (
                    kind,
                    confirm_status_digest,
                    resolver,
                    reason,
                    replacement_plan_id,
                )
            ):
                raise typer.BadParameter(
                    "resolution fields require --execute",
                    param_hint="--execute",
                )
            result = mirror.inspect_batch(batch_id, trusted_public_keys=trusted_keys)
        else:
            missing = tuple(
                name
                for name, value in (
                    ("--kind", kind),
                    ("--confirm-status-digest", confirm_status_digest),
                    ("--resolver", resolver),
                    ("--reason", reason),
                )
                if value is None
            )
            if missing:
                raise typer.BadParameter(
                    f"--execute requires {', '.join(missing)}",
                    param_hint="--execute",
                )
            result = mirror.resolve_batch(
                batch_id,
                kind=kind or RunArtifactMirrorBatchResolutionKind.CANCELLED,
                confirm_status_digest=confirm_status_digest or "",
                resolver=resolver or "",
                reason=reason or "",
                replacement_plan_id=replacement_plan_id,
                trusted_public_keys=trusted_keys,
            )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                result.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except typer.BadParameter:
        raise
    except (
        BlobConflictError,
        OSError,
        RunArtifactMirrorError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(result.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-list")
def run_artifacts_list(
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
) -> None:
    """List only fully committed run archive releases."""
    try:
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        archive_ids = RunArtifactTransport(store).list_committed_archive_ids()
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data({"count": len(archive_ids), "archive_ids": archive_ids}))


@app.command("run-artifacts-inventory")
def run_artifacts_inventory(
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Inventory every release prefix and summarize storage by lifecycle state."""
    try:
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        inventory = RunArtifactTransport(store).inventory()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                inventory.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (BlobConflictError, OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(inventory.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-status")
def run_artifacts_status(
    archive_id: Annotated[str, typer.Argument()],
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
) -> None:
    """Inspect an absent, staged, committed, or garbage-collected release."""
    try:
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        status = RunArtifactTransport(store).status(archive_id)
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(status.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-gc")
def run_artifacts_gc(
    archive_id: Annotated[str, typer.Argument()],
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
    min_age_seconds: Annotated[
        float,
        typer.Option("--min-age-seconds", min=1),
    ] = 86_400,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_state_digest: Annotated[
        str | None,
        typer.Option("--confirm-state-digest"),
    ] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Preview aged staged-release GC, or execute with an exact digest confirmation."""
    if not execute and any(value is not None for value in (confirm_state_digest, operator, reason)):
        raise typer.BadParameter(
            "confirmation, operator, and reason require --execute",
            param_hint="--execute",
        )
    if execute:
        missing = tuple(
            name
            for name, value in (
                ("--confirm-state-digest", confirm_state_digest),
                ("--operator", operator),
                ("--reason", reason),
            )
            if value is None
        )
        if missing:
            raise typer.BadParameter(
                f"--execute requires {', '.join(missing)}",
                param_hint="--execute",
            )
    try:
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        transport = RunArtifactTransport(store)
        result = (
            transport.garbage_collect(
                archive_id,
                min_age_seconds=min_age_seconds,
                confirm_state_digest=confirm_state_digest or "",
                operator=operator or "",
                reason=reason or "",
            )
            if execute
            else transport.preview_garbage_collection(
                archive_id,
                min_age_seconds=min_age_seconds,
            )
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                result.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(result.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-gc-plan")
def run_artifacts_gc_plan(
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
    min_age_seconds: Annotated[
        float,
        typer.Option("--min-age-seconds", min=1),
    ] = 86_400,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Build a deterministic, read-only retention plan for every eligible staged release."""
    try:
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        plan = RunArtifactTransport(store).plan_garbage_collection(
            min_age_seconds=min_age_seconds,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                plan.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(plan.model_dump(mode="json", exclude_none=True)))


@app.command("run-artifacts-gc-batch")
def run_artifacts_gc_batch(
    plan_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    store_root: Annotated[
        Path | None,
        typer.Option("--store-root", exists=True, file_okay=False, resolve_path=True),
    ] = None,
    s3_bucket: Annotated[str | None, typer.Option("--s3-bucket")] = None,
    s3_prefix: Annotated[str, typer.Option("--s3-prefix")] = "",
    s3_endpoint_url: Annotated[str | None, typer.Option("--s3-endpoint-url")] = None,
    s3_region: Annotated[str | None, typer.Option("--s3-region")] = None,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm_plan_id: Annotated[str | None, typer.Option("--confirm-plan-id")] = None,
    operator: Annotated[str | None, typer.Option("--operator")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Validate a retention plan, or execute it with exact plan-ID confirmation."""
    plan = _load_gc_plan(plan_path)
    if not execute:
        if any(value is not None for value in (confirm_plan_id, operator, reason)):
            raise typer.BadParameter(
                "confirmation, operator, and reason require --execute",
                param_hint="--execute",
            )
        console.print(JSON.from_data(plan.model_dump(mode="json", exclude_none=True)))
        return
    missing = tuple(
        name
        for name, value in (
            ("--confirm-plan-id", confirm_plan_id),
            ("--operator", operator),
            ("--reason", reason),
        )
        if value is None
    )
    if missing:
        raise typer.BadParameter(
            f"--execute requires {', '.join(missing)}",
            param_hint="--execute",
        )
    try:
        store = _create_archive_transport_store(
            store_root=store_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=s3_endpoint_url,
            s3_region=s3_region,
        )
        record = RunArtifactTransport(store).execute_garbage_collection_plan(
            plan,
            confirm_plan_id=confirm_plan_id or "",
            operator=operator or "",
            reason=reason or "",
            max_workers=workers,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            LocalBlobStore(output.parent).put_if_absent(
                output.name,
                record.canonical_bytes() + b"\n",
            )
            output.chmod(0o644)
    except (
        BlobConflictError,
        OSError,
        RunArtifactTransportError,
        RuntimeError,
        ValueError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(JSON.from_data(record.model_dump(mode="json", exclude_none=True)))


@app.command("trainer-batch-export")
def trainer_batch_export(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    root: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    policy_version: Annotated[str, typer.Option("--policy-version")],
    group_size: Annotated[int, typer.Option("--group-size", min=2)],
    source_run_id: Annotated[str | None, typer.Option()] = None,
    prefix: Annotated[str, typer.Option()] = "trainer",
) -> None:
    """Export validated on-policy GRPO groups for trainer consumption."""
    trajectories = load_trajectories_jsonl(source)
    exporter = TrainerBatchExporter(LocalBlobStore(root), prefix=prefix)
    manifest = exporter.export(
        trajectories,
        expected_policy_version=policy_version,
        expected_group_size=group_size,
        source_run_id=source_run_id,
    )
    manifest_key = "/".join(
        part
        for part in (
            prefix.strip("/"),
            "batches",
            manifest.batch_id,
            "manifest.json",
        )
        if part
    )
    console.print(
        JSON.from_data(
            {
                "manifest_key": manifest_key,
                "manifest": manifest.model_dump(mode="json"),
                "verification": exporter.verify(manifest).model_dump(mode="json"),
            }
        )
    )


@app.command("trainer-batch-verify")
def trainer_batch_verify(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    manifest_key: Annotated[str, typer.Option("--manifest-key")],
    prefix: Annotated[str, typer.Option()] = "trainer",
) -> None:
    """Verify a trainer batch payload and its exact trajectory membership."""
    exporter = TrainerBatchExporter(LocalBlobStore(root), prefix=prefix)
    manifest = exporter.load_manifest(manifest_key)
    console.print(JSON.from_data(exporter.verify(manifest).model_dump(mode="json")))


@app.command("dataset-manifest")
def dataset_manifest(
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    name: Annotated[str, typer.Option("--name")],
    split_file: Annotated[list[str] | None, typer.Option("--split-file")] = None,
    id_field: Annotated[str, typer.Option("--id-field")] = "task_id",
    root: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Fingerprint JSONL splits and reject train/validation/test overlap."""
    split_paths: dict[str, list[Path]] = {}
    for value in split_file or []:
        split, separator, raw_path = value.partition("=")
        if not separator or not split or not raw_path:
            raise typer.BadParameter("split files must use SPLIT=PATH")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise typer.BadParameter(f"dataset path is not a file: {path}")
        split_paths.setdefault(split, []).append(path)
    if not split_paths:
        raise typer.BadParameter("provide at least one --split-file SPLIT=PATH")
    manifest = DatasetManifestBuilder().build_collection(
        split_paths,
        name=name,
        id_field=id_field,
        root=root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(manifest.canonical_bytes() + b"\n")
    console.print(
        JSON.from_data(
            {
                "output": str(output),
                "collection_id": manifest.collection_id,
                "splits": {
                    split: {
                        "records": item.record_count,
                        "unique_ids": item.unique_id_count,
                        "content_digest": item.content_digest,
                    }
                    for split, item in manifest.splits.items()
                },
                "total_unique_ids": manifest.total_unique_id_count,
            }
        )
    )


@app.command("manifest-keygen")
def manifest_keygen(
    private_key: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    public_key: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
) -> None:
    """Generate an Ed25519 key pair for manifest or artifact signing."""
    signer = Ed25519ManifestSigner.generate()
    private_key.parent.mkdir(parents=True, exist_ok=True)
    public_key.parent.mkdir(parents=True, exist_ok=True)
    private_key.write_text(signer.private_key_base64 + "\n", encoding="ascii")
    private_key.chmod(0o600)
    public_key.write_text(signer.public_key_base64 + "\n", encoding="ascii")
    public_key.chmod(0o644)
    console.print(f"wrote private key to {private_key} and public key to {public_key}")


@app.command("manifest-sign")
def manifest_sign(
    manifest_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    private_key: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, resolve_path=True),
    ],
) -> None:
    """Sign a dataset collection manifest with an Ed25519 private key."""
    manifest = DatasetCollectionManifest.model_validate_json(manifest_path.read_bytes())
    signer = Ed25519ManifestSigner.from_private_key_base64(
        private_key.read_text(encoding="ascii").strip()
    )
    signed = signer.sign(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(signed.canonical_bytes() + b"\n")
    console.print(f"wrote signed manifest to {output}")


@app.command("manifest-verify")
def manifest_verify(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
) -> None:
    """Verify a signed dataset manifest and payload digest."""
    signed = SignedDatasetCollectionManifest.model_validate_json(path.read_bytes())
    valid = Ed25519ManifestSigner.verify(signed)
    console.print(JSON.from_data({"valid": valid, "collection_id": signed.manifest.collection_id}))
    if not valid:
        raise typer.Exit(code=1)


@app.command("release-audit")
def release_audit(
    project: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, resolve_path=True),
    ] = None,
    strict_git: Annotated[
        bool,
        typer.Option("--strict-git/--allow-dirty"),
    ] = True,
    run_checks: Annotated[bool, typer.Option()] = False,
    build_wheel: Annotated[bool, typer.Option()] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Audit release evidence and fail when a publish gate is unmet."""
    report = ReleaseAuditor(project or Path.cwd()).run(
        strict_git=strict_git,
        run_checks=run_checks,
        build_wheel=build_wheel,
    )
    table = Table(title="AgenticRLForge release audit")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Message")
    for finding in report.findings:
        table.add_row(finding.severity.value, finding.code, finding.message)
    console.print(table)
    console.print(
        f"passes={report.pass_count} warnings={report.warning_count} "
        f"errors={report.error_count} ready={str(report.ready).lower()}"
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(report.canonical_bytes() + b"\n")
    if not report.ready:
        raise typer.Exit(code=1)


@app.command("trajectory-export")
def trajectory_export(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    run_id: Annotated[str | None, typer.Option()] = None,
    policy_version: Annotated[str | None, typer.Option()] = None,
    status: Annotated[str | None, typer.Option()] = None,
    origin: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Export a filtered, deterministic trajectory JSONL file."""
    statuses = (_parse_status(status),) if status is not None else ()
    origins = (_parse_origin(origin),) if origin is not None else ()
    with SQLiteTrajectoryStore(database) as store:
        trajectories = store.query(
            TrajectoryQuery(
                run_id=run_id,
                policy_version=policy_version,
                statuses=statuses,
                origins=origins,
            )
        )
    count = export_trajectories_jsonl(trajectories, output)
    console.print(f"wrote {count} trajectories to {output}")


@app.command("trajectory-summary")
def trajectory_summary(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
) -> None:
    """Show persistent-store counts and version coverage."""
    with SQLiteTrajectoryStore(database) as store:
        summary = store.summary()
    console.print(
        JSON.from_data(
            {
                "trajectory_count": summary.trajectory_count,
                "run_count": summary.run_count,
                "task_count": summary.task_count,
                "policy_versions": summary.policy_versions,
                "environment_versions": summary.environment_versions,
            }
        )
    )


@app.command("inspect-trajectory")
def inspect_trajectory(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    trajectory_id: Annotated[str, typer.Argument()],
) -> None:
    """Pretty-print one stored trajectory."""
    with SQLiteTrajectoryStore(database) as store:
        trajectory = store.get(trajectory_id)
    if trajectory is None:
        raise typer.BadParameter(f"trajectory {trajectory_id!r} was not found")
    console.print(JSON.from_data(trajectory.model_dump(mode="json", exclude_none=True)))


@app.command("evaluate")
def evaluate(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    benchmark: Annotated[str, typer.Option("--benchmark")] = "custom",
    run_id: Annotated[str | None, typer.Option()] = None,
    policy_version: Annotated[str | None, typer.Option()] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Aggregate benchmark results with explicit metric denominators."""
    with SQLiteTrajectoryStore(database) as store:
        trajectories = store.query(TrajectoryQuery(run_id=run_id, policy_version=policy_version))
    report = BenchmarkAggregator().aggregate(
        trajectories,
        benchmark=benchmark,
        run_id=run_id,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(report.canonical_bytes() + b"\n")
        console.print(f"wrote benchmark report to {output}")
    else:
        console.print(JSON.from_data(report.model_dump(mode="json", exclude_none=True)))


@app.command("build-prm-dataset")
def build_prm_dataset(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    run_id: Annotated[str | None, typer.Option()] = None,
    gamma: Annotated[float, typer.Option(min=0.0, max=1.0)] = 1.0,
    min_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    skip_zero_variance_groups: Annotated[bool, typer.Option()] = False,
) -> None:
    """Build task-isolated, lineage-preserving process-reward examples."""
    with SQLiteTrajectoryStore(database) as store:
        trajectories = store.query(TrajectoryQuery(run_id=run_id))
    builder = PRMDatasetBuilder(
        gamma=gamma,
        min_confidence=min_confidence,
        skip_zero_variance_groups=skip_zero_variance_groups,
    )
    examples = builder.build(trajectories)
    count = export_prm_jsonl(examples, output)
    summary = builder.summarize(examples)
    console.print(
        JSON.from_data(
            {
                "output": str(output),
                "example_count": count,
                "trajectory_count": summary.trajectory_count,
                "task_count": summary.task_count,
                "split_counts": summary.split_counts,
                "mean_target": summary.mean_target,
            }
        )
    )


@app.command("filter-rollouts")
def filter_rollouts(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    run_id: Annotated[str | None, typer.Option()] = None,
    policy_version: Annotated[str | None, typer.Option()] = None,
    expected_group_size: Annotated[int | None, typer.Option(min=2)] = None,
    min_reward_stddev: Annotated[float, typer.Option(min=0.0)] = 1e-6,
    min_unique_ratio: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    keep_top_fraction: Annotated[float, typer.Option(min=0.000001, max=1.0)] = 1.0,
    report: Annotated[
        Path | None,
        typer.Option("--report", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Filter low-signal rollout groups before a GRPO update."""
    with SQLiteTrajectoryStore(database) as store:
        trajectories = store.query(TrajectoryQuery(run_id=run_id, policy_version=policy_version))
    result = SignalAwareRolloutFilter(
        expected_group_size=expected_group_size,
        min_reward_stddev=min_reward_stddev,
        min_unique_trajectory_ratio=min_unique_ratio,
        keep_top_fraction=keep_top_fraction,
    ).filter(trajectories, expected_policy_version=policy_version)
    export_trajectories_jsonl(result.accepted, output)
    report_data = {
        "policy_version": result.policy_version,
        "accepted_trajectory_ids": [item.trajectory_id for item in result.accepted],
        "accepted_group_count": result.accepted_group_count,
        "rejected_groups": [item.model_dump(mode="json") for item in result.rejected_groups],
        "signals": [item.model_dump(mode="json") for item in result.signals],
    }
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(orjson.dumps(report_data, option=orjson.OPT_SORT_KEYS) + b"\n")
    console.print(JSON.from_data(report_data))


@app.command("compare")
def compare(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    baseline_policy: Annotated[str, typer.Option("--baseline-policy")],
    candidate_policy: Annotated[str, typer.Option("--candidate-policy")],
    benchmark: Annotated[str, typer.Option("--benchmark")] = "custom",
    bootstrap_samples: Annotated[int, typer.Option(min=100)] = 2000,
    confidence_level: Annotated[float, typer.Option(min=0.01, max=0.99)] = 0.95,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Compare two policies on matched tasks with paired bootstrap intervals."""
    with SQLiteTrajectoryStore(database) as store:
        baseline = store.query(TrajectoryQuery(policy_version=baseline_policy))
        candidate = store.query(TrajectoryQuery(policy_version=candidate_policy))
    report = BenchmarkComparator(
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    ).compare(
        baseline,
        candidate,
        benchmark=benchmark,
        baseline_name=baseline_policy,
        candidate_name=candidate_policy,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(report.canonical_bytes() + b"\n")
        console.print(f"wrote comparison report to {output}")
    else:
        console.print(JSON.from_data(report.model_dump(mode="json", exclude_none=True)))


@app.command("checkpoint-register")
def checkpoint_register(
    registry: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    run_id: Annotated[str, typer.Option("--run-id")],
    step: Annotated[int, typer.Option(min=0)],
    policy_version: Annotated[str, typer.Option("--policy-version")],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, resolve_path=True)],
    artifact: Annotated[list[str] | None, typer.Option("--artifact")] = None,
    parent_checkpoint_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Register an immutable checkpoint manifest and verify local artifacts."""
    artifacts = [CheckpointArtifact.from_file("config", config)]
    for value in artifact or []:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise typer.BadParameter("artifacts must use NAME=PATH")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise typer.BadParameter(f"artifact path is not a file: {path}")
        artifacts.append(CheckpointArtifact.from_file(name, path))
    manifest = CheckpointManifest(
        run_id=run_id,
        step=step,
        policy_version=policy_version,
        config_digest=_sha256_file(config),
        artifacts=tuple(artifacts),
        parent_checkpoint_id=parent_checkpoint_id,
        framework_versions={"agentic_rl_forge": __version__},
    )
    checkpoint_registry = CheckpointRegistry(registry)
    checkpoint_registry.save(manifest)
    verification = checkpoint_registry.verify(manifest)
    console.print(
        JSON.from_data(
            {
                "manifest": manifest.model_dump(mode="json", exclude_none=True),
                "verification": verification.model_dump(mode="json"),
            }
        )
    )


@app.command("checkpoint-list")
def checkpoint_list(
    registry: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, resolve_path=True),
    ],
    run_id: Annotated[str | None, typer.Option()] = None,
    policy_version: Annotated[str | None, typer.Option()] = None,
) -> None:
    """List checkpoint manifests in deterministic step order."""
    manifests = CheckpointRegistry(registry).list(
        run_id=run_id,
        policy_version=policy_version,
    )
    console.print(
        JSON.from_data(
            [manifest.model_dump(mode="json", exclude_none=True) for manifest in manifests]
        )
    )


@app.command("checkpoint-verify")
def checkpoint_verify(
    registry: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, resolve_path=True),
    ],
    checkpoint_id: Annotated[str, typer.Argument()],
) -> None:
    """Recompute all local checkpoint artifact hashes."""
    checkpoint_registry = CheckpointRegistry(registry)
    manifest = checkpoint_registry.get(checkpoint_id)
    if manifest is None:
        raise typer.BadParameter(f"checkpoint {checkpoint_id!r} was not found")
    verification = checkpoint_registry.verify(manifest)
    console.print(JSON.from_data(verification.model_dump(mode="json")))


@app.command("prepare-search-r1")
def prepare_search_r1(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    dataset_name: Annotated[str, typer.Option("--dataset")] = "custom",
    question_field: Annotated[str, typer.Option()] = "question",
    answer_field: Annotated[str, typer.Option()] = "answer",
) -> None:
    """Convert a JSONL QA dataset into a verl-compatible Search-R1 dataset."""
    records = []
    with source.open("rb") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise typer.BadParameter(f"line {line_number} is not a JSON object")
            records.append(payload)
    prepared = [
        task_to_verl_record(task)
        for task in iter_search_r1_tasks(
            records,
            dataset_name=dataset_name,
            question_field=question_field,
            answer_field=answer_field,
        )
    ]
    if output.suffix.casefold() == ".parquet":
        try:
            from datasets import Dataset
        except ImportError as error:
            raise typer.BadParameter(
                "install agentic-rl-forge[data] to write Parquet datasets"
            ) from error
        output.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list([record.model_dump(mode="json") for record in prepared]).to_parquet(
            str(output)
        )
        count = len(prepared)
    else:
        count = export_jsonl(prepared, output)
    console.print(f"wrote {count} records to {output}")


@app.command("corpus-build")
def corpus_build(
    source: Annotated[Path, typer.Argument(exists=True, resolve_path=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False, resolve_path=True)],
    chunk_size: Annotated[int, typer.Option(min=100, max=100_000)] = 1200,
    chunk_overlap: Annotated[int, typer.Option(min=0, max=99_999)] = 120,
    extension: Annotated[
        list[str] | None,
        typer.Option("--extension", help="Repeat to include custom text file extensions."),
    ] = None,
    max_file_mb: Annotated[int, typer.Option(min=1, max=1024)] = 10,
    force: Annotated[
        bool,
        typer.Option(help="Atomically replace an existing output file."),
    ] = False,
) -> None:
    """Build a deterministic searchable JSONL corpus from text and Markdown files."""
    extensions = tuple(extension) if extension else (".txt", ".md", ".markdown", ".rst")
    try:
        report = build_text_corpus(
            source,
            output,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extensions=extensions,
            force=force,
            max_file_bytes=max_file_mb * 1024 * 1024,
        )
    except (FileExistsError, OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_machine_json(report.canonical_bytes())


@app.command("corpus-check")
def corpus_check(
    corpus: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full machine-readable report."),
    ] = False,
) -> None:
    """Validate a corpus and report its stable identity and size statistics."""
    try:
        report = inspect_corpus(corpus)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if json_output:
        _echo_machine_json(report.canonical_bytes())
        return
    _print_corpus_inspection(report)


def _print_corpus_inspection(report: CorpusInspection) -> None:
    table = Table(title="Corpus inspection")
    table.add_column("Field")
    table.add_column("Value")
    for field, value in (
        ("path", report.path),
        ("documents", report.document_count),
        ("sources", report.source_count),
        ("characters", report.total_characters),
        ("average characters", f"{report.average_characters:.1f}"),
        ("file sha256", report.file_sha256),
        ("corpus sha256", report.corpus_sha256),
    ):
        table.add_row(field, str(value))
    console.print(table)


@app.command("search")
def search_corpus(
    query: Annotated[str, typer.Argument(help="Natural-language query to search for.")],
    corpus: Annotated[
        Path,
        typer.Option("--corpus", exists=True, dir_okay=False, resolve_path=True),
    ],
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 3,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Use the Search-R1-compatible JSON response shape."),
    ] = False,
) -> None:
    """Search a local corpus directly without Docker or an HTTP service."""
    if not query.strip():
        raise typer.BadParameter("query cannot be empty")
    try:
        documents = load_jsonl_documents(corpus)
        results = BM25Index(documents).search(query, top_k=top_k)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if json_output:
        _echo_machine_json({"result": [results]})
        return
    table = Table(title=f"Search results for: {query.strip()}")
    table.add_column("Score", justify="right")
    table.add_column("Document")
    table.add_column("Source")
    table.add_column("Contents", overflow="fold")
    for item in results:
        document = item["document"]
        if not isinstance(document, dict):
            continue
        table.add_row(
            str(item.get("score", "")),
            str(document.get("id", "")),
            str(document.get("source", "")),
            str(document.get("contents", "")),
        )
    console.print(table)
    if not results:
        console.print("No matching documents were found.")


@app.command("serve-retriever")
def serve_retriever(
    corpus: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    reload_interval: Annotated[
        float,
        typer.Option(
            min=0,
            max=3600,
            help="Seconds between safe corpus reload checks; use 0 to disable.",
        ),
    ] = 0.0,
    max_concurrent_searches: Annotated[
        int,
        typer.Option(min=1, max=128, help="Maximum retrieval batches processed concurrently."),
    ] = 8,
) -> None:
    """Serve a Search-R1-compatible BM25 retrieval endpoint."""
    try:
        import uvicorn
    except ImportError as error:
        raise typer.BadParameter("install agentic-rl-forge[server]") from error
    try:
        retriever = ReloadableRetriever(corpus, reload_interval=reload_interval)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    app_instance = create_retriever_app(
        retriever,
        max_concurrent_searches=max_concurrent_searches,
    )
    console.print(
        f"loaded {retriever.stats().document_count} documents; "
        f"reload interval: {reload_interval:g}s"
    )
    uvicorn.run(app_instance, host=host, port=port)


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().casefold()
    if normalized == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _studio_browser_url(host: str, port: int) -> str:
    normalized = host.strip()
    if normalized == "0.0.0.0":
        normalized = "127.0.0.1"
    elif normalized in {"::", "[::]"}:
        normalized = "::1"
    if ":" in normalized and not normalized.startswith("["):
        normalized = f"[{normalized}]"
    return f"http://{normalized}:{port}/"


def _open_studio_browser_when_ready(url: str) -> None:
    for _ in range(60):
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status < 500:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)


def _start_studio_browser_opener(url: str) -> None:
    threading.Thread(
        target=_open_studio_browser_when_ready,
        args=(url,),
        name="arf-studio-browser",
        daemon=True,
    ).start()


@app.command()
def studio(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            file_okay=False,
            resolve_path=True,
            help="Directory used for documents, indexes, settings, and the local database.",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help="Address to listen on. Non-loopback addresses require --allow-network."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 7860,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open Studio in the default browser after startup."),
    ] = True,
    allow_network: Annotated[
        bool,
        typer.Option(
            help=(
                "Explicitly allow a non-loopback listener; add separate authentication "
                "before sharing."
            )
        ),
    ] = False,
) -> None:
    """Open the local Agent RL workbench and supporting knowledge tools."""
    normalized_host = host.strip()
    if not normalized_host:
        raise typer.BadParameter("host cannot be empty", param_hint="--host")
    loopback = _is_loopback_host(normalized_host)
    if not loopback and not allow_network:
        raise typer.BadParameter(
            "non-loopback hosts require explicit --allow-network; do not expose Studio directly "
            "to the public internet",
            param_hint="--host",
        )
    try:
        import uvicorn

        from agentic_rl_forge.studio import StudioPaths, create_studio_app
    except ImportError as error:
        raise typer.BadParameter(
            'install the local app with: pip install "agentic-rl-forge[studio]"'
        ) from error

    storage_root = data_dir if data_dir is not None else StudioPaths.default().root
    try:
        app_instance = create_studio_app(
            data_dir,
            allow_network=allow_network or not loopback,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(f"could not initialize Studio: {error}") from error

    url = _studio_browser_url(normalized_host, port)
    console.print(f"Studio: {url}")
    console.print(f"Data: {storage_root}")
    console.print("Press Ctrl+C to stop Studio.")
    if open_browser:
        _start_studio_browser_opener(url)
    uvicorn.run(app_instance, host=normalized_host, port=port)


@app.command("inspect-task")
def inspect_task(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
) -> None:
    """Pretty-print one prepared task record."""
    with path.open(encoding="utf-8") as source:
        payload = json.loads(source.readline())
    console.print(JSON.from_data(payload))


def _parse_status(value: str) -> TrajectoryStatus:
    try:
        return TrajectoryStatus(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in TrajectoryStatus)
        raise typer.BadParameter(f"status must be one of: {choices}") from error


def _parse_origin(value: str) -> DataOrigin:
    try:
        return DataOrigin(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in DataOrigin)
        raise typer.BadParameter(f"origin must be one of: {choices}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    app()

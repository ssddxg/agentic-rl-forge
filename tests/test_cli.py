import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agentic_rl_forge.cli as cli_module
from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    Message,
    MessageRole,
    RunKind,
    RunManifest,
    RunStatus,
    TaskSpec,
    VerifierSpec,
)
from agentic_rl_forge.pipelines import SearchR1CollectionResult
from agentic_rl_forge.rollout import RolloutPlanBuilder
from agentic_rl_forge.storage import (
    ClaimStoreConsistency,
    LocalBlobStore,
    SlotClaimCoordinator,
    SQLiteTrajectoryStore,
)

runner = CliRunner()


def test_version_doctor_and_demo_commands() -> None:
    version = runner.invoke(app, ["version"])
    doctor = runner.invoke(app, ["doctor"])
    demo = runner.invoke(app, ["demo"])

    assert version.exit_code == 0
    assert "0.3.0" in version.stdout
    assert doctor.exit_code == 0
    assert "AgenticRLForge environment" in doctor.stdout
    assert demo.exit_code == 0
    assert '"status": "succeeded"' in demo.stdout


def test_doctor_supports_strict_machine_readable_reports() -> None:
    project_root = Path(__file__).parents[1]
    result = runner.invoke(
        app,
        ["doctor", "--profile", "core", "--project", str(project_root), "--strict", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"] == "core"
    assert payload["status"] == "pass"
    assert payload["project_path"] == str(project_root.resolve())


def test_prepare_and_inspect_search_r1_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "qa.jsonl"
    output = tmp_path / "prepared.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "1",
                "question": "What is the capital of France?",
                "answer": "Paris",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prepare = runner.invoke(
        app,
        [
            "prepare-search-r1",
            str(source),
            str(output),
            "--dataset",
            "tiny",
        ],
    )
    inspect = runner.invoke(app, ["inspect-task", str(output)])

    assert prepare.exit_code == 0
    assert "wrote 1 records" in prepare.stdout
    assert inspect.exit_code == 0
    assert "tiny-" in inspect.stdout


def test_corpus_build_check_and_local_search_commands(tmp_path: Path) -> None:
    source = tmp_path / "我的资料"
    source.mkdir()
    (source / "法国.md").write_text(
        "巴黎是法国的首都, 也是法国人口最多的城市。",
        encoding="utf-8",
    )
    (source / "德国.txt").write_text(
        "Berlin is the capital of Germany.",
        encoding="utf-8",
    )
    corpus = tmp_path / "generated" / "知识库.jsonl"

    built = runner.invoke(app, ["corpus-build", str(source), str(corpus)])
    checked = runner.invoke(app, ["corpus-check", str(corpus), "--json"])
    searched = runner.invoke(
        app,
        ["search", "法国首都", "--corpus", str(corpus), "--top-k", "2", "--json"],
    )

    assert built.exit_code == 0, built.stdout
    assert checked.exit_code == 0, checked.stdout
    assert searched.exit_code == 0, searched.stdout
    build_report = json.loads(built.stdout)
    check_report = json.loads(checked.stdout)
    results = json.loads(searched.stdout)["result"][0]
    assert build_report["document_count"] == 2
    assert check_report["corpus_sha256"] == build_report["corpus_sha256"]
    assert results[0]["document"]["source"] == "法国.md"
    assert "巴黎" in results[0]["document"]["contents"]


def test_search_json_is_locale_independent_when_stdout_is_redirected(tmp_path: Path) -> None:
    corpus = tmp_path / "知识库.jsonl"
    corpus.write_text(
        '{"id":"法国","contents":"巴黎是法国的首都。","source":"资料.md"}\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "cp936"
    environment["PYTHONUTF8"] = "0"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_rl_forge.cli",
            "search",
            "法国首都",
            "--corpus",
            str(corpus),
            "--json",
        ],
        capture_output=True,
        check=False,
        cwd=Path(__file__).parents[1],
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert payload["result"][0][0]["document"]["contents"] == "巴黎是法国的首都。"


def test_corpus_commands_report_friendly_validation_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        '{"id":"duplicate","contents":"one"}\n{"id":"duplicate","contents":"two"}\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["search", "query", "--corpus", str(invalid)])
    output = result.stdout + result.stderr

    assert result.exit_code != 0
    assert "duplicates document id" in output
    assert "Traceback" not in output


def test_offline_pipeline_command_uses_custom_inputs(tmp_path: Path) -> None:
    data = tmp_path / "qa.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "generated" / "offline-run"
    data.write_text(
        '{"id":"one","question":"Which city is the answer?","answer":"Paris"}\n',
        encoding="utf-8",
    )
    corpus.write_text(
        '{"id":"evidence","contents":"The answer is Paris."}\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "offline-pipeline",
            str(data),
            str(corpus),
            str(output),
            "--rollouts-per-task",
            "2",
            "--max-concurrency",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["trajectory_count"] == 2
    assert payload["accepted_trajectory_count"] == 2
    assert payload["trainer_batch_valid"] is True
    assert Path(payload["artifacts"]["database"]).is_file()


def test_collect_search_r1_command_reads_api_key_without_printing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "qa.jsonl"
    source.write_text(
        '{"id":"1","question":"Question?","answer":"Answer"}\n',
        encoding="utf-8",
    )
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            (
                "name: cli-test",
                "dataset_name: qa",
                "model: model",
                "policy_version: policy-v1",
                "model_base_url: http://model.local",
                "retrieval_endpoint: http://retriever.local/retrieve",
                "rollouts_per_task: 2",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    async def fake_collect(
        source_path: Path,
        output_dir: Path,
        collection_config: object,
        *,
        api_key: str | None = None,
    ) -> SearchR1CollectionResult:
        captured.update(
            source=source_path,
            output_dir=output_dir,
            config=collection_config,
            api_key=api_key,
        )
        return SearchR1CollectionResult(
            run_id="run_cli",
            plan_id="plan_" + "0" * 24,
            policy_version="policy-v1",
            source_sha256="0" * 64,
            task_count=1,
            trajectory_count=2,
            collected_trajectory_count=2,
            reused_trajectory_count=0,
            success_count=1,
            status_counts={"succeeded": 1, "failed": 1},
            shard_manifest_id="manifest_cli",
            artifacts={},
        )

    monkeypatch.setattr(cli_module, "collect_search_r1", fake_collect)
    monkeypatch.setenv("MODEL_API_KEY", "secret-model-token")

    result = runner.invoke(
        app,
        [
            "collect-search-r1",
            str(source),
            str(tmp_path / "output"),
            "--config",
            str(config),
            "--api-key-env",
            "MODEL_API_KEY",
        ],
    )

    assert result.exit_code == 0
    assert "run_cli" in result.stdout
    assert "secret-model-token" not in result.stdout
    assert captured["api_key"] == "secret-model-token"

    status = runner.invoke(
        app,
        [
            "search-r1-plan-status",
            str(source),
            str(tmp_path / "new-output"),
            "--config",
            str(config),
        ],
    )
    assert status.exit_code == 0
    assert '"missing_slot_count": 2' in status.stdout
    assert not (tmp_path / "new-output").exists()


def test_run_liveness_cli_reports_stale_runs_without_mutating_them(tmp_path: Path) -> None:
    database = tmp_path / "liveness.db"
    run = RunManifest(
        name="abandoned-run",
        kind=RunKind.ROLLOUT,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    with SQLiteTrajectoryStore(database) as store:
        store.create_run(run)

    result = runner.invoke(
        app,
        [
            "run-liveness",
            str(database),
            "--stale-after",
            "60",
            "--only-stale",
            "--fail-on-stale",
        ],
    )

    assert result.exit_code == 1
    assert '"stale_count": 1' in result.stdout
    assert "heartbeat_missing" in result.stdout
    with SQLiteTrajectoryStore(database) as store:
        unchanged = store.get_run(run.run_id)
    assert unchanged is not None
    assert unchanged.status is RunStatus.RUNNING


def test_run_reconcile_cli_requires_preview_confirmation_and_writes_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "reconcile.db"
    now = datetime.now(timezone.utc)
    run = RunManifest(
        name="stale-cli-run",
        kind=RunKind.ROLLOUT,
        started_at=now - timedelta(minutes=10),
    )
    with SQLiteTrajectoryStore(database) as store:
        store.create_run(run)
        store.acquire_run_lease(
            run.run_id,
            owner_id="lost-cli-worker",
            ttl_s=1,
            now=now - timedelta(minutes=10),
        )

    preview_result = runner.invoke(
        app,
        ["run-reconcile", str(database), run.run_id, "--stale-after", "60"],
    )
    assert preview_result.exit_code == 0
    preview = json.loads(preview_result.stdout)
    assert preview["eligible"] is True
    assert preview["liveness"]["detail"] == "lease_expired"

    missing_confirmation = runner.invoke(
        app,
        [
            "run-reconcile",
            str(database),
            run.run_id,
            "--execute",
            "--operator",
            "cli-operator",
            "--reason",
            "worker machine was removed",
        ],
    )
    assert missing_confirmation.exit_code == 2
    with SQLiteTrajectoryStore(database) as store:
        assert store.get_run(run.run_id).status is RunStatus.RUNNING

    evidence_path = tmp_path / "evidence" / "reconciliation.json"
    execute_args = [
        "run-reconcile",
        str(database),
        run.run_id,
        "--stale-after",
        "60",
        "--execute",
        "--confirm-state-digest",
        preview["state_digest"],
        "--operator",
        "cli-operator",
        "--reason",
        "worker machine was removed",
        "--output",
        str(evidence_path),
    ]
    executed = runner.invoke(app, execute_args)
    retried = runner.invoke(app, execute_args)

    assert executed.exit_code == 0
    assert retried.exit_code == 0
    executed_record = json.loads(executed.stdout)
    retried_record = json.loads(retried.stdout)
    assert executed_record == retried_record
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == executed_record
    with SQLiteTrajectoryStore(database) as store:
        finished = store.get_run(run.run_id)
        heartbeat = store.get_run_heartbeat(run.run_id)
        record = store.get_run_reconciliation(run.run_id)
    assert finished is not None
    assert finished.status is RunStatus.FAILED
    assert heartbeat is not None
    assert heartbeat.epoch == 2
    assert record is not None
    assert record.reconciliation_id == executed_record["reconciliation_id"]


def test_slot_claim_status_cli_is_read_only_and_supports_monitoring_exit_codes(
    tmp_path: Path,
) -> None:
    task = TaskSpec(
        task_id="claim-status-task",
        messages=(Message(role=MessageRole.USER, content="Return yes."),),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "yes"}),
    )
    plan = RolloutPlanBuilder().build(
        (task,),
        policy_version="claim-status-policy-v1",
        source_sha256="0" * 64,
        config_digest="1" * 64,
        rollouts_per_task=2,
        seed=11,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(plan.canonical_bytes() + b"\n")
    coordinator = SlotClaimCoordinator(
        LocalBlobStore(tmp_path),
        consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
        prefix="coordination/slot-claims",
    )
    now = datetime.now(timezone.utc)
    active_claim = coordinator.acquire(
        plan.plan_id,
        plan.slots[0],
        owner_id="active-worker",
        ttl_s=3600,
        now=now - timedelta(seconds=10),
    )
    coordinator.renew(
        active_claim,
        ttl_s=3600,
        now=now - timedelta(seconds=5),
    )
    coordinator.acquire(
        plan.plan_id,
        plan.slots[1],
        owner_id="expired-worker",
        ttl_s=1,
        now=now - timedelta(minutes=1),
    )
    files_before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))

    result = runner.invoke(
        app,
        [
            "slot-claim-status",
            str(plan_path),
            str(tmp_path),
            "--fail-on-active",
            "--fail-on-expired",
        ],
    )

    assert result.exit_code == 1
    assert '"active": 1' in result.stdout
    assert '"expired": 1' in result.stdout
    assert "active-worker" in result.stdout
    assert '"renewal_index": 1' in result.stdout
    files_after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert files_after == files_before

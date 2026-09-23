from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    ArtifactLocation,
    CheckpointArtifact,
    CheckpointManifest,
)
from agentic_rl_forge.data import Ed25519ManifestSigner
from agentic_rl_forge.experiments import (
    ExperimentAnalyzer,
    ExperimentIndexBuilder,
    ExperimentObjectiveDirection,
    ExperimentPromoter,
    ExperimentPromotionAcquirer,
    ExperimentPromotionAcquisitionError,
    ExperimentPromotionAcquisitionPlan,
    ExperimentPromotionAcquisitionPolicy,
    ExperimentPromotionAcquisitionRecord,
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestor,
    ExperimentPromotionArtifactAvailability,
    ExperimentPromotionNativeStatus,
    ExperimentPromotionPolicy,
    ExperimentRankingSpec,
    ExperimentRunner,
    SignedExperimentPromotionArchiveAttestation,
    build_experiment_plan,
)
from test_experiment_operations import objective, write_base_config, write_matrix


def read_filesystem_bytes(path: Path) -> bytes:
    if os.name == "nt":
        path = Path(f"\\\\?\\{os.path.abspath(path)}")
    return path.read_bytes()


@dataclass(frozen=True)
class AcquisitionFixture:
    archive: Path
    attestation: SignedExperimentPromotionArchiveAttestation
    public_key: str
    project_root: Path
    state_root: Path
    checkpoint_payload: Path | None


def build_acquisition_fixture(
    root: Path,
    *,
    remote_checkpoint: bool = False,
    remote_payload: bytes = b"remote-checkpoint",
    parent_checkpoint_id: str | None = None,
    broken_dataset_lineage: bool = False,
) -> AcquisitionFixture:
    project = root / "project"
    project.mkdir(parents=True)
    write_base_config(project)
    dataset = project / "dataset.jsonl"
    dataset.write_text('{"task_id":"task-1"}\n', encoding="utf-8")
    dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    checkpoint_payload = None if remote_checkpoint else project / "checkpoint.bin"
    if checkpoint_payload is not None:
        checkpoint_payload.write_bytes(b"verified-acquisition-checkpoint")
        artifact = CheckpointArtifact.from_file("model", checkpoint_payload)
    else:
        artifact = CheckpointArtifact(
            name="model",
            uri="https://unreachable.invalid/releases/model.bin",
            location=ArtifactLocation.REMOTE,
            sha256=hashlib.sha256(remote_payload).hexdigest(),
            size_bytes=len(remote_payload),
        )
    checkpoint = CheckpointManifest(
        checkpoint_id="checkpoint-acquisition-fixture",
        run_id="run-acquisition-fixture",
        step=10,
        policy_version="fixture-policy",
        config_digest="1" * 64,
        artifacts=(artifact,),
        created_at=datetime(2026, 9, 19, tzinfo=timezone.utc),
        parent_checkpoint_id=parent_checkpoint_id,
        dataset_manifest_digest=("2" * 64 if broken_dataset_lineage else dataset_digest),
    )
    (project / "checkpoint.json").write_bytes(checkpoint.canonical_bytes() + b"\n")
    plan = build_experiment_plan(write_matrix(project))
    state = project / "state"
    ExperimentRunner(project, state).run(
        plan,
        confirm_plan_id=plan.plan_id,
        max_workers=3,
    )
    index = ExperimentIndexBuilder(project, state).build()
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    analysis = ExperimentAnalyzer().analyze(
        index,
        ExperimentRankingSpec(objectives=(score,)),
    )
    selected = analysis.trials[0]
    promotion_root = root / "promotions"
    promoter = ExperimentPromoter(project, state, promotion_root)
    preview = promoter.preview(
        index,
        analysis,
        promotion_name="acquisition-candidate",
        plan_id=selected.plan_id,
        trial_id=selected.trial_id,
        policy=ExperimentPromotionPolicy(
            require_fully_verified_checkpoint=not remote_checkpoint,
            require_dataset_lineage=not broken_dataset_lineage,
        ),
    )
    assert preview.eligible
    promoter.promote(
        index,
        analysis,
        preview,
        confirm_preview_id=preview.preview_id,
        operator="release-operator",
        reason="approve receiver-side acquisition evidence",
        approvers=("release-reviewer",),
    )
    archive = root / "acquisition-candidate.promotion.tar.gz"
    receipt = ExperimentPromotionArchive(promotion_root / "acquisition-candidate").pack(archive)
    signer = Ed25519ManifestSigner.generate()
    attestation = ExperimentPromotionArchiveAttestor(signer).sign(receipt)
    return AcquisitionFixture(
        archive=archive,
        attestation=attestation,
        public_key=signer.public_key_base64,
        project_root=project,
        state_root=state,
        checkpoint_payload=checkpoint_payload,
    )


def test_acquisition_previews_materializes_and_retries_exactly(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(tmp_path / "source")
    destination = tmp_path / "received"
    acquirer = ExperimentPromotionAcquirer(
        fixture.project_root,
        fixture.state_root,
    )

    plan = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    repeated = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )

    assert plan == repeated
    assert plan.canonical_bytes() == repeated.canonical_bytes()
    assert plan.eligible
    assert plan.failed_check_count == 0
    assert plan.verified_artifact_count == plan.manifest.artifact_count
    assert plan.unresolved_remote_count == 0
    assert plan.materialized_file_count == len(plan.materialized_files)
    assert plan.materialized_size_bytes == sum(item.size_bytes for item in plan.materialized_files)
    assert not destination.exists()
    assert all(
        item.availability is ExperimentPromotionArtifactAvailability.VERIFIED
        for item in plan.artifacts
    )
    assert all(
        item.native_status
        in {
            ExperimentPromotionNativeStatus.VERIFIED,
            ExperimentPromotionNativeStatus.NOT_APPLICABLE,
        }
        for item in plan.artifacts
    )

    partial = plan.materialized_files[0]
    partial_source_root = (
        fixture.project_root if partial.scope.value == "project" else fixture.state_root
    )
    partial_target = (
        destination
        / "acquisitions"
        / ExperimentPromotionAcquisitionRecord.expected_acquisition_id(plan.plan_id)
        / partial.output_path
    )
    partial_target.parent.mkdir(parents=True)
    partial_target.write_bytes((partial_source_root / partial.locator).read_bytes())

    record = acquirer.execute(
        plan,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=plan.plan_id,
    )
    retried = acquirer.execute(
        plan,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=plan.plan_id,
    )

    assert retried == record
    prefix = destination / "acquisitions" / record.acquisition_id
    record_path = prefix / "record.json"
    assert record_path.read_bytes() == record.canonical_bytes() + b"\n"
    assert (
        ExperimentPromotionAcquisitionRecord.model_validate_json(record_path.read_bytes()) == record
    )
    for item in plan.materialized_files:
        source_root = fixture.project_root if item.scope.value == "project" else fixture.state_root
        assert (
            read_filesystem_bytes(prefix / item.output_path)
            == (source_root / item.locator).read_bytes()
        )


def test_acquisition_detects_source_and_committed_output_changes(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(tmp_path / "source")
    assert fixture.checkpoint_payload is not None
    destination = tmp_path / "received"
    acquirer = ExperimentPromotionAcquirer(fixture.project_root, fixture.state_root)
    plan = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    original = fixture.checkpoint_payload.read_bytes()
    fixture.checkpoint_payload.write_bytes(b"changed-after-preview")

    changed = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    assert not changed.eligible
    assert any(
        item.reference.locator == "checkpoint.bin"
        and item.availability is ExperimentPromotionArtifactAvailability.MISMATCHED
        for item in changed.artifacts
    )
    with pytest.raises(ExperimentPromotionAcquisitionError, match="changed after preview"):
        acquirer.execute(
            plan,
            fixture.archive,
            fixture.attestation,
            trusted_public_keys=(fixture.public_key,),
            confirm_plan_id=plan.plan_id,
        )

    fixture.checkpoint_payload.write_bytes(original)
    record = acquirer.execute(
        plan,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=plan.plan_id,
    )
    copied = destination / "acquisitions" / record.acquisition_id / "project" / "checkpoint.bin"
    copied.write_bytes(b"tampered-receiver-copy")
    with pytest.raises(ExperimentPromotionAcquisitionError, match="conflicts with the plan"):
        acquirer.execute(
            plan,
            fixture.archive,
            fixture.attestation,
            trusted_public_keys=(fixture.public_key,),
            confirm_plan_id=plan.plan_id,
        )


def test_acquisition_reports_remote_references_without_network_access(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(
        tmp_path / "source",
        remote_checkpoint=True,
    )
    destination = tmp_path / "received"
    acquirer = ExperimentPromotionAcquirer(fixture.project_root, fixture.state_root)

    strict = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    assert not strict.eligible
    assert strict.unresolved_remote_count == 1
    assert {item.code for item in strict.checks if not item.passed} == {
        "artifacts.remote",
        "checkpoint.payloads",
    }
    remote = next(
        item
        for item in strict.artifacts
        if item.availability is ExperimentPromotionArtifactAvailability.REMOTE
    )
    assert remote.reference.locator.startswith("https://unreachable.invalid/")
    assert remote.output_path is None

    relaxed = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
        policy=ExperimentPromotionAcquisitionPolicy(
            allow_unresolved_remote=True,
            require_checkpoint_payloads=False,
        ),
    )
    assert relaxed.eligible
    record = acquirer.execute(
        relaxed,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=relaxed.plan_id,
    )
    prefix = destination / "acquisitions" / record.acquisition_id
    assert not any("unreachable.invalid" in item.as_posix() for item in prefix.rglob("*"))


def test_acquisition_requires_complete_checkpoint_ancestry(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(
        tmp_path / "source",
        parent_checkpoint_id="checkpoint-missing-parent",
    )
    destination = tmp_path / "received"
    acquirer = ExperimentPromotionAcquirer(fixture.project_root, fixture.state_root)

    strict = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    assert not strict.eligible
    assert strict.checkpoint_ancestry_issues == (
        "missing_parent:checkpoint-acquisition-fixture:checkpoint-missing-parent",
    )
    assert {item.code for item in strict.checks if not item.passed} == {"checkpoint.ancestry"}

    relaxed = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
        policy=ExperimentPromotionAcquisitionPolicy(
            require_checkpoint_ancestry=False,
        ),
    )
    assert relaxed.eligible


def test_acquisition_requires_receiver_dataset_lineage(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(
        tmp_path / "source",
        broken_dataset_lineage=True,
    )
    destination = tmp_path / "received"
    acquirer = ExperimentPromotionAcquirer(fixture.project_root, fixture.state_root)

    strict = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    assert not strict.eligible
    assert strict.dataset_lineage_issues == (
        "unresolved_dataset_lineage:checkpoint-acquisition-fixture:" + "2" * 64,
    )
    assert {item.code for item in strict.checks if not item.passed} == {"dataset.lineage"}

    relaxed = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
        policy=ExperimentPromotionAcquisitionPolicy(
            require_dataset_lineage=False,
        ),
    )
    assert relaxed.eligible


def test_acquisition_fails_closed_on_trust_paths_and_unknown_progress(
    tmp_path: Path,
) -> None:
    fixture = build_acquisition_fixture(tmp_path / "source")
    destination = tmp_path / "received"
    acquirer = ExperimentPromotionAcquirer(fixture.project_root, fixture.state_root)
    outsider = Ed25519ManifestSigner.generate()

    untrusted = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(outsider.public_key_base64,),
    )
    assert not untrusted.eligible
    assert {item.code for item in untrusted.checks if not item.passed} == {"archive.trusted"}
    with pytest.raises(ValueError, match="separate from sources"):
        acquirer.preview(
            fixture.archive,
            fixture.attestation,
            fixture.project_root / "receiver",
            trusted_public_keys=(fixture.public_key,),
        )

    dataset = fixture.project_root / "dataset.jsonl"
    dataset_bytes = dataset.read_bytes()
    dataset.unlink()
    missing = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    assert any(
        item.reference.locator == "dataset.jsonl"
        and item.availability is ExperimentPromotionArtifactAvailability.MISSING
        for item in missing.artifacts
    )
    try:
        dataset.symlink_to(fixture.project_root / "checkpoint.bin")
    except OSError as error:
        if os.name != "nt" or getattr(error, "winerror", None) != 1314:
            raise
        dataset.write_bytes(dataset_bytes)
    else:
        unsafe = acquirer.preview(
            fixture.archive,
            fixture.attestation,
            destination,
            trusted_public_keys=(fixture.public_key,),
        )
        assert any(
            item.reference.locator == "dataset.jsonl"
            and item.availability is ExperimentPromotionArtifactAvailability.UNSAFE
            for item in unsafe.artifacts
        )
        dataset.unlink()
        dataset.write_bytes(dataset_bytes)

    plan = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    with pytest.raises(ExperimentPromotionAcquisitionError, match="confirmation"):
        acquirer.execute(
            plan,
            fixture.archive,
            fixture.attestation,
            trusted_public_keys=(fixture.public_key,),
            confirm_plan_id="promotion_acquisition_plan_" + "0" * 24,
        )
    acquisition_id = ExperimentPromotionAcquisitionRecord.expected_acquisition_id(plan.plan_id)
    prefix = destination / "acquisitions" / acquisition_id
    prefix.mkdir(parents=True)
    (prefix / "notes.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ExperimentPromotionAcquisitionError, match="unexpected file"):
        acquirer.execute(
            plan,
            fixture.archive,
            fixture.attestation,
            trusted_public_keys=(fixture.public_key,),
            confirm_plan_id=plan.plan_id,
        )


def test_acquisition_contracts_reject_mutated_evidence(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(tmp_path / "source")
    acquirer = ExperimentPromotionAcquirer(fixture.project_root, fixture.state_root)
    plan = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        tmp_path / "received",
        trusted_public_keys=(fixture.public_key,),
    )
    payload = plan.model_dump(mode="python")

    with pytest.raises(ValueError, match="counts"):
        ExperimentPromotionAcquisitionPlan.model_validate({**payload, "verified_artifact_count": 0})
    with pytest.raises(ValueError, match="checks"):
        ExperimentPromotionAcquisitionPlan.model_validate(
            {**payload, "checks": tuple(reversed(plan.checks))}
        )
    with pytest.raises(ValueError, match="plan ID"):
        ExperimentPromotionAcquisitionPlan.model_validate(
            {**payload, "plan_id": "promotion_acquisition_plan_" + "0" * 24}
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        ExperimentPromotionAcquisitionPlan.model_validate(
            {
                **payload,
                "checkpoint_payload_issues": ("z", "a"),
            }
        )
    record = ExperimentPromotionAcquisitionRecord(
        acquisition_id=ExperimentPromotionAcquisitionRecord.expected_acquisition_id(plan.plan_id),
        plan=plan,
        materialized_file_count=plan.materialized_file_count,
        materialized_size_bytes=plan.materialized_size_bytes,
    )
    with pytest.raises(ValueError, match="acquisition ID"):
        ExperimentPromotionAcquisitionRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "acquisition_id": "promotion_acquisition_" + "0" * 24,
            }
        )


def test_acquisition_cli_preview_execute_and_ci_failure(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(tmp_path / "source")
    runner = CliRunner()
    destination = tmp_path / "received"
    attestation_path = tmp_path / "attestation.json"
    public_key_path = tmp_path / "release.pub"
    plan_path = tmp_path / "plan.json"
    record_path = tmp_path / "record.json"
    attestation_path.write_bytes(fixture.attestation.canonical_bytes() + b"\n")
    public_key_path.write_text(f"{fixture.public_key}\n", encoding="ascii")
    arguments = [
        "experiment-promotion-acquire",
        str(fixture.archive),
        str(attestation_path),
        str(destination),
        "--root",
        str(fixture.project_root),
        "--state-dir",
        str(fixture.state_root),
        "--public-key",
        str(public_key_path),
    ]

    previewed = runner.invoke(
        app,
        [
            *arguments,
            "--plan-output",
            str(plan_path),
            "--fail-on-ineligible",
        ],
    )
    assert previewed.exit_code == 0, previewed.output
    plan = ExperimentPromotionAcquisitionPlan.model_validate_json(plan_path.read_bytes())
    assert plan.eligible
    assert not destination.exists()

    executed = runner.invoke(
        app,
        [
            *arguments,
            "--execute",
            "--confirm-plan-id",
            plan.plan_id,
            "--record-output",
            str(record_path),
        ],
    )
    assert executed.exit_code == 0, executed.output
    record = ExperimentPromotionAcquisitionRecord.model_validate_json(record_path.read_bytes())
    assert record.plan == plan
    assert (destination / "acquisitions" / record.acquisition_id / "record.json").is_file()

    ineligible_path = tmp_path / "ineligible-plan.json"
    ineligible = runner.invoke(
        app,
        [
            "experiment-promotion-acquire",
            str(fixture.archive),
            str(attestation_path),
            str(tmp_path / "bounded"),
            "--root",
            str(fixture.project_root),
            "--state-dir",
            str(fixture.state_root),
            "--public-key",
            str(public_key_path),
            "--max-materialized-bytes",
            "1",
            "--plan-output",
            str(ineligible_path),
            "--fail-on-ineligible",
        ],
    )
    assert ineligible.exit_code == 1, ineligible.output
    bounded = ExperimentPromotionAcquisitionPlan.model_validate_json(ineligible_path.read_bytes())
    assert not bounded.eligible
    assert {item.code for item in bounded.checks if not item.passed} == {"materialization.bytes"}

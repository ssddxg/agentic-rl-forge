from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.data import Ed25519ManifestSigner
from agentic_rl_forge.experiments import (
    ExperimentPromotionAliasAction,
    ExperimentPromotionAliasEvent,
    ExperimentPromotionAliasPolicy,
    ExperimentPromotionAliasPreview,
    ExperimentPromotionGovernancePolicy,
    ExperimentPromotionLifecycleEvent,
    ExperimentPromotionLifecyclePreview,
    ExperimentPromotionRegistry,
    ExperimentPromotionRegistryError,
    ExperimentPromotionRegistryIssue,
    ExperimentPromotionRegistryStatus,
    ExperimentPromotionStage,
    SignedExperimentPromotionArchiveAttestation,
)
from agentic_rl_forge.experiments.promotion_archives import (
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestor,
    ExperimentPromotionArchiveReceipt,
)
from agentic_rl_forge.storage import LocalBlobStore
from test_experiment_promotion_archives import build_promotion


@dataclass(frozen=True)
class SignedPromotion:
    archive: Path
    receipt: ExperimentPromotionArchiveReceipt
    attestation: SignedExperimentPromotionArchiveAttestation
    public_key: str


def build_signed_promotion(
    root: Path,
    *,
    name: str,
    signer: Ed25519ManifestSigner | None = None,
) -> SignedPromotion:
    promotion, _ = build_promotion(root, name=name)
    archive = root / f"{name}.promotion.tar.gz"
    receipt = ExperimentPromotionArchive(promotion).pack(archive)
    resolved_signer = signer or Ed25519ManifestSigner.generate()
    attestation = ExperimentPromotionArchiveAttestor(resolved_signer).sign(receipt)
    return SignedPromotion(
        archive=archive,
        receipt=receipt,
        attestation=attestation,
        public_key=resolved_signer.public_key_base64,
    )


def transition(
    registry: ExperimentPromotionRegistry,
    release: SignedPromotion,
    stage: ExperimentPromotionStage,
    *,
    operator: str = "release-operator",
    authorizers: tuple[str, ...] = ("release-authorizer",),
) -> ExperimentPromotionLifecycleEvent:
    preview = registry.preview_lifecycle(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        target_stage=stage,
        operator=operator,
        reason=f"advance verified release to {stage.value}",
        authorizers=authorizers,
    )
    assert preview.eligible
    return registry.execute_lifecycle(
        preview,
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        confirm_preview_id=preview.preview_id,
    )


def transition_to_production(
    registry: ExperimentPromotionRegistry,
    release: SignedPromotion,
) -> ExperimentPromotionLifecycleEvent:
    transition(registry, release, ExperimentPromotionStage.CANDIDATE)
    transition(registry, release, ExperimentPromotionStage.STAGING)
    return transition(registry, release, ExperimentPromotionStage.PRODUCTION)


def assign(
    registry: ExperimentPromotionRegistry,
    release: SignedPromotion,
    *,
    environment: str = "production",
    action: ExperimentPromotionAliasAction = ExperimentPromotionAliasAction.ASSIGN,
) -> ExperimentPromotionAliasEvent:
    preview = registry.preview_alias(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        environment=environment,
        action=action,
        operator="deployment-operator",
        reason=f"{action.value} verified release for {environment}",
        authorizers=("deployment-authorizer",),
    )
    assert preview.eligible
    return registry.execute_alias(
        preview,
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        confirm_preview_id=preview.preview_id,
    )


def test_lifecycle_requires_trust_governance_and_ordered_transitions(tmp_path: Path) -> None:
    release = build_signed_promotion(tmp_path / "release", name="candidate-a")
    registry = ExperimentPromotionRegistry(LocalBlobStore(tmp_path / "registry"))

    empty = registry.status()
    assert empty.valid
    assert not empty.promotions
    assert not empty.aliases
    assert registry.status() == empty

    skipped = registry.preview_lifecycle(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        target_stage=ExperimentPromotionStage.STAGING,
        operator="release-operator",
        reason="reject a lifecycle stage skip",
        authorizers=("release-authorizer",),
    )
    assert not skipped.eligible
    assert {item.code for item in skipped.checks if not item.passed} == {"transition.allowed"}
    with pytest.raises(ExperimentPromotionRegistryError, match="not eligible"):
        registry.execute_lifecycle(
            skipped,
            release.archive,
            release.attestation,
            trusted_public_keys=(release.public_key,),
            confirm_preview_id=skipped.preview_id,
        )

    outsider = Ed25519ManifestSigner.generate()
    untrusted = registry.preview_lifecycle(
        release.archive,
        release.attestation,
        trusted_public_keys=(outsider.public_key_base64,),
        target_stage=ExperimentPromotionStage.CANDIDATE,
        operator="release-operator",
        reason="reject an untrusted publisher",
        authorizers=("release-authorizer",),
    )
    assert not untrusted.eligible
    assert not next(item for item in untrusted.checks if item.code == "attestation.trusted").passed

    weak_governance = registry.preview_lifecycle(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        target_stage=ExperimentPromotionStage.CANDIDATE,
        operator="release-operator",
        reason="require an independent two-person quorum",
        authorizers=("release-operator",),
        policy=ExperimentPromotionGovernancePolicy(minimum_authorizers=2),
    )
    assert not weak_governance.eligible
    assert {item.code for item in weak_governance.checks if not item.passed} == {
        "authorization.independent",
        "authorization.quorum",
    }

    candidate_preview = registry.preview_lifecycle(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        target_stage=ExperimentPromotionStage.CANDIDATE,
        operator="release-operator",
        reason="register the verified release candidate",
        authorizers=("release-authorizer",),
    )
    candidate = registry.execute_lifecycle(
        candidate_preview,
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        confirm_preview_id=candidate_preview.preview_id,
    )
    repeated = registry.execute_lifecycle(
        candidate_preview,
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        confirm_preview_id=candidate_preview.preview_id,
    )
    assert repeated == candidate
    assert registry.lifecycle(release.receipt.promotion_id) == (candidate,)

    staging = transition(registry, release, ExperimentPromotionStage.STAGING)
    production = transition(registry, release, ExperimentPromotionStage.PRODUCTION)
    history = registry.lifecycle(release.receipt.promotion_id)
    assert history == (candidate, staging, production)
    assert [item.preview.target_stage for item in history] == [
        ExperimentPromotionStage.CANDIDATE,
        ExperimentPromotionStage.STAGING,
        ExperimentPromotionStage.PRODUCTION,
    ]
    status = registry.status()
    assert status.valid
    assert status.promotions[0].stage is ExperimentPromotionStage.PRODUCTION
    assert status.promotions[0].event_count == 3


def test_environment_alias_is_cas_guarded_and_preserves_rollback_ancestry(
    tmp_path: Path,
) -> None:
    signer = Ed25519ManifestSigner.generate()
    first = build_signed_promotion(
        tmp_path / "first",
        name="candidate-a",
        signer=signer,
    )
    second = build_signed_promotion(
        tmp_path / "second",
        name="candidate-b",
        signer=signer,
    )
    registry = ExperimentPromotionRegistry(LocalBlobStore(tmp_path / "registry"))
    transition_to_production(registry, first)
    transition_to_production(registry, second)

    first_preview = registry.preview_alias(
        first.archive,
        first.attestation,
        trusted_public_keys=(first.public_key,),
        environment="production",
        action=ExperimentPromotionAliasAction.ASSIGN,
        operator="deployment-operator",
        reason="deploy the first verified production release",
        authorizers=("deployment-authorizer",),
    )
    competing_preview = registry.preview_alias(
        second.archive,
        second.attestation,
        trusted_public_keys=(second.public_key,),
        environment="production",
        action=ExperimentPromotionAliasAction.ASSIGN,
        operator="deployment-operator",
        reason="deploy the competing production release",
        authorizers=("deployment-authorizer",),
    )
    first_alias = registry.execute_alias(
        first_preview,
        first.archive,
        first.attestation,
        trusted_public_keys=(first.public_key,),
        confirm_preview_id=first_preview.preview_id,
    )
    with pytest.raises(ExperimentPromotionRegistryError, match="another decision"):
        registry.execute_alias(
            competing_preview,
            second.archive,
            second.attestation,
            trusted_public_keys=(second.public_key,),
            confirm_preview_id=competing_preview.preview_id,
        )

    no_change = registry.preview_alias(
        first.archive,
        first.attestation,
        trusted_public_keys=(first.public_key,),
        environment="production",
        action=ExperimentPromotionAliasAction.ASSIGN,
        operator="deployment-operator",
        reason="reject a duplicate alias assignment",
        authorizers=("deployment-authorizer",),
    )
    assert not no_change.eligible
    assert not next(item for item in no_change.checks if item.code == "alias.changed").passed

    second_alias = assign(registry, second)
    rollback = assign(
        registry,
        first,
        action=ExperimentPromotionAliasAction.ROLLBACK,
    )
    assert rollback.preview.rollback_of_event_id == second_alias.event_id
    assert rollback.preview.rollback_target_event_id == first_alias.event_id
    assert rollback.preview.rollback_ancestor_verified
    history = registry.alias_history("production")
    assert history == (first_alias, second_alias, rollback)
    assert (
        registry.execute_alias(
            rollback.preview,
            first.archive,
            first.attestation,
            trusted_public_keys=(first.public_key,),
            confirm_preview_id=rollback.preview.preview_id,
        )
        == rollback
    )

    status = registry.status()
    assert status.valid
    assert status.aliases[0].promotion_id == first.receipt.promotion_id
    assert status.aliases[0].generation == 3
    assert status.aliases[0].action is ExperimentPromotionAliasAction.ROLLBACK


def test_retirement_requires_detached_alias_and_status_rejects_corruption(
    tmp_path: Path,
) -> None:
    first = build_signed_promotion(tmp_path / "first", name="candidate-a")
    second = build_signed_promotion(tmp_path / "second", name="candidate-b")
    registry_root = tmp_path / "registry"
    registry = ExperimentPromotionRegistry(LocalBlobStore(registry_root))
    transition_to_production(registry, first)
    transition_to_production(registry, second)
    assign(registry, first)

    attached = registry.preview_lifecycle(
        first.archive,
        first.attestation,
        trusted_public_keys=(first.public_key,),
        target_stage=ExperimentPromotionStage.RETIRED,
        operator="release-operator",
        reason="attempt retirement while production still points here",
        authorizers=("release-authorizer",),
    )
    assert not attached.eligible
    retirement_check = next(item for item in attached.checks if item.code == "retirement.detached")
    assert not retirement_check.passed
    assert retirement_check.evidence == ("production",)

    assign(registry, second)
    retired = transition(registry, first, ExperimentPromotionStage.RETIRED)
    assert retired.preview.target_stage is ExperimentPromotionStage.RETIRED
    status = registry.status()
    assert status.valid
    stages = {item.promotion_id: item.stage for item in status.promotions}
    assert stages[first.receipt.promotion_id] is ExperimentPromotionStage.RETIRED
    assert status.aliases[0].promotion_id == second.receipt.promotion_id

    (registry_root / "operator-notes.txt").write_text("unexpected", encoding="utf-8")
    corrupted = registry.status()
    assert not corrupted.valid
    assert corrupted.issues == (
        ExperimentPromotionRegistryIssue(
            path="operator-notes.txt",
            detail="unrecognized promotion registry entry",
        ),
    )
    with pytest.raises(ExperimentPromotionRegistryError, match="invalid or unrecognized"):
        registry.preview_alias(
            second.archive,
            second.attestation,
            trusted_public_keys=(second.public_key,),
            environment="canary",
            action=ExperimentPromotionAliasAction.ASSIGN,
            operator="deployment-operator",
            reason="block mutation against a corrupt registry",
            authorizers=("deployment-authorizer",),
        )


def test_alias_stage_policy_and_rollback_fail_closed(tmp_path: Path) -> None:
    release = build_signed_promotion(tmp_path / "release", name="candidate-a")
    registry = ExperimentPromotionRegistry(LocalBlobStore(tmp_path / "registry"))
    transition(registry, release, ExperimentPromotionStage.CANDIDATE)

    production_only = registry.preview_alias(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        environment="production",
        action=ExperimentPromotionAliasAction.ASSIGN,
        operator="deployment-operator",
        reason="reject a candidate from production",
        authorizers=("deployment-authorizer",),
    )
    assert not production_only.eligible
    assert not next(
        item for item in production_only.checks if item.code == "lifecycle.stage"
    ).passed

    candidate_policy = ExperimentPromotionAliasPolicy(
        allowed_stages=(ExperimentPromotionStage.CANDIDATE,),
    )
    candidate_alias = registry.preview_alias(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        environment="evaluation",
        action=ExperimentPromotionAliasAction.ASSIGN,
        operator="deployment-operator",
        reason="assign a reviewed candidate to evaluation",
        authorizers=("deployment-authorizer",),
        policy=candidate_policy,
    )
    assert candidate_alias.eligible
    registry.execute_alias(
        candidate_alias,
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        confirm_preview_id=candidate_alias.preview_id,
    )

    impossible_rollback = registry.preview_alias(
        release.archive,
        release.attestation,
        trusted_public_keys=(release.public_key,),
        environment="new-environment",
        action=ExperimentPromotionAliasAction.ROLLBACK,
        operator="deployment-operator",
        reason="reject rollback without environment ancestry",
        authorizers=("deployment-authorizer",),
        policy=candidate_policy,
    )
    assert not impossible_rollback.eligible
    assert not next(
        item for item in impossible_rollback.checks if item.code == "rollback.ancestry"
    ).passed


def test_registry_contracts_and_canonical_streams_reject_mutation(tmp_path: Path) -> None:
    release = build_signed_promotion(tmp_path / "release", name="candidate-a")
    registry_root = tmp_path / "registry"
    registry = ExperimentPromotionRegistry(LocalBlobStore(registry_root))
    event = transition(registry, release, ExperimentPromotionStage.CANDIDATE)

    with pytest.raises(ValueError, match=r"sorted.*unique"):
        ExperimentPromotionAliasPolicy(
            allowed_stages=(
                ExperimentPromotionStage.PRODUCTION,
                ExperimentPromotionStage.CANDIDATE,
            )
        )
    with pytest.raises(ValueError, match="retired promotions"):
        ExperimentPromotionAliasPolicy(
            allowed_stages=(ExperimentPromotionStage.RETIRED,),
        )
    with pytest.raises(ValueError, match="safe and relative"):
        ExperimentPromotionRegistryIssue(path="../escape", detail="invalid")
    with pytest.raises(ValueError, match="event ID"):
        ExperimentPromotionLifecycleEvent(
            event_id="promotion_lifecycle_event_" + "0" * 24,
            preview=event.preview,
        )

    status = registry.status()
    with pytest.raises(ValueError, match="status ID"):
        ExperimentPromotionRegistryStatus(
            status_id="promotion_registry_status_" + "0" * 24,
            promotions=status.promotions,
            aliases=status.aliases,
            issues=status.issues,
            valid=status.valid,
        )

    event_path = registry_root / ExperimentPromotionRegistry.lifecycle_key(
        release.receipt.promotion_id,
        1,
    )
    event_path.write_bytes(event_path.read_bytes() + b"\n")
    corrupted = registry.status()
    assert not corrupted.valid
    assert "not canonical" in corrupted.issues[0].detail
    with pytest.raises(ExperimentPromotionRegistryError, match="not canonical"):
        registry.lifecycle(release.receipt.promotion_id)

    assert ExperimentPromotionRegistry.lifecycle_key(release.receipt.promotion_id, 1).endswith(
        "/00000001.json"
    )
    assert ExperimentPromotionRegistry.alias_key("production", 1) == (
        "environments/production/00000001.json"
    )
    with pytest.raises(ValueError, match="invalid lifecycle"):
        ExperimentPromotionRegistry.lifecycle_key("bad", 0)
    with pytest.raises(ValueError, match="invalid environment"):
        ExperimentPromotionRegistry.alias_key("../bad", 0)


def test_registry_cli_previews_confirms_and_reports_ci_status(tmp_path: Path) -> None:
    release = build_signed_promotion(tmp_path / "release", name="candidate-a")
    registry_dir = tmp_path / "registry"
    public_key = tmp_path / "release.pub"
    attestation = tmp_path / "release.attestation.json"
    public_key.write_text(release.public_key + "\n", encoding="ascii")
    attestation.write_bytes(release.attestation.canonical_bytes() + b"\n")
    runner = CliRunner()

    for index, stage in enumerate(
        (
            ExperimentPromotionStage.CANDIDATE,
            ExperimentPromotionStage.STAGING,
            ExperimentPromotionStage.PRODUCTION,
        ),
        1,
    ):
        preview_path = tmp_path / f"lifecycle-{index}-preview.json"
        event_path = tmp_path / f"lifecycle-{index}-event.json"
        common = [
            "experiment-promotion-lifecycle",
            str(release.archive),
            str(attestation),
            "--registry-dir",
            str(registry_dir),
            "--target-stage",
            stage.value,
            "--operator",
            "cli-release-operator",
            "--reason",
            f"advance CLI release to {stage.value}",
            "--authorizer",
            "cli-release-authorizer",
            "--public-key",
            str(public_key),
        ]
        previewed = runner.invoke(
            app,
            [*common, "--preview-output", str(preview_path)],
        )
        assert previewed.exit_code == 0, previewed.output
        preview = ExperimentPromotionLifecyclePreview.model_validate_json(preview_path.read_bytes())
        assert preview.eligible
        missing_confirmation = runner.invoke(app, [*common, "--execute"])
        assert missing_confirmation.exit_code == 2
        executed = runner.invoke(
            app,
            [
                *common,
                "--execute",
                "--confirm-preview-id",
                preview.preview_id,
                "--event-output",
                str(event_path),
            ],
        )
        assert executed.exit_code == 0, executed.output
        event = ExperimentPromotionLifecycleEvent.model_validate_json(event_path.read_bytes())
        assert event.preview == preview

    alias_preview_path = tmp_path / "alias-preview.json"
    alias_event_path = tmp_path / "alias-event.json"
    alias_common = [
        "experiment-promotion-alias",
        str(release.archive),
        str(attestation),
        "--registry-dir",
        str(registry_dir),
        "--environment",
        "production",
        "--action",
        "assign",
        "--operator",
        "cli-deployment-operator",
        "--reason",
        "assign the verified CLI production release",
        "--authorizer",
        "cli-deployment-authorizer",
        "--public-key",
        str(public_key),
    ]
    alias_previewed = runner.invoke(
        app,
        [*alias_common, "--preview-output", str(alias_preview_path)],
    )
    assert alias_previewed.exit_code == 0, alias_previewed.output
    alias_preview = ExperimentPromotionAliasPreview.model_validate_json(
        alias_preview_path.read_bytes()
    )
    assert alias_preview.eligible
    alias_executed = runner.invoke(
        app,
        [
            *alias_common,
            "--execute",
            "--confirm-preview-id",
            alias_preview.preview_id,
            "--event-output",
            str(alias_event_path),
        ],
    )
    assert alias_executed.exit_code == 0, alias_executed.output
    assert (
        ExperimentPromotionAliasEvent.model_validate_json(alias_event_path.read_bytes()).preview
        == alias_preview
    )

    status_path = tmp_path / "registry-status.json"
    status_result = runner.invoke(
        app,
        [
            "experiment-promotion-registry-status",
            str(registry_dir),
            "--output",
            str(status_path),
            "--fail-on-issues",
        ],
    )
    assert status_result.exit_code == 0, status_result.output
    status = ExperimentPromotionRegistryStatus.model_validate_json(status_path.read_bytes())
    assert status.valid
    assert status.promotions[0].stage is ExperimentPromotionStage.PRODUCTION
    assert status.aliases[0].environment == "production"

    ineligible_path = tmp_path / "ineligible-lifecycle.json"
    ineligible = runner.invoke(
        app,
        [
            "experiment-promotion-lifecycle",
            str(release.archive),
            str(attestation),
            "--registry-dir",
            str(registry_dir),
            "--target-stage",
            "candidate",
            "--operator",
            "cli-release-operator",
            "--reason",
            "reject a production to candidate reversal",
            "--authorizer",
            "cli-release-authorizer",
            "--public-key",
            str(public_key),
            "--preview-output",
            str(ineligible_path),
            "--fail-on-ineligible",
        ],
    )
    assert ineligible.exit_code == 1
    assert not ExperimentPromotionLifecyclePreview.model_validate_json(
        ineligible_path.read_bytes()
    ).eligible

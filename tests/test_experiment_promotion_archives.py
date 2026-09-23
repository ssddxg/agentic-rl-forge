from __future__ import annotations

import base64
import hashlib
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.data import Ed25519ManifestSigner
from agentic_rl_forge.experiments import (
    ExperimentAnalyzer,
    ExperimentObjectiveDirection,
    ExperimentPromoter,
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestation,
    ExperimentPromotionArchiveAttestationVerification,
    ExperimentPromotionArchiveAttestor,
    ExperimentPromotionArchiveError,
    ExperimentPromotionArchiveReceipt,
    ExperimentPromotionPolicy,
    ExperimentPromotionRecord,
    ExperimentRankingSpec,
    SignedExperimentPromotionArchiveAttestation,
)
from test_experiment_operations import completed_index, objective
from test_run_artifacts import read_archive_members, write_test_archive


def build_promotion(
    tmp_path: Path,
    *,
    name: str = "release-candidate",
) -> tuple[Path, ExperimentPromotionRecord]:
    project = tmp_path / "project"
    index, state = completed_index(project, include_issue=False)
    score = objective(index, "score", ExperimentObjectiveDirection.MAXIMIZE)
    analysis = ExperimentAnalyzer().analyze(
        index,
        ExperimentRankingSpec(objectives=(score,)),
    )
    selected = analysis.trials[0]
    promotion_root = tmp_path / "promotions"
    promoter = ExperimentPromoter(project, state, promotion_root)
    preview = promoter.preview(
        index,
        analysis,
        promotion_name=name,
        plan_id=selected.plan_id,
        trial_id=selected.trial_id,
        policy=ExperimentPromotionPolicy(),
    )
    record = promoter.promote(
        index,
        analysis,
        preview,
        confirm_preview_id=preview.preview_id,
        operator="release-operator",
        reason="approve the verified release candidate metadata",
        approvers=("release-reviewer",),
    )
    return promotion_root / name, record


def test_promotion_archive_is_deterministic_semantic_and_portable(tmp_path: Path) -> None:
    promotion, record = build_promotion(tmp_path)
    first_path = tmp_path / "first.tar.gz"
    second_path = tmp_path / "second.tar.gz"

    first = ExperimentPromotionArchive(promotion).pack(first_path)
    second = ExperimentPromotionArchive(promotion).pack(second_path)
    retried = ExperimentPromotionArchive(promotion).pack(first_path)

    assert first == second == retried
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.promotion_id == record.promotion_id
    assert first.promotion_name == record.preview.promotion_name
    assert first.reproducibility_manifest_id == record.preview.manifest.manifest_id
    assert first.member_count == 8
    assert (
        ExperimentPromotionArchive.inspect(
            first_path,
            expected_sha256=first.content_digest,
        )
        == first
    )
    checksum = ExperimentPromotionArchive.checksum_path(first_path)
    assert ExperimentPromotionArchive.read_checksum(checksum) == first.content_digest
    assert checksum.read_text(encoding="ascii") == f"{first.content_digest}\n"
    windows_checksum = tmp_path / "windows.sha256"
    windows_checksum.write_bytes(f"{first.content_digest}\r\n".encode("ascii"))
    assert ExperimentPromotionArchive.read_checksum(windows_checksum) == first.content_digest
    if os.name != "nt":
        assert first_path.stat().st_mode & 0o777 == 0o644
        assert checksum.stat().st_mode & 0o777 == 0o644

    destination = tmp_path / "received"
    unpacked = ExperimentPromotionArchive.unpack(
        first_path,
        destination,
        expected_sha256=first.content_digest,
    )

    assert unpacked == first
    assert ExperimentPromotionArchive.verify_directory(destination) == record
    assert {item.name for item in destination.iterdir()} == set(ExperimentPromotionArchive.MEMBERS)
    if os.name != "nt":
        assert destination.stat().st_mode & 0o777 == 0o755
        assert all(item.stat().st_mode & 0o777 == 0o644 for item in destination.iterdir())
    assert not any(item.is_symlink() for item in destination.iterdir())


def test_promotion_archive_rejects_source_conflicts_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    promotion, _ = build_promotion(tmp_path)
    archive = tmp_path / "canonical.tar.gz"
    receipt = ExperimentPromotionArchive(promotion).pack(archive)

    model_card = promotion / "model-card.md"
    original_model_card = model_card.read_bytes()
    model_card.write_bytes(original_model_card + b"changed\n")
    with pytest.raises(ExperimentPromotionArchiveError, match="cannot be reproduced"):
        ExperimentPromotionArchive(promotion).pack(tmp_path / "changed.tar.gz")
    model_card.write_bytes(original_model_card)

    extra = promotion / "notes.txt"
    extra.write_text("not part of the immutable decision", encoding="utf-8")
    with pytest.raises(ExperimentPromotionArchiveError, match=r"unexpected notes\.txt"):
        ExperimentPromotionArchive(promotion).pack(tmp_path / "extra.tar.gz")
    extra.unlink()

    manifest = promotion / "manifest.json"
    manifest_payload = manifest.read_bytes()
    manifest.unlink()
    try:
        manifest.symlink_to("decision.json")
    except OSError as error:
        if os.name != "nt" or getattr(error, "winerror", None) != 1314:
            raise
        manifest.write_bytes(manifest_payload)
    else:
        with pytest.raises(ExperimentPromotionArchiveError, match="must be a regular file"):
            ExperimentPromotionArchive(promotion).pack(tmp_path / "symlink.tar.gz")
        manifest.unlink()
        manifest.write_bytes(manifest_payload)

    with pytest.raises(ExperimentPromotionArchiveError, match="outside its source"):
        ExperimentPromotionArchive(promotion).pack(promotion / "nested.tar.gz")

    conflict = tmp_path / "conflict.tar.gz"
    conflict.write_bytes(b"existing")
    with pytest.raises(ExperimentPromotionArchiveError, match="different content"):
        ExperimentPromotionArchive(promotion).pack(conflict)

    checksum = ExperimentPromotionArchive.checksum_path(archive)
    checksum.write_text("invalid\n", encoding="ascii")
    with pytest.raises(ExperimentPromotionArchiveError, match="not canonical"):
        ExperimentPromotionArchive.read_checksum(checksum)

    changed_wrapper = bytearray(archive.read_bytes())
    changed_wrapper[8] = 0
    noncanonical = tmp_path / "noncanonical.tar.gz"
    noncanonical.write_bytes(changed_wrapper)
    digest = hashlib.sha256(changed_wrapper).hexdigest()
    with pytest.raises(ExperimentPromotionArchiveError, match="bytes are not canonical"):
        ExperimentPromotionArchive.inspect(noncanonical, expected_sha256=digest)

    with pytest.raises(ExperimentPromotionArchiveError, match="SHA-256 does not match"):
        ExperimentPromotionArchive.unpack(
            archive,
            tmp_path / "digest-mismatch",
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "digest-mismatch").exists()
    assert receipt.content_digest != "0" * 64


def test_promotion_archive_rejects_unsafe_members_and_cleans_partial_output(
    tmp_path: Path,
) -> None:
    promotion, _ = build_promotion(tmp_path)
    canonical = tmp_path / "canonical.tar.gz"
    ExperimentPromotionArchive(promotion).pack(canonical)
    original = read_archive_members(canonical)

    unsafe_archives: dict[str, list[tuple[tarfile.TarInfo, bytes]]] = {}
    reordered = list(original)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    unsafe_archives["reordered"] = reordered

    traversal = list(original)
    traversal_info = tarfile.TarInfo("../decision.json")
    traversal_info.mode = 0o644
    traversal_info.mtime = 0
    traversal[0] = (traversal_info, traversal[0][1])
    unsafe_archives["traversal"] = traversal

    linked = list(original)
    link = tarfile.TarInfo(linked[1][0].name)
    link.mode = 0o644
    link.mtime = 0
    link.type = tarfile.SYMTYPE
    link.linkname = original[0][0].name
    linked[1] = (link, b"")
    unsafe_archives["symlink"] = linked

    changed_metadata = list(original)
    metadata_info = tarfile.TarInfo(changed_metadata[0][0].name)
    metadata_info.mode = 0o600
    metadata_info.mtime = 0
    changed_metadata[0] = (metadata_info, changed_metadata[0][1])
    unsafe_archives["metadata"] = changed_metadata

    corrupted = list(original)
    model_card_index = next(
        index for index, (info, _) in enumerate(corrupted) if info.name == "model-card.md"
    )
    model_card_info, model_card_payload = corrupted[model_card_index]
    corrupted[model_card_index] = (model_card_info, model_card_payload + b"tampered\n")
    unsafe_archives["semantic"] = corrupted

    for label, members in unsafe_archives.items():
        archive = tmp_path / f"{label}.tar.gz"
        destination = tmp_path / f"{label}-destination"
        write_test_archive(archive, members)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        with pytest.raises(ExperimentPromotionArchiveError):
            ExperimentPromotionArchive.unpack(
                archive,
                destination,
                expected_sha256=digest,
            )
        assert not destination.exists()
        assert not tuple(tmp_path.glob(".arf-promotion-unpack-*"))


def test_promotion_archive_attestation_binds_receipt_and_detects_tampering(
    tmp_path: Path,
) -> None:
    promotion, _ = build_promotion(tmp_path)
    receipt = ExperimentPromotionArchive(promotion).pack(tmp_path / "release.tar.gz")
    signer = Ed25519ManifestSigner.generate()
    rotated = Ed25519ManifestSigner.generate()
    outsider = Ed25519ManifestSigner.generate()
    signed_at = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    signed = ExperimentPromotionArchiveAttestor(signer).sign(
        receipt,
        signed_at=signed_at,
    )
    repeated = ExperimentPromotionArchiveAttestor(signer).sign(
        receipt,
        signed_at=signed_at,
    )
    rotated_signed = ExperimentPromotionArchiveAttestor(rotated).sign(
        receipt,
        signed_at=signed_at,
    )
    trusted = (signer.public_key_base64, rotated.public_key_base64)

    assert signed == repeated
    assert ExperimentPromotionArchiveAttestor.verify(
        signed,
        receipt,
        trusted_public_keys=trusted,
    ).valid
    assert ExperimentPromotionArchiveAttestor.verify(
        rotated_signed,
        receipt,
        trusted_public_keys=trusted,
    ).valid
    untrusted = ExperimentPromotionArchiveAttestor.verify(
        signed,
        receipt,
        trusted_public_keys=(outsider.public_key_base64,),
    )
    assert not untrusted.valid
    assert untrusted.signature_valid
    assert not untrusted.trusted_signer

    other_digest = "f" * 64
    other_receipt = ExperimentPromotionArchiveReceipt(
        archive_id=ExperimentPromotionArchiveReceipt.expected_archive_id(other_digest),
        promotion_name=receipt.promotion_name,
        promotion_id=receipt.promotion_id,
        reproducibility_manifest_id=receipt.reproducibility_manifest_id,
        content_digest=other_digest,
        size_bytes=receipt.size_bytes,
    )
    replayed = ExperimentPromotionArchiveAttestor.verify(
        signed,
        other_receipt,
        trusted_public_keys=trusted,
    )
    assert not replayed.valid
    assert not replayed.archive_receipt_matches
    assert replayed.signature_valid

    signature_bytes = bytearray(base64.b64decode(signed.signature.signature_base64))
    signature_bytes[0] ^= 1
    changed_signature = signed.signature.model_copy(
        update={"signature_base64": base64.b64encode(signature_bytes).decode("ascii")}
    )
    tampered = signed.model_copy(update={"signature": changed_signature})
    tampered_result = ExperimentPromotionArchiveAttestor.verify(
        tampered,
        receipt,
        trusted_public_keys=trusted,
    )
    assert not tampered_result.valid
    assert not tampered_result.signature_valid


def test_promotion_archive_contracts_and_transport_inputs_fail_closed(tmp_path: Path) -> None:
    promotion, _ = build_promotion(tmp_path)
    archive = tmp_path / "release.tar.gz"
    receipt = ExperimentPromotionArchive(promotion).pack(archive)
    signer = Ed25519ManifestSigner.generate()
    signed_at = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    signed = ExperimentPromotionArchiveAttestor(signer).sign(
        receipt,
        signed_at=signed_at,
    )

    with pytest.raises(ValueError, match="archive ID"):
        ExperimentPromotionArchiveReceipt.model_validate(
            {
                **receipt.model_dump(mode="python"),
                "archive_id": "experiment_promotion_archive_" + "0" * 24,
            }
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        naive = datetime(2026, 9, 18, 12, 0)
        ExperimentPromotionArchiveAttestation(
            attestation_id=ExperimentPromotionArchiveAttestation.expected_attestation_id(
                receipt=receipt,
                signer_key_id=signer.key_id,
                signed_at=naive,
            ),
            receipt=receipt,
            signer_key_id=signer.key_id,
            signed_at=naive,
        )
    with pytest.raises(ValueError, match="attestation ID"):
        ExperimentPromotionArchiveAttestation.model_validate(
            {
                **signed.attestation.model_dump(mode="python"),
                "attestation_id": "experiment_promotion_attestation_" + "0" * 24,
            }
        )

    invalid_signatures = (
        (signed.signature.model_copy(update={"algorithm": "rsa"}), "requires Ed25519"),
        (
            signed.signature.model_copy(update={"public_key_base64": "not-base64"}),
            "public key is invalid",
        ),
        (
            signed.signature.model_copy(
                update={"public_key_base64": base64.b64encode(b"short").decode("ascii")}
            ),
            "must contain 32 bytes",
        ),
        (
            Ed25519ManifestSigner.generate().sign_payload(signed.attestation.canonical_bytes()),
            "signer key ID",
        ),
        (
            signed.signature.model_copy(update={"payload_sha256": "0" * 64}),
            "payload digest",
        ),
    )
    for signature, message in invalid_signatures:
        with pytest.raises(ValueError, match=message):
            SignedExperimentPromotionArchiveAttestation(
                attestation=signed.attestation,
                signature=signature,
            )

    with pytest.raises(ValueError, match="validity"):
        ExperimentPromotionArchiveAttestationVerification(
            attestation_id=signed.attestation.attestation_id,
            valid=True,
            archive_receipt_matches=True,
            signer_identity_valid=True,
            payload_digest_valid=True,
            signature_valid=True,
            trusted_signer=False,
        )

    with pytest.raises(ExperimentPromotionArchiveError, match="64 hex digits"):
        ExperimentPromotionArchive.normalize_digest("not-a-digest")
    with pytest.raises(ExperimentPromotionArchiveError, match="cannot read"):
        ExperimentPromotionArchive.read_checksum(tmp_path / "missing.sha256")
    non_ascii = tmp_path / "non-ascii.sha256"
    non_ascii.write_bytes(b"\xff" * 64 + b"\n")
    with pytest.raises(ExperimentPromotionArchiveError, match="not ASCII"):
        ExperimentPromotionArchive.read_checksum(non_ascii)
    with pytest.raises(
        ExperimentPromotionArchiveError,
        match=r"\.tar\.gz suffix",
    ):
        ExperimentPromotionArchive(promotion).pack(tmp_path / "release.zip")

    existing_destination = tmp_path / "existing"
    existing_destination.mkdir()
    with pytest.raises(ExperimentPromotionArchiveError, match="destination already exists"):
        ExperimentPromotionArchive.unpack(
            archive,
            existing_destination,
            expected_sha256=receipt.content_digest,
        )

    conflict = tmp_path / "checksum-conflict.tar.gz"
    ExperimentPromotionArchive.checksum_path(conflict).write_text(
        "0" * 64 + "\n",
        encoding="ascii",
    )
    with pytest.raises(ExperimentPromotionArchiveError, match="different content"):
        ExperimentPromotionArchive(promotion).pack(conflict)
    assert not conflict.exists()


def test_promotion_archive_cli_supports_trusted_inspection_and_unpack(tmp_path: Path) -> None:
    promotion, record = build_promotion(tmp_path)
    archive = tmp_path / "published.tar.gz"
    private_key = tmp_path / "publisher.key"
    public_key = tmp_path / "publisher.pub"
    outsider_key = tmp_path / "outsider.pub"
    attestation = tmp_path / "published.attestation.json"
    destination = tmp_path / "received"
    runner = CliRunner()

    packed = runner.invoke(
        app,
        ["experiment-promotion-pack", str(promotion), str(archive)],
    )
    inspected = runner.invoke(app, ["experiment-promotion-inspect", str(archive)])
    generated = runner.invoke(
        app,
        ["manifest-keygen", str(private_key), str(public_key)],
    )
    outsider = Ed25519ManifestSigner.generate()
    outsider_key.write_text(outsider.public_key_base64 + "\n", encoding="ascii")
    signed = runner.invoke(
        app,
        [
            "experiment-promotion-sign",
            str(archive),
            str(attestation),
            "--private-key",
            str(private_key),
        ],
    )
    signed_bytes = attestation.read_bytes()
    retried = runner.invoke(
        app,
        [
            "experiment-promotion-sign",
            str(archive),
            str(attestation),
            "--private-key",
            str(private_key),
        ],
    )

    ExperimentPromotionArchive.checksum_path(archive).unlink()
    trusted_inspection = runner.invoke(
        app,
        [
            "experiment-promotion-inspect",
            str(archive),
            "--attestation",
            str(attestation),
            "--public-key",
            str(outsider_key),
            "--public-key",
            str(public_key),
        ],
    )
    wrong_key = runner.invoke(
        app,
        [
            "experiment-promotion-signature-verify",
            str(archive),
            str(attestation),
            "--public-key",
            str(outsider_key),
        ],
    )
    unpacked = runner.invoke(
        app,
        [
            "experiment-promotion-unpack",
            str(archive),
            str(destination),
            "--attestation",
            str(attestation),
            "--public-key",
            str(public_key),
        ],
    )
    missing_trust = runner.invoke(
        app,
        [
            "experiment-promotion-inspect",
            str(archive),
            "--public-key",
            str(public_key),
        ],
    )

    assert packed.exit_code == 0, packed.output
    assert inspected.exit_code == 0, inspected.output
    assert generated.exit_code == 0, generated.output
    assert signed.exit_code == 0, signed.output
    assert retried.exit_code == 0, retried.output
    assert attestation.read_bytes() == signed_bytes
    if os.name != "nt":
        assert attestation.stat().st_mode & 0o777 == 0o644
    loaded = SignedExperimentPromotionArchiveAttestation.model_validate_json(signed_bytes)
    assert loaded.attestation.receipt.promotion_id == record.promotion_id
    assert trusted_inspection.exit_code == 0, trusted_inspection.output
    assert '"valid": true' in trusted_inspection.stdout
    assert wrong_key.exit_code == 1
    assert '"trusted_signer": false' in wrong_key.stdout
    assert unpacked.exit_code == 0, unpacked.output
    assert '"attestation_verification"' in unpacked.stdout
    assert ExperimentPromotionArchive.verify_directory(destination) == record
    assert missing_trust.exit_code == 2
    assert "requires --attestation" in unstyle(missing_trust.output)

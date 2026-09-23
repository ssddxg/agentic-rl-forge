from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.data import Ed25519ManifestSigner, RunArtifactArchiveAttestor
from agentic_rl_forge.storage import RunArtifactArchive, RunArtifactBundle
from test_run_artifacts import build_bundle


def test_archive_attestation_binds_receipt_and_supports_key_rotation(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _, manifest, _ = build_bundle(root)
    archive = tmp_path / "run.tar.gz"
    receipt = RunArtifactArchive(root).pack(manifest, archive)
    old_signer = Ed25519ManifestSigner.generate()
    rotated_signer = Ed25519ManifestSigner.generate()
    outsider = Ed25519ManifestSigner.generate()
    signed_at = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)

    old_attestation = RunArtifactArchiveAttestor(old_signer).sign(
        receipt,
        signed_at=signed_at,
    )
    repeated = RunArtifactArchiveAttestor(old_signer).sign(
        receipt,
        signed_at=signed_at,
    )
    rotated_attestation = RunArtifactArchiveAttestor(rotated_signer).sign(
        receipt,
        signed_at=signed_at,
    )
    trusted_keys = (old_signer.public_key_base64, rotated_signer.public_key_base64)

    old_verification = RunArtifactArchiveAttestor.verify(
        old_attestation,
        receipt,
        trusted_public_keys=trusted_keys,
    )
    rotated_verification = RunArtifactArchiveAttestor.verify(
        rotated_attestation,
        receipt,
        trusted_public_keys=trusted_keys,
    )
    wrong_key = RunArtifactArchiveAttestor.verify(
        old_attestation,
        receipt,
        trusted_public_keys=(outsider.public_key_base64,),
    )

    assert old_attestation == repeated
    assert old_signer.key_id == old_attestation.attestation.signer_key_id
    assert old_verification.valid
    assert rotated_verification.valid
    assert not wrong_key.valid
    assert wrong_key.signature_valid
    assert not wrong_key.trusted_signer

    second_root = tmp_path / "second-source"
    _, second_manifest, _ = build_bundle(second_root)
    second_receipt = RunArtifactArchive(second_root).pack(
        second_manifest,
        tmp_path / "second.tar.gz",
    )
    replayed = RunArtifactArchiveAttestor.verify(
        old_attestation,
        second_receipt,
        trusted_public_keys=trusted_keys,
    )
    assert not replayed.valid
    assert not replayed.archive_receipt_matches
    assert replayed.signature_valid


def test_archive_attestation_detects_signature_and_payload_tampering(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _, manifest, _ = build_bundle(root)
    receipt = RunArtifactArchive(root).pack(manifest, tmp_path / "run.tar.gz")
    signer = Ed25519ManifestSigner.generate()
    signed = RunArtifactArchiveAttestor(signer).sign(
        receipt,
        signed_at=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
    )

    signature_bytes = bytearray(base64.b64decode(signed.signature.signature_base64))
    signature_bytes[0] ^= 1
    changed_signature = signed.signature.model_copy(
        update={"signature_base64": base64.b64encode(signature_bytes).decode("ascii")}
    )
    signature_tampered = signed.model_copy(update={"signature": changed_signature})
    signature_result = RunArtifactArchiveAttestor.verify(
        signature_tampered,
        receipt,
        trusted_public_keys=(signer.public_key_base64,),
    )

    changed_statement = signed.attestation.model_copy(
        update={"signed_at": datetime(2026, 9, 18, 12, 1, tzinfo=timezone.utc)}
    )
    payload_tampered = signed.model_copy(update={"attestation": changed_statement})
    payload_result = RunArtifactArchiveAttestor.verify(
        payload_tampered,
        receipt,
        trusted_public_keys=(signer.public_key_base64,),
    )

    assert not signature_result.valid
    assert signature_result.payload_digest_valid
    assert not signature_result.signature_valid
    assert not payload_result.valid
    assert not payload_result.payload_digest_valid
    assert not payload_result.signature_valid


def test_archive_attestation_cli_signs_verifies_and_authorizes_unpack(tmp_path: Path) -> None:
    root = tmp_path / "source"
    bundle, manifest, _ = build_bundle(root)
    archive = tmp_path / "published.tar.gz"
    RunArtifactArchive(root).pack(manifest, archive)
    private_key = tmp_path / "publisher.key"
    public_key = tmp_path / "publisher.pub"
    outsider_public_key = tmp_path / "outsider.pub"
    attestation = tmp_path / "published.attestation.json"
    destination = tmp_path / "received"
    runner = CliRunner()

    generated = runner.invoke(
        app,
        ["manifest-keygen", str(private_key), str(public_key)],
    )
    outsider = Ed25519ManifestSigner.generate()
    outsider_public_key.write_text(outsider.public_key_base64 + "\n", encoding="ascii")
    signed = runner.invoke(
        app,
        [
            "run-artifacts-sign",
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
            "run-artifacts-sign",
            str(archive),
            str(attestation),
            "--private-key",
            str(private_key),
        ],
    )

    RunArtifactArchive.checksum_path(archive).unlink()
    verified = runner.invoke(
        app,
        [
            "run-artifacts-signature-verify",
            str(archive),
            str(attestation),
            "--public-key",
            str(outsider_public_key),
            "--public-key",
            str(public_key),
        ],
    )
    wrong_key = runner.invoke(
        app,
        [
            "run-artifacts-signature-verify",
            str(archive),
            str(attestation),
            "--public-key",
            str(outsider_public_key),
        ],
    )
    unpacked = runner.invoke(
        app,
        [
            "run-artifacts-unpack",
            str(archive),
            str(destination),
            "--attestation",
            str(attestation),
            "--public-key",
            str(public_key),
        ],
    )
    portable_manifest_path = destination / bundle.manifest_relative_path(manifest.run_id)
    portable_manifest = RunArtifactBundle(destination).load(portable_manifest_path)

    assert generated.exit_code == 0
    assert signed.exit_code == 0
    assert retried.exit_code == 0
    assert attestation.read_bytes() == signed_bytes
    if os.name != "nt":
        assert attestation.stat().st_mode & 0o777 == 0o644
    assert verified.exit_code == 0
    assert '"valid": true' in verified.stdout
    assert wrong_key.exit_code == 1
    assert '"trusted_signer": false' in wrong_key.stdout
    assert unpacked.exit_code == 0
    assert '"attestation_verification"' in unpacked.stdout
    assert RunArtifactBundle(destination).verify(portable_manifest).valid

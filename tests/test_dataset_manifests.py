import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.data import DatasetManifestBuilder, Ed25519ManifestSigner


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_dataset_manifest_is_deterministic_and_counts_duplicates(tmp_path: Path) -> None:
    train_a = tmp_path / "train-a.jsonl"
    train_b = tmp_path / "train-b.jsonl"
    validation = tmp_path / "validation.jsonl"
    write_jsonl(train_a, [{"task_id": "a"}, {"task_id": "b"}])
    write_jsonl(train_b, [{"task_id": "b"}, {"task_id": "c"}])
    write_jsonl(validation, [{"task_id": "v1"}])
    builder = DatasetManifestBuilder()

    first = builder.build_collection(
        {"train": (train_b, train_a), "validation": (validation,)},
        name="tiny",
        root=tmp_path,
    )
    second = builder.build_collection(
        {"validation": (validation,), "train": (train_a, train_b)},
        name="tiny",
        root=tmp_path,
    )

    assert first == second
    assert first.splits["train"].record_count == 4
    assert first.splits["train"].unique_id_count == 3
    assert first.splits["train"].duplicate_id_count == 1
    assert first.total_unique_id_count == 4


def test_dataset_manifest_rejects_split_overlap_and_missing_ids(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    missing = tmp_path / "missing.jsonl"
    write_jsonl(train, [{"extra": {"task": "shared"}}])
    write_jsonl(validation, [{"extra": {"task": "shared"}}])
    write_jsonl(missing, [{"question": "no ID"}])
    builder = DatasetManifestBuilder()

    with pytest.raises(ValueError, match="splits overlap"):
        builder.build_collection(
            {"train": (train,), "validation": (validation,)},
            name="overlap",
            id_field="extra.task",
        )
    with pytest.raises(ValueError, match="does not contain ID field"):
        builder.build_split((missing,), name="invalid", split="train")


def test_ed25519_manifest_signature_detects_tampering(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(train, [{"task_id": "a"}])
    manifest = DatasetManifestBuilder().build_collection(
        {"train": (train,)},
        name="signed",
    )
    signer = Ed25519ManifestSigner.generate()
    restored = Ed25519ManifestSigner.from_private_key_base64(signer.private_key_base64)

    signed = restored.sign(manifest)
    tampered = signed.model_copy(
        update={"manifest": manifest.model_copy(update={"name": "tampered"})}
    )

    assert signer.public_key_base64 == restored.public_key_base64
    assert Ed25519ManifestSigner.verify(signed)
    assert not Ed25519ManifestSigner.verify(tampered)


def test_dataset_manifest_cli_builds_signs_and_verifies(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    manifest = tmp_path / "manifest.json"
    signed = tmp_path / "manifest.signed.json"
    private_key = tmp_path / "manifest.key"
    public_key = tmp_path / "manifest.pub"
    write_jsonl(train, [{"task_id": "train-1"}])
    write_jsonl(validation, [{"task_id": "validation-1"}])
    runner = CliRunner()

    built = runner.invoke(
        app,
        [
            "dataset-manifest",
            str(manifest),
            "--name",
            "tiny",
            "--split-file",
            f"train={train}",
            "--split-file",
            f"validation={validation}",
            "--root",
            str(tmp_path),
        ],
    )
    generated = runner.invoke(
        app,
        ["manifest-keygen", str(private_key), str(public_key)],
    )
    signed_result = runner.invoke(
        app,
        [
            "manifest-sign",
            str(manifest),
            str(signed),
            "--private-key",
            str(private_key),
        ],
    )
    verified = runner.invoke(app, ["manifest-verify", str(signed)])

    assert built.exit_code == 0
    assert generated.exit_code == 0
    # POSIX exposes the chmod bits used by the CLI. Windows stores access
    # control through ACLs and does not report those bits equivalently.
    if os.name != "nt":
        assert private_key.stat().st_mode & 0o777 == 0o600
        assert public_key.stat().st_mode & 0o777 == 0o644
    assert signed_result.exit_code == 0
    assert verified.exit_code == 0
    assert '"valid": true' in verified.stdout

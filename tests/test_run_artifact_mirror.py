from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    BlobInfo,
    RunArtifactMirrorAction,
    RunArtifactMirrorObjectPlan,
    RunArtifactMirrorPlan,
    RunArtifactMirrorRecord,
)
from agentic_rl_forge.data import Ed25519ManifestSigner, RunArtifactArchiveAttestor
from agentic_rl_forge.storage import (
    BlobConflictError,
    LocalBlobStore,
    RunArtifactMirror,
    RunArtifactMirrorError,
    RunArtifactTransport,
    RunArtifactTransportError,
    S3ConditionalBlobStore,
)
from test_blob_stores import FakeS3Client
from test_run_artifact_transport import build_archive


class InstrumentedMirrorStore:
    def __init__(
        self,
        delegate: LocalBlobStore,
        *,
        fail_key_fragment: str | None = None,
        delay_seconds: float = 0,
    ) -> None:
        self.delegate = delegate
        self.fail_key_fragment = fail_key_fragment
        self.delay_seconds = delay_seconds
        self.put_keys: list[str] = []
        self._failed = False
        self._lock = threading.Lock()
        self.active_puts = 0
        self.max_active_puts = 0

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        with self._lock:
            self.put_keys.append(key)
            self.active_puts += 1
            self.max_active_puts = max(self.max_active_puts, self.active_puts)
            should_fail = (
                self.fail_key_fragment is not None
                and self.fail_key_fragment in key
                and not self._failed
            )
            if should_fail:
                self._failed = True
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        try:
            if should_fail:
                raise RuntimeError("injected mirror write failure")
            return self.delegate.put_if_absent(key, data, metadata=metadata)
        finally:
            with self._lock:
                self.active_puts -= 1

    def get(self, key: str) -> bytes:
        return self.delegate.get(key)

    def head(self, key: str) -> BlobInfo | None:
        return self.delegate.head(key)

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return self.delegate.list(prefix)

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        return self.delegate.delete_if_match(key, expected)


def publish_source(
    tmp_path: Path,
    name: str,
    *,
    signed: bool = True,
) -> tuple[Path, Any, LocalBlobStore, Ed25519ManifestSigner | None]:
    archive, receipt = build_archive(tmp_path, name)
    source = LocalBlobStore(tmp_path / f"{name}-source")
    signer = Ed25519ManifestSigner.generate() if signed else None
    attestation = (
        RunArtifactArchiveAttestor(signer).sign(
            receipt,
            signed_at=datetime(2026, 9, 18, 17, 0, tzinfo=timezone.utc),
        )
        if signer is not None
        else None
    )
    RunArtifactTransport(source, chunk_size_bytes=128).publish(
        archive,
        expected_sha256=receipt.content_digest,
        attestation=attestation,
    )
    return archive, receipt, source, signer


def test_authenticated_local_mirror_is_commit_last_and_idempotent(tmp_path: Path) -> None:
    archive, receipt, source, signer = publish_source(tmp_path, "mirror-authenticated")
    assert signer is not None
    destination_backing = LocalBlobStore(tmp_path / "mirror-destination")
    destination = InstrumentedMirrorStore(destination_backing, delay_seconds=0.002)
    mirror = RunArtifactMirror(source, destination)
    observed_at = datetime(2026, 9, 18, 18, 0, tzinfo=timezone.utc)

    plan = mirror.plan(
        receipt.archive_id,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
        now=observed_at,
    )
    record = mirror.execute(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="mirror-operator",
        reason="replicate authenticated release",
        trusted_public_keys=(signer.public_key_base64,),
        max_workers=3,
        now=observed_at,
    )
    repeated = mirror.execute(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="mirror-operator",
        reason="replicate authenticated release",
        trusted_public_keys=(signer.public_key_base64,),
        max_workers=1,
        now=observed_at,
    )

    release_puts = [key for key in destination.put_keys if key.startswith("run-releases/")]
    assert plan.attestation_verification is not None
    assert plan.attestation_verification.valid
    assert plan.copy_object_count == len(plan.objects)
    assert plan.objects[-1].role == "commit"
    assert record == repeated
    assert record.commit_created
    assert release_puts[-1] == RunArtifactTransport.commit_key(receipt.archive_id)
    assert 2 <= destination.max_active_puts <= 3
    assert destination_backing.head(mirror.record_key(record.mirror_id)) is not None

    output = tmp_path / "mirrored-authenticated.tar.gz"
    fetched = RunArtifactTransport(destination_backing).fetch(
        receipt.archive_id,
        output,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
    )
    assert fetched.attestation_verification is not None
    assert output.read_bytes() == archive.read_bytes()

    reuse_plan = RunArtifactMirror(source, destination_backing).plan(
        receipt.archive_id,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
        now=observed_at,
    )
    assert reuse_plan.copy_object_count == 0
    assert all(item.action is RunArtifactMirrorAction.REUSE for item in reuse_plan.objects)
    reuse_record = RunArtifactMirror(source, destination_backing).execute(
        reuse_plan,
        confirm_plan_id=reuse_plan.plan_id,
        operator="mirror-operator",
        reason="record already mirrored release",
        trusted_public_keys=(signer.public_key_base64,),
    )
    assert not reuse_record.commit_created
    assert reuse_record.created_object_count == 0


def test_interrupted_mirror_resumes_without_exposing_a_commit(tmp_path: Path) -> None:
    _, receipt, source, _ = publish_source(tmp_path, "mirror-resume", signed=False)
    destination_backing = LocalBlobStore(tmp_path / "resume-destination")
    failing = InstrumentedMirrorStore(
        destination_backing,
        fail_key_fragment="/chunks/00000002-",
        delay_seconds=0.001,
    )
    mirror = RunArtifactMirror(source, failing)
    plan = mirror.plan(receipt.archive_id)

    with pytest.raises(RuntimeError, match="injected mirror write failure"):
        mirror.execute(
            plan,
            confirm_plan_id=plan.plan_id,
            operator="mirror-operator",
            reason="resume interrupted mirror transfer",
            max_workers=3,
        )
    assert destination_backing.head(RunArtifactTransport.commit_key(receipt.archive_id)) is None
    staged = destination_backing.list(f"{RunArtifactTransport.release_root(receipt.archive_id)}/")
    assert staged

    tracking = InstrumentedMirrorStore(destination_backing)
    resumed = RunArtifactMirror(source, tracking).execute(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="mirror-operator",
        reason="resume interrupted mirror transfer",
        max_workers=2,
    )
    release_puts = [key for key in tracking.put_keys if key.startswith("run-releases/")]
    assert resumed.commit_created
    assert resumed.reused_object_count > 0
    assert release_puts[-1] == RunArtifactTransport.commit_key(receipt.archive_id)
    assert RunArtifactTransport(destination_backing).list_committed_archive_ids() == (
        receipt.archive_id,
    )


def test_mirror_plan_and_execution_fail_closed_on_state_changes(tmp_path: Path) -> None:
    _, receipt, source, _ = publish_source(tmp_path, "mirror-conflicts", signed=False)
    source_transport = RunArtifactTransport(source)
    _, manifest, _, _ = source_transport.inspect_committed_release(receipt.archive_id)
    destination = LocalBlobStore(tmp_path / "conflict-destination")
    destination.put_if_absent(manifest.chunks[0].key, b"X" * manifest.chunks[0].size_bytes)

    with pytest.raises(BlobConflictError, match="conflicts with source"):
        RunArtifactMirror(source, destination).plan(receipt.archive_id)

    clean_destination = LocalBlobStore(tmp_path / "changing-destination")
    extra_key = f"{RunArtifactTransport.release_root(receipt.archive_id)}/operator-note.txt"
    clean_destination.put_if_absent(extra_key, b"keep")
    mirror = RunArtifactMirror(source, clean_destination)
    plan = mirror.plan(receipt.archive_id)
    (clean_destination.root / extra_key).write_bytes(b"edit")
    with pytest.raises(RunArtifactMirrorError, match="changed after planning"):
        mirror.execute(
            plan,
            confirm_plan_id=plan.plan_id,
            operator="mirror-operator",
            reason="mirror with bound destination inventory",
        )
    assert clean_destination.head(RunArtifactTransport.commit_key(receipt.archive_id)) is None

    late_destination = LocalBlobStore(tmp_path / "late-destination")
    late_mirror = RunArtifactMirror(source, late_destination)
    late_plan = late_mirror.plan(receipt.archive_id)
    late_destination.put_if_absent(
        f"{RunArtifactTransport.release_root(receipt.archive_id)}/late.bin",
        b"late",
    )
    with pytest.raises(RunArtifactMirrorError, match="unplanned object"):
        late_mirror.execute(
            late_plan,
            confirm_plan_id=late_plan.plan_id,
            operator="mirror-operator",
            reason="reject late destination objects",
        )
    assert late_destination.head(RunArtifactTransport.commit_key(receipt.archive_id)) is None

    source_change_destination = LocalBlobStore(tmp_path / "source-change-destination")
    source_change_mirror = RunArtifactMirror(source, source_change_destination)
    source_plan = source_change_mirror.plan(receipt.archive_id)
    source_chunk = source.root / manifest.chunks[0].key
    source_chunk.write_bytes(b"Y" * manifest.chunks[0].size_bytes)
    with pytest.raises((BlobConflictError, RunArtifactMirrorError)):
        source_change_mirror.execute(
            source_plan,
            confirm_plan_id=source_plan.plan_id,
            operator="mirror-operator",
            reason="reject changed source release",
        )
    assert (
        source_change_destination.head(RunArtifactTransport.commit_key(receipt.archive_id)) is None
    )


def test_mirror_requires_the_planned_trust_policy(tmp_path: Path) -> None:
    _, receipt, source, signer = publish_source(tmp_path, "mirror-trust")
    assert signer is not None
    outsider = Ed25519ManifestSigner.generate()
    destination = LocalBlobStore(tmp_path / "trust-destination")
    mirror = RunArtifactMirror(source, destination)

    with pytest.raises(RunArtifactTransportError, match="publisher is not trusted"):
        mirror.plan(
            receipt.archive_id,
            trusted_public_keys=(outsider.public_key_base64,),
            require_attestation=True,
        )
    plan = mirror.plan(
        receipt.archive_id,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
    )
    with pytest.raises(RunArtifactTransportError, match="trusted public keys are required"):
        mirror.execute(
            plan,
            confirm_plan_id=plan.plan_id,
            operator="mirror-operator",
            reason="missing execution trust roots",
        )
    assert destination.head(RunArtifactTransport.commit_key(receipt.archive_id)) is None


def test_local_and_s3_compatible_stores_mirror_in_both_directions(tmp_path: Path) -> None:
    archive, receipt, local_source, _ = publish_source(tmp_path, "mirror-s3", signed=False)
    client = FakeS3Client()
    s3_store = S3ConditionalBlobStore(client, bucket="bucket", prefix="replica")
    local_to_s3 = RunArtifactMirror(local_source, s3_store)
    plan = local_to_s3.plan(receipt.archive_id)
    record = local_to_s3.execute(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="mirror-operator",
        reason="replicate release to S3 compatible store",
        max_workers=2,
    )
    assert record.commit_created
    assert client.last_put["Key"] == f"replica/{local_to_s3.record_key(record.mirror_id)}"
    assert RunArtifactTransport(s3_store).list_committed_archive_ids() == (receipt.archive_id,)

    second_local = LocalBlobStore(tmp_path / "second-local")
    s3_to_local = RunArtifactMirror(s3_store, second_local)
    reverse_plan = s3_to_local.plan(receipt.archive_id)
    reverse = s3_to_local.execute(
        reverse_plan,
        confirm_plan_id=reverse_plan.plan_id,
        operator="mirror-operator",
        reason="restore release from S3 compatible store",
        max_workers=2,
    )
    output = tmp_path / "restored-from-s3.tar.gz"
    RunArtifactTransport(second_local).fetch(receipt.archive_id, output)
    assert reverse.commit_created
    assert output.read_bytes() == archive.read_bytes()


def test_mirror_cli_is_preview_first_and_requires_exact_confirmation(tmp_path: Path) -> None:
    archive, receipt, source, signer = publish_source(tmp_path, "mirror-cli")
    assert signer is not None
    destination = tmp_path / "mirror-cli-destination"
    public_key = tmp_path / "mirror-cli.pub"
    public_key.write_text(signer.public_key_base64 + "\n", encoding="ascii")
    plan_path = tmp_path / "mirror-plan.json"
    record_path = tmp_path / "mirror-record.json"
    runner = CliRunner()

    planned = runner.invoke(
        app,
        [
            "run-artifacts-mirror-plan",
            receipt.archive_id,
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination),
            "--public-key",
            str(public_key),
            "--require-attestation",
            "--output",
            str(plan_path),
        ],
    )
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    validated = runner.invoke(app, ["run-artifacts-mirror", str(plan_path)])
    assert planned.exit_code == 0
    assert validated.exit_code == 0
    assert not LocalBlobStore(destination).list("run-releases")

    missing_confirmation = runner.invoke(
        app,
        [
            "run-artifacts-mirror",
            str(plan_path),
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination),
            "--execute",
        ],
    )
    assert missing_confirmation.exit_code == 2
    assert not LocalBlobStore(destination).list("run-releases")

    executed = runner.invoke(
        app,
        [
            "run-artifacts-mirror",
            str(plan_path),
            "--source-store-root",
            str(source.root),
            "--destination-store-root",
            str(destination),
            "--public-key",
            str(public_key),
            "--execute",
            "--confirm-plan-id",
            plan_payload["plan_id"],
            "--operator",
            "cli-mirror-operator",
            "--reason",
            "reviewed authenticated mirror plan",
            "--workers",
            "2",
            "--output",
            str(record_path),
        ],
    )
    output = tmp_path / "mirror-cli-fetched.tar.gz"
    RunArtifactTransport(LocalBlobStore(destination)).fetch(
        receipt.archive_id,
        output,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
    )
    assert executed.exit_code == 0
    assert record_path.is_file()
    assert output.read_bytes() == archive.read_bytes()


def test_mirror_contracts_reject_inconsistent_plans_and_evidence(tmp_path: Path) -> None:
    _, receipt, source, _ = publish_source(tmp_path, "mirror-contracts", signed=False)
    destination = LocalBlobStore(tmp_path / "mirror-contract-destination")
    mirror = RunArtifactMirror(source, destination)
    plan = mirror.plan(receipt.archive_id)
    first_payload = plan.objects[0].model_dump(mode="python")

    with pytest.raises(ValueError, match="must be copied"):
        RunArtifactMirrorObjectPlan.model_validate(
            {**first_payload, "action": RunArtifactMirrorAction.REUSE}
        )
    plan_payload = plan.model_dump(mode="python")
    for update, message in (
        ({"copy_object_count": plan.copy_object_count + 1}, "object counts"),
        ({"copy_bytes": plan.copy_bytes + 1}, "copy bytes"),
        ({"plan_id": "run_mirror_plan_" + "0" * 24}, "plan ID"),
        ({"objects": tuple(reversed(plan.objects))}, "commit object|object graph"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactMirrorPlan.model_validate({**plan_payload, **update})

    record = mirror.execute(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="mirror-operator",
        reason="validate mirror evidence contracts",
    )
    record_payload = record.model_dump(mode="python")
    with pytest.raises(ValueError, match="plan order"):
        RunArtifactMirrorRecord.model_validate(
            {**record_payload, "created_keys": tuple(reversed(record.created_keys))}
        )
    with pytest.raises(ValueError, match="do not partition"):
        RunArtifactMirrorRecord.model_validate(
            {**record_payload, "reused_keys": (record.created_keys[0],)}
        )
    with pytest.raises(ValueError, match="created bytes"):
        RunArtifactMirrorRecord.model_validate(
            {**record_payload, "created_bytes": record.created_bytes + 1}
        )
    with pytest.raises(ValueError, match="record ID"):
        RunArtifactMirrorRecord.model_validate(
            {**record_payload, "mirror_id": "run_mirror_" + "0" * 24}
        )

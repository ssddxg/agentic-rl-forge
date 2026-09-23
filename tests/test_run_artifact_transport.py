from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.contracts import (
    BlobInfo,
    RunArtifactArchiveReceipt,
    RunArtifactGcBatchIntent,
    RunArtifactGcBatchRecord,
    RunArtifactGcPlan,
    RunArtifactGcPreview,
    RunArtifactGcReason,
    RunArtifactGcRecord,
    RunArtifactGcTombstone,
    RunArtifactReleaseState,
    RunArtifactStoreInventory,
    RunArtifactTransportCommit,
    RunArtifactTransportManifest,
    RunArtifactTransportObjectRef,
    RunArtifactTransportStatus,
)
from agentic_rl_forge.data import Ed25519ManifestSigner, RunArtifactArchiveAttestor
from agentic_rl_forge.storage import (
    BlobConflictError,
    LocalBlobStore,
    RunArtifactArchive,
    RunArtifactTransport,
    RunArtifactTransportError,
    S3ConditionalBlobStore,
)
from test_blob_stores import FakeS3Client
from test_run_artifacts import build_bundle


class InstrumentedStore:
    def __init__(
        self,
        delegate: LocalBlobStore,
        *,
        fail_put: Callable[[str], bool] | None = None,
        fail_get: Callable[[str], bool] | None = None,
    ) -> None:
        self.delegate = delegate
        self.fail_put = fail_put
        self.fail_get = fail_get
        self.put_keys: list[str] = []
        self.get_keys: list[str] = []
        self._put_failed = False
        self._get_failed = False

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        self.put_keys.append(key)
        if self.fail_put is not None and self.fail_put(key) and not self._put_failed:
            self._put_failed = True
            raise RuntimeError("injected upload failure")
        return self.delegate.put_if_absent(key, data, metadata=metadata)

    def get(self, key: str) -> bytes:
        self.get_keys.append(key)
        if self.fail_get is not None and self.fail_get(key) and not self._get_failed:
            self._get_failed = True
            raise RuntimeError("injected download failure")
        return self.delegate.get(key)

    def head(self, key: str) -> Any:
        return self.delegate.head(key)

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return self.delegate.list(prefix)

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        return self.delegate.delete_if_match(key, expected)


class PostWriteFaultStore:
    def __init__(self, root: Path, mode: str) -> None:
        self.delegate = LocalBlobStore(root)
        self.mode = mode

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        return self.delegate.put_if_absent(key, data, metadata=metadata)

    def get(self, key: str) -> bytes:
        payload = self.delegate.get(key)
        return b"wrong" if self.mode == "content" else payload

    def head(self, key: str) -> BlobInfo | None:
        info = self.delegate.head(key)
        if self.mode == "missing-head":
            return None
        if info is not None and self.mode == "digest":
            return info.model_copy(update={"content_sha256": "0" * 64})
        return info

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return self.delegate.list(prefix)

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        return self.delegate.delete_if_match(key, expected)


class ConcurrencyStore:
    def __init__(self, delegate: LocalBlobStore) -> None:
        self.delegate = delegate
        self._lock = threading.Lock()
        self.active_chunk_operations = 0
        self.max_chunk_operations = 0

    def reset(self) -> None:
        with self._lock:
            self.active_chunk_operations = 0
            self.max_chunk_operations = 0

    def _enter(self, key: str) -> bool:
        if "/chunks/" not in key:
            return False
        with self._lock:
            self.active_chunk_operations += 1
            self.max_chunk_operations = max(
                self.max_chunk_operations,
                self.active_chunk_operations,
            )
        time.sleep(0.005)
        return True

    def _leave(self, tracked: bool) -> None:
        if tracked:
            with self._lock:
                self.active_chunk_operations -= 1

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        tracked = self._enter(key)
        try:
            return self.delegate.put_if_absent(key, data, metadata=metadata)
        finally:
            self._leave(tracked)

    def get(self, key: str) -> bytes:
        tracked = self._enter(key)
        try:
            return self.delegate.get(key)
        finally:
            self._leave(tracked)

    def head(self, key: str) -> BlobInfo | None:
        return self.delegate.head(key)

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return self.delegate.list(prefix)

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        return self.delegate.delete_if_match(key, expected)


class DeleteFaultStore:
    def __init__(self, delegate: LocalBlobStore, *, fail_after: int) -> None:
        self.delegate = delegate
        self.fail_after = fail_after
        self.deleted = 0
        self.failed = False

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        return self.delegate.put_if_absent(key, data, metadata=metadata)

    def get(self, key: str) -> bytes:
        return self.delegate.get(key)

    def head(self, key: str) -> BlobInfo | None:
        return self.delegate.head(key)

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return self.delegate.list(prefix)

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        if self.deleted >= self.fail_after and not self.failed:
            self.failed = True
            raise RuntimeError("injected delete failure")
        deleted = self.delegate.delete_if_match(key, expected)
        self.deleted += int(deleted)
        return deleted


class IdentityMaskingStore:
    def __init__(
        self,
        delegate: LocalBlobStore,
        *,
        hide_digest: bool = False,
        hide_etag: bool = False,
        hide_last_modified: bool = False,
        size_delta: int = 0,
    ) -> None:
        self.delegate = delegate
        self.hide_digest = hide_digest
        self.hide_etag = hide_etag
        self.hide_last_modified = hide_last_modified
        self.size_delta = size_delta

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        return self.delegate.put_if_absent(key, data, metadata=metadata)

    def get(self, key: str) -> bytes:
        return self.delegate.get(key)

    def head(self, key: str) -> BlobInfo | None:
        info = self.delegate.head(key)
        if info is None:
            return None
        return info.model_copy(
            update={
                "size_bytes": info.size_bytes + self.size_delta,
                "etag": None if self.hide_etag else info.etag,
                "content_sha256": None if self.hide_digest else info.content_sha256,
                "last_modified": None if self.hide_last_modified else info.last_modified,
            }
        )

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return self.delegate.list(prefix)

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        return self.delegate.delete_if_match(key, expected)


class FalseDeleteStore(IdentityMaskingStore):
    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        self.delegate.delete_if_match(key, expected)
        return False


class BatchDeleteConcurrencyStore(IdentityMaskingStore):
    def __init__(self, delegate: LocalBlobStore) -> None:
        super().__init__(delegate)
        self._lock = threading.Lock()
        self._delete_barrier = threading.Barrier(2)
        self.active_deletes = 0
        self.max_active_deletes = 0

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        with self._lock:
            self.active_deletes += 1
            self.max_active_deletes = max(self.max_active_deletes, self.active_deletes)
        try:
            self._delete_barrier.wait(timeout=60)
            return self.delegate.delete_if_match(key, expected)
        finally:
            with self._lock:
                self.active_deletes -= 1


class SelectiveDeleteFaultStore(IdentityMaskingStore):
    def __init__(self, delegate: LocalBlobStore, archive_id: str) -> None:
        super().__init__(delegate)
        self.archive_id = archive_id
        self.failed = False
        self._lock = threading.Lock()

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        with self._lock:
            should_fail = self.archive_id in key and not self.failed
            if should_fail:
                self.failed = True
        if should_fail:
            raise RuntimeError("injected batch delete failure")
        return self.delegate.delete_if_match(key, expected)


def build_archive(
    tmp_path: Path,
    name: str = "source",
) -> tuple[Path, RunArtifactArchiveReceipt]:
    root = tmp_path / name
    _, manifest, _ = build_bundle(root)
    archive = tmp_path / f"{name}.tar.gz"
    receipt = RunArtifactArchive(root).pack(manifest, archive)
    return archive, receipt


def test_local_transport_publishes_commit_last_and_fetches_trusted_release(
    tmp_path: Path,
) -> None:
    archive, receipt = build_archive(tmp_path)
    signer = Ed25519ManifestSigner.generate()
    attestation = RunArtifactArchiveAttestor(signer).sign(
        receipt,
        signed_at=datetime(2026, 9, 18, 15, 0, tzinfo=timezone.utc),
    )
    backing = LocalBlobStore(tmp_path / "remote")
    instrumented = InstrumentedStore(backing)
    transport = RunArtifactTransport(instrumented, chunk_size_bytes=257)

    published = transport.publish(
        archive,
        expected_sha256=receipt.content_digest,
        attestation=attestation,
    )
    retried = transport.publish(
        archive,
        expected_sha256=receipt.content_digest,
        attestation=attestation,
    )

    assert published.commit_created
    assert published.created_chunk_count > 1
    assert published.created_object_count == 4
    assert instrumented.put_keys[-1].endswith("/commit.json")
    assert not retried.commit_created
    assert retried.created_chunk_count == 0
    assert retried.reused_chunk_count == published.created_chunk_count
    assert retried.reused_object_count == 4
    assert transport.list_committed_archive_ids() == (receipt.archive_id,)

    output = tmp_path / "received" / "run.tar.gz"
    fetched = transport.fetch(
        receipt.archive_id,
        output,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
    )
    repeated_fetch = transport.fetch(
        receipt.archive_id,
        output,
        trusted_public_keys=(signer.public_key_base64,),
        require_attestation=True,
    )

    assert output.read_bytes() == archive.read_bytes()
    assert (
        RunArtifactArchive.read_checksum(RunArtifactArchive.checksum_path(output))
        == receipt.content_digest
    )
    assert RunArtifactTransport.fetched_attestation_path(output).read_bytes() == (
        attestation.canonical_bytes() + b"\n"
    )
    assert fetched.receipt == receipt
    assert fetched.downloaded_chunk_count == published.created_chunk_count
    assert fetched.attestation_verification is not None
    assert fetched.attestation_verification.valid
    assert repeated_fetch.downloaded_chunk_count == 0
    assert repeated_fetch.reused_local_chunk_count == published.created_chunk_count


def test_interrupted_upload_is_invisible_and_resume_reuses_chunks(tmp_path: Path) -> None:
    archive, receipt = build_archive(tmp_path)
    backing = LocalBlobStore(tmp_path / "remote")
    failing = InstrumentedStore(
        backing,
        fail_put=lambda key: "/chunks/00000002-" in key,
    )
    transport = RunArtifactTransport(failing, chunk_size_bytes=128)

    with pytest.raises(RuntimeError, match="injected upload failure"):
        transport.publish(archive, expected_sha256=receipt.content_digest)

    assert transport.list_committed_archive_ids() == ()
    assert backing.head(transport.commit_key(receipt.archive_id)) is None
    assert len(backing.list(f"{transport.release_root(receipt.archive_id)}/chunks")) == 2

    resumed = RunArtifactTransport(backing, chunk_size_bytes=128).publish(
        archive,
        expected_sha256=receipt.content_digest,
    )

    assert resumed.commit_created
    assert resumed.reused_chunk_count == 2
    assert resumed.created_chunk_count > 0
    assert RunArtifactTransport(backing).list_committed_archive_ids() == (receipt.archive_id,)


def test_commit_failure_never_exposes_fully_staged_release(tmp_path: Path) -> None:
    archive, receipt = build_archive(tmp_path)
    backing = LocalBlobStore(tmp_path / "remote")
    failing = InstrumentedStore(
        backing,
        fail_put=lambda key: key.endswith("/commit.json"),
    )
    transport = RunArtifactTransport(failing, chunk_size_bytes=160)

    with pytest.raises(RuntimeError, match="injected upload failure"):
        transport.publish(archive, expected_sha256=receipt.content_digest)

    staged_chunk_count = len(backing.list(f"{transport.release_root(receipt.archive_id)}/chunks"))
    assert staged_chunk_count > 1
    assert transport.list_committed_archive_ids() == ()
    assert backing.head(transport.commit_key(receipt.archive_id)) is None

    resumed = RunArtifactTransport(backing, chunk_size_bytes=160).publish(
        archive,
        expected_sha256=receipt.content_digest,
    )

    assert resumed.created_chunk_count == 0
    assert resumed.reused_chunk_count == staged_chunk_count
    assert resumed.created_object_count == 1
    assert resumed.reused_object_count == 2
    assert resumed.commit_created


def test_interrupted_download_resumes_verified_prefix_without_visible_output(
    tmp_path: Path,
) -> None:
    archive, receipt = build_archive(tmp_path)
    backing = LocalBlobStore(tmp_path / "remote")
    publisher = RunArtifactTransport(backing, chunk_size_bytes=192)
    published = publisher.publish(archive, expected_sha256=receipt.content_digest)
    output = tmp_path / "downloads" / "run.tar.gz"
    failing = InstrumentedStore(
        backing,
        fail_get=lambda key: "/chunks/00000002-" in key,
    )

    with pytest.raises(RuntimeError, match="injected download failure"):
        RunArtifactTransport(failing).fetch(receipt.archive_id, output)

    partial = RunArtifactTransport.partial_path(output, receipt.archive_id)
    assert partial.is_file()
    assert not output.exists()
    assert not RunArtifactArchive.checksum_path(output).exists()
    assert partial.stat().st_size == 2 * 192

    resumed = RunArtifactTransport(backing).fetch(receipt.archive_id, output)

    assert output.read_bytes() == archive.read_bytes()
    assert resumed.reused_local_chunk_count == 2
    assert resumed.downloaded_chunk_count == published.created_chunk_count - 2
    assert resumed.downloaded_bytes == receipt.size_bytes - 2 * 192
    assert not partial.exists()


def test_fetch_rejects_untrusted_or_corrupt_releases_without_publishing(
    tmp_path: Path,
) -> None:
    archive, receipt = build_archive(tmp_path)
    signer = Ed25519ManifestSigner.generate()
    outsider = Ed25519ManifestSigner.generate()
    attestation = RunArtifactArchiveAttestor(signer).sign(receipt)
    backing = LocalBlobStore(tmp_path / "remote")
    transport = RunArtifactTransport(backing, chunk_size_bytes=256)
    published = transport.publish(
        archive,
        expected_sha256=receipt.content_digest,
        attestation=attestation,
    )
    wrong_key_output = tmp_path / "wrong-key.tar.gz"

    with pytest.raises(RunArtifactTransportError, match="publisher is not trusted"):
        transport.fetch(
            receipt.archive_id,
            wrong_key_output,
            trusted_public_keys=(outsider.public_key_base64,),
            require_attestation=True,
        )
    assert not wrong_key_output.exists()

    manifest_payload = backing.get(published.commit.manifest.key)
    manifest = RunArtifactTransportManifest.model_validate_json(manifest_payload)
    corrupt_chunk = backing.root / manifest.chunks[0].key
    corrupt_chunk.write_bytes(b"X" * manifest.chunks[0].size_bytes)
    corrupt_output = tmp_path / "corrupt.tar.gz"

    with pytest.raises(RunArtifactTransportError, match="failed content verification"):
        transport.fetch(receipt.archive_id, corrupt_output)
    assert not corrupt_output.exists()
    assert not RunArtifactArchive.checksum_path(corrupt_output).exists()

    unsigned_archive, unsigned_receipt = build_archive(tmp_path, "unsigned")
    unsigned_store = LocalBlobStore(tmp_path / "unsigned-remote")
    unsigned_transport = RunArtifactTransport(unsigned_store)
    unsigned_transport.publish(
        unsigned_archive,
        expected_sha256=unsigned_receipt.content_digest,
    )
    with pytest.raises(RunArtifactTransportError, match="does not contain an attestation"):
        unsigned_transport.fetch(
            unsigned_receipt.archive_id,
            tmp_path / "unsigned-fetch.tar.gz",
            trusted_public_keys=(signer.public_key_base64,),
            require_attestation=True,
        )


def test_transport_rejects_symlink_partial_and_invalid_commit_metadata(
    tmp_path: Path,
) -> None:
    archive, receipt = build_archive(tmp_path)
    backing = LocalBlobStore(tmp_path / "remote")
    transport = RunArtifactTransport(backing, chunk_size_bytes=256)
    transport.publish(archive, expected_sha256=receipt.content_digest)
    output = tmp_path / "download" / "run.tar.gz"
    output.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    partial = transport.partial_path(output, receipt.archive_id)
    try:
        partial.symlink_to(victim)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("creating symlinks requires Windows Developer Mode or elevation")
        raise

    with pytest.raises(RunArtifactTransportError, match="regular file"):
        transport.fetch(receipt.archive_id, output)

    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert not output.exists()

    corrupt_store = LocalBlobStore(tmp_path / "corrupt-remote")
    corrupt_transport = RunArtifactTransport(corrupt_store)
    corrupt_store.put_if_absent(
        corrupt_transport.commit_key(receipt.archive_id),
        b"{}\n",
    )
    with pytest.raises(RunArtifactTransportError, match="metadata contract is invalid"):
        corrupt_transport.list_committed_archive_ids()


def test_s3_compatible_transport_streams_chunks_with_conditional_creation(
    tmp_path: Path,
) -> None:
    archive, receipt = build_archive(tmp_path)
    client = FakeS3Client()
    store = S3ConditionalBlobStore(client, bucket="bucket", prefix="project")
    transport = RunArtifactTransport(store, chunk_size_bytes=211)

    published = transport.publish(archive, expected_sha256=receipt.content_digest)
    output = tmp_path / "s3-downloaded.tar.gz"
    fetched = transport.fetch(receipt.archive_id, output)

    assert published.created_chunk_count > 1
    assert fetched.downloaded_chunk_count == published.created_chunk_count
    assert output.read_bytes() == archive.read_bytes()
    assert client.last_put["IfNoneMatch"] == "*"
    assert client.last_put["Key"] == f"project/{transport.commit_key(receipt.archive_id)}"
    assert client.last_put["Metadata"]["kind"] == "run-archive-release-commit"


def test_s3_compatible_gc_uses_identity_checked_deletes() -> None:
    archive_id = "run_archive_" + "9" * 24
    client = FakeS3Client()
    store = S3ConditionalBlobStore(client, bucket="bucket", prefix="project")
    transport = RunArtifactTransport(store)
    store.put_if_absent(f"{transport.release_root(archive_id)}/one.bin", b"one")
    store.put_if_absent(f"{transport.release_root(archive_id)}/two.bin", b"two")
    action_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
    preview = transport.preview_garbage_collection(
        archive_id,
        min_age_seconds=60,
        now=action_time,
    )

    record = transport.garbage_collect(
        archive_id,
        min_age_seconds=60,
        confirm_state_digest=preview.state_digest,
        operator="s3-operator",
        reason="expired staged S3 publication",
        now=action_time,
    )

    assert record.deleted_object_count == 2
    assert client.last_delete["IfMatch"].startswith("etag-")
    assert store.list(f"{transport.release_root(archive_id)}/") == (
        transport.commit_key(archive_id),
    )
    assert store.head(transport.gc_record_key(record.gc_id)) is not None


def test_transport_cli_publishes_lists_and_fetches_authenticated_release(
    tmp_path: Path,
) -> None:
    archive, receipt = build_archive(tmp_path)
    signer = Ed25519ManifestSigner.generate()
    attestation = RunArtifactArchiveAttestor(signer).sign(receipt)
    attestation_path = tmp_path / "run.attestation.json"
    public_key = tmp_path / "release.pub"
    store_root = tmp_path / "release-store"
    output = tmp_path / "fetched" / "run.tar.gz"
    attestation_path.write_bytes(attestation.canonical_bytes() + b"\n")
    public_key.write_text(signer.public_key_base64 + "\n", encoding="ascii")
    runner = CliRunner()

    missing_store = runner.invoke(app, ["run-artifacts-publish", str(archive)])
    published = runner.invoke(
        app,
        [
            "run-artifacts-publish",
            str(archive),
            "--store-root",
            str(store_root),
            "--attestation",
            str(attestation_path),
            "--chunk-size-mib",
            "1",
            "--workers",
            "2",
        ],
    )
    listed = runner.invoke(
        app,
        ["run-artifacts-list", "--store-root", str(store_root)],
    )
    fetched = runner.invoke(
        app,
        [
            "run-artifacts-fetch",
            receipt.archive_id,
            str(output),
            "--store-root",
            str(store_root),
            "--public-key",
            str(public_key),
            "--require-attestation",
            "--workers",
            "2",
        ],
    )

    assert missing_store.exit_code == 2
    assert "provide exactly one" in missing_store.stderr
    assert published.exit_code == 0
    assert '"commit_created": true' in published.stdout
    assert listed.exit_code == 0
    assert receipt.archive_id in listed.stdout
    assert fetched.exit_code == 0
    assert '"attestation_verification"' in fetched.stdout
    assert output.read_bytes() == archive.read_bytes()


def test_transport_cli_status_and_preview_confirmed_gc(tmp_path: Path) -> None:
    archive, receipt = build_archive(tmp_path, "gc-cli")
    store_root = tmp_path / "gc-cli-store"
    backing = LocalBlobStore(store_root)
    failing = InstrumentedStore(
        backing,
        fail_put=lambda key: key.endswith("/commit.json"),
    )
    with pytest.raises(RuntimeError, match="injected upload failure"):
        RunArtifactTransport(failing, chunk_size_bytes=128).publish(
            archive,
            expected_sha256=receipt.content_digest,
        )
    release_prefix = f"{RunArtifactTransport.release_root(receipt.archive_id)}/"
    old_timestamp = time.time() - 3_600
    for key in backing.list(release_prefix):
        os.utime(backing.root / key, (old_timestamp, old_timestamp))

    runner = CliRunner()
    status = runner.invoke(
        app,
        [
            "run-artifacts-status",
            receipt.archive_id,
            "--store-root",
            str(store_root),
        ],
    )
    before_keys = backing.list(release_prefix)
    preview_output = tmp_path / "gc-preview.json"
    preview = runner.invoke(
        app,
        [
            "run-artifacts-gc",
            receipt.archive_id,
            "--store-root",
            str(store_root),
            "--min-age-seconds",
            "1",
            "--output",
            str(preview_output),
        ],
    )

    assert status.exit_code == 0
    assert '"state": "staged"' in status.stdout
    assert preview.exit_code == 0
    preview_payload = json.loads(preview_output.read_text(encoding="utf-8"))
    assert preview_payload["eligible"] is True
    assert backing.list(release_prefix) == before_keys

    missing_confirmation = runner.invoke(
        app,
        [
            "run-artifacts-gc",
            receipt.archive_id,
            "--store-root",
            str(store_root),
            "--min-age-seconds",
            "1",
            "--execute",
        ],
    )
    assert missing_confirmation.exit_code == 2
    assert backing.head(RunArtifactTransport.commit_key(receipt.archive_id)) is None

    evidence = tmp_path / "gc-record.json"
    executed = runner.invoke(
        app,
        [
            "run-artifacts-gc",
            receipt.archive_id,
            "--store-root",
            str(store_root),
            "--min-age-seconds",
            "1",
            "--execute",
            "--confirm-state-digest",
            preview_payload["state_digest"],
            "--operator",
            "cli-operator",
            "--reason",
            "abandoned staged CLI release",
            "--output",
            str(evidence),
        ],
    )
    final_status = runner.invoke(
        app,
        [
            "run-artifacts-status",
            receipt.archive_id,
            "--store-root",
            str(store_root),
        ],
    )
    assert executed.exit_code == 0
    assert '"gc_id"' in executed.stdout
    assert evidence.is_file()
    assert final_status.exit_code == 0
    assert '"state": "garbage_collected"' in final_status.stdout


def test_transport_cli_inventory_plan_and_batch_gc(tmp_path: Path) -> None:
    store_root = tmp_path / "batch-cli-store"
    backing = LocalBlobStore(store_root)
    archive_ids = ("run_archive_" + "c3" * 12, "run_archive_" + "d4" * 12)
    old_timestamp = time.time() - 3_600
    for archive_id in archive_ids:
        key = f"{RunArtifactTransport.release_root(archive_id)}/staged.bin"
        backing.put_if_absent(key, archive_id.encode("ascii"))
        os.utime(backing.root / key, (old_timestamp, old_timestamp))
    runner = CliRunner()
    inventory_path = tmp_path / "inventory.json"
    plan_path = tmp_path / "gc-plan.json"
    inventory = runner.invoke(
        app,
        [
            "run-artifacts-inventory",
            "--store-root",
            str(store_root),
            "--output",
            str(inventory_path),
        ],
    )
    plan_result = runner.invoke(
        app,
        [
            "run-artifacts-gc-plan",
            "--store-root",
            str(store_root),
            "--min-age-seconds",
            "1",
            "--output",
            str(plan_path),
        ],
    )
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))

    preview_batch = runner.invoke(app, ["run-artifacts-gc-batch", str(plan_path)])
    assert inventory.exit_code == 0
    assert inventory_path.is_file()
    assert plan_result.exit_code == 0
    assert plan_payload["candidate_count"] == 2
    assert preview_batch.exit_code == 0
    assert all(
        backing.head(f"{RunArtifactTransport.release_root(item)}/staged.bin") is not None
        for item in archive_ids
    )

    missing = runner.invoke(
        app,
        [
            "run-artifacts-gc-batch",
            str(plan_path),
            "--store-root",
            str(store_root),
            "--execute",
        ],
    )
    assert missing.exit_code == 2

    record_path = tmp_path / "gc-batch-record.json"
    executed = runner.invoke(
        app,
        [
            "run-artifacts-gc-batch",
            str(plan_path),
            "--store-root",
            str(store_root),
            "--execute",
            "--confirm-plan-id",
            plan_payload["plan_id"],
            "--operator",
            "cli-batch-operator",
            "--reason",
            "reviewed expired CLI batch",
            "--workers",
            "2",
            "--output",
            str(record_path),
        ],
    )
    assert executed.exit_code == 0
    assert record_path.is_file()
    record_payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert record_payload["candidate_count"] == 2
    assert all(
        RunArtifactTransport(backing).status(item).state
        is RunArtifactReleaseState.GARBAGE_COLLECTED
        for item in archive_ids
    )


def test_transport_contracts_reject_noncanonical_release_graphs(tmp_path: Path) -> None:
    archive, receipt = build_archive(tmp_path)
    signer = Ed25519ManifestSigner.generate()
    attestation = RunArtifactArchiveAttestor(signer).sign(receipt)
    backing = LocalBlobStore(tmp_path / "remote")
    transport = RunArtifactTransport(backing, chunk_size_bytes=128)
    published = transport.publish(
        archive,
        expected_sha256=receipt.content_digest,
        attestation=attestation,
    )
    manifest = RunArtifactTransportManifest.model_validate_json(
        backing.get(published.commit.manifest.key)
    )

    with pytest.raises(ValueError, match="safe relative path"):
        RunArtifactTransportObjectRef(
            key="../escape",
            media_type="application/octet-stream",
            size_bytes=1,
            content_digest="0" * 64,
        )

    bad_indexes = list(manifest.chunks)
    bad_indexes[0] = bad_indexes[0].model_copy(update={"index": 1})
    bad_size = list(manifest.chunks)
    bad_size[0] = bad_size[0].model_copy(update={"size_bytes": bad_size[0].size_bytes + 1})
    bad_nonfinal = list(manifest.chunks)
    bad_nonfinal[0] = bad_nonfinal[0].model_copy(
        update={"size_bytes": bad_nonfinal[0].size_bytes - 1}
    )
    bad_nonfinal[-1] = bad_nonfinal[-1].model_copy(
        update={"size_bytes": bad_nonfinal[-1].size_bytes + 1}
    )
    bad_chunk_key = list(manifest.chunks)
    bad_chunk_key[0] = bad_chunk_key[0].model_copy(update={"key": "chunks/wrong.part"})
    cases = (
        ({"chunks": tuple(bad_indexes)}, "indexes must be contiguous"),
        ({"chunks": tuple(bad_size)}, "chunks do not match archive size"),
        ({"chunks": tuple(bad_nonfinal)}, "non-final transport chunks"),
        ({"chunks": tuple(bad_chunk_key)}, "chunk metadata is not canonical"),
        (
            {"checksum": manifest.checksum.model_copy(update={"media_type": "bad/type"})},
            "checksum metadata is not canonical",
        ),
        (
            {
                "attestation": manifest.attestation.model_copy(update={"key": "wrong.json"})
                if manifest.attestation is not None
                else None
            },
            "attestation metadata is not canonical",
        ),
        ({"transport_id": "run_transport_" + "0" * 24}, "ID does not match"),
    )
    manifest_payload = manifest.model_dump(mode="python")
    for update, message in cases:
        with pytest.raises(ValueError, match=message):
            RunArtifactTransportManifest.model_validate({**manifest_payload, **update})

    commit_payload = published.commit.model_dump(mode="python")
    with pytest.raises(ValueError, match="manifest reference is not canonical"):
        RunArtifactTransportCommit.model_validate(
            {
                **commit_payload,
                "manifest": published.commit.manifest.model_copy(update={"key": "wrong.json"}),
            }
        )
    with pytest.raises(ValueError, match="commit ID does not match"):
        RunArtifactTransportCommit.model_validate(
            {**commit_payload, "commit_id": "run_release_" + "0" * 24}
        )


def test_transport_validates_post_write_state_and_resume_prefix(tmp_path: Path) -> None:
    for mode, message in (
        ("content", "failed post-write verification"),
        ("missing-head", "invalid metadata"),
        ("digest", "invalid digest metadata"),
    ):
        transport = RunArtifactTransport(PostWriteFaultStore(tmp_path / mode, mode))
        with pytest.raises(RunArtifactTransportError, match=message):
            transport._put_verified("object.bin", b"payload", metadata={})

    archive, receipt = build_archive(tmp_path, "resume-validation")
    backing = LocalBlobStore(tmp_path / "resume-remote")
    publisher = RunArtifactTransport(backing, chunk_size_bytes=192)
    published = publisher.publish(archive, expected_sha256=receipt.content_digest)
    output = tmp_path / "resume" / "run.tar.gz"
    failing = InstrumentedStore(
        backing,
        fail_get=lambda key: "/chunks/00000002-" in key,
    )
    with pytest.raises(RuntimeError, match="injected download failure"):
        RunArtifactTransport(failing).fetch(receipt.archive_id, output)
    partial = RunArtifactTransport.partial_path(output, receipt.archive_id)
    payload = bytearray(partial.read_bytes())
    payload[0] ^= 1
    partial.write_bytes(payload)

    resumed = RunArtifactTransport(backing).fetch(receipt.archive_id, output)

    assert resumed.reused_local_chunk_count == 0
    assert resumed.downloaded_chunk_count == published.created_chunk_count
    assert output.read_bytes() == archive.read_bytes()


def test_transport_reports_missing_conflicting_and_unauthenticated_states(
    tmp_path: Path,
) -> None:
    empty = RunArtifactTransport(LocalBlobStore(tmp_path / "empty"))
    with pytest.raises(RunArtifactTransportError, match="is not committed"):
        empty.fetch("run_archive_" + "0" * 24, tmp_path / "missing.tar.gz")
    with pytest.raises(ValueError, match="chunk size"):
        RunArtifactTransport(LocalBlobStore(tmp_path / "invalid"), chunk_size_bytes=0)

    archive, receipt = build_archive(tmp_path, "signed-conflict")
    signer = Ed25519ManifestSigner.generate()
    attestation = RunArtifactArchiveAttestor(signer).sign(receipt)
    backing = LocalBlobStore(tmp_path / "signed-remote")
    transport = RunArtifactTransport(backing, chunk_size_bytes=256)
    transport.publish(
        archive,
        expected_sha256=receipt.content_digest,
        attestation=attestation,
    )
    with pytest.raises(BlobConflictError, match="different attestation"):
        transport.publish(archive, expected_sha256=receipt.content_digest)
    with pytest.raises(RunArtifactTransportError, match="trusted public keys are required"):
        transport.fetch(
            receipt.archive_id,
            tmp_path / "no-trust.tar.gz",
            require_attestation=True,
        )

    integrity_only = transport.fetch(
        receipt.archive_id,
        tmp_path / "integrity-only.tar.gz",
    )
    assert integrity_only.attestation_verification is None

    conflict_output = tmp_path / "conflict-output.tar.gz"
    conflict_output.write_bytes(b"wrong")
    with pytest.raises(BlobConflictError, match="different content"):
        transport.fetch(receipt.archive_id, conflict_output)


def test_parallel_transport_is_bounded_ordered_and_resumable(tmp_path: Path) -> None:
    archive, receipt = build_archive(tmp_path, "parallel")
    backing = LocalBlobStore(tmp_path / "parallel-remote")
    concurrent = ConcurrencyStore(backing)
    transport = RunArtifactTransport(
        concurrent,
        chunk_size_bytes=64,
        max_workers=3,
    )

    published = transport.publish(archive, expected_sha256=receipt.content_digest)

    assert published.created_chunk_count > 3
    assert 2 <= concurrent.max_chunk_operations <= 3
    concurrent.reset()
    output = tmp_path / "parallel-download.tar.gz"
    fetched = transport.fetch(receipt.archive_id, output)
    assert fetched.downloaded_chunk_count == published.created_chunk_count
    assert output.read_bytes() == archive.read_bytes()
    assert 2 <= concurrent.max_chunk_operations <= 3

    with pytest.raises(ValueError, match="worker count"):
        RunArtifactTransport(backing, max_workers=0)
    with pytest.raises(ValueError, match="worker count"):
        RunArtifactTransport(backing, max_workers=65)


def test_release_status_and_gc_preview_are_stable_and_protect_commits(
    tmp_path: Path,
) -> None:
    archive_id = "run_archive_" + "a" * 24
    backing = LocalBlobStore(tmp_path / "status-remote")
    transport = RunArtifactTransport(backing)
    absent = transport.status(archive_id)
    assert absent.state is RunArtifactReleaseState.ABSENT

    staged_key = f"{transport.release_root(archive_id)}/chunks/staged.part"
    backing.put_if_absent(staged_key, b"staged")
    staged = transport.status(archive_id)
    assert staged.state is RunArtifactReleaseState.STAGED
    assert staged.newest_modified_at is not None
    early = transport.preview_garbage_collection(
        archive_id,
        min_age_seconds=60,
        now=staged.newest_modified_at,
    )
    assert not early.eligible
    assert early.reason is RunArtifactGcReason.TOO_RECENT
    assert early.eligible_at is not None
    mature = transport.preview_garbage_collection(
        archive_id,
        min_age_seconds=60,
        now=early.eligible_at,
    )
    assert mature.eligible
    assert mature.reason is RunArtifactGcReason.ELIGIBLE
    assert mature.state_digest == early.state_digest

    archive, receipt = build_archive(tmp_path, "committed-status")
    committed_transport = RunArtifactTransport(
        LocalBlobStore(tmp_path / "committed-status-remote"),
        chunk_size_bytes=256,
    )
    committed_transport.publish(archive, expected_sha256=receipt.content_digest)
    committed = committed_transport.status(receipt.archive_id)
    protected = committed_transport.preview_garbage_collection(
        receipt.archive_id,
        min_age_seconds=1,
        now=committed.newest_modified_at,
    )
    assert committed.state is RunArtifactReleaseState.COMMITTED
    assert not protected.eligible
    assert protected.reason is RunArtifactGcReason.COMMITTED


def test_gc_requires_exact_preview_and_leaves_durable_evidence(tmp_path: Path) -> None:
    archive, receipt = build_archive(tmp_path, "gc-release")
    backing = LocalBlobStore(tmp_path / "gc-remote")
    failing = InstrumentedStore(
        backing,
        fail_put=lambda key: key.endswith("/commit.json"),
    )
    with pytest.raises(RuntimeError, match="injected upload failure"):
        RunArtifactTransport(failing, chunk_size_bytes=160).publish(
            archive,
            expected_sha256=receipt.content_digest,
        )
    transport = RunArtifactTransport(backing, chunk_size_bytes=160)
    preliminary = transport.preview_garbage_collection(
        receipt.archive_id,
        min_age_seconds=1,
    )
    assert preliminary.eligible_at is not None
    preview = transport.preview_garbage_collection(
        receipt.archive_id,
        min_age_seconds=1,
        now=preliminary.eligible_at,
    )
    assert preview.eligible

    with pytest.raises(RunArtifactTransportError, match="state digest"):
        transport.garbage_collect(
            receipt.archive_id,
            min_age_seconds=1,
            confirm_state_digest="0" * 64,
            operator="release-operator",
            reason="abandoned interrupted publication",
            now=preview.eligible_at,
        )
    assert backing.head(transport.commit_key(receipt.archive_id)) is None

    record = transport.garbage_collect(
        receipt.archive_id,
        min_age_seconds=1,
        confirm_state_digest=preview.state_digest,
        operator="release-operator",
        reason="abandoned interrupted publication",
        now=preview.eligible_at,
    )
    repeated = transport.garbage_collect(
        receipt.archive_id,
        min_age_seconds=1,
        confirm_state_digest=preview.state_digest,
        operator="release-operator",
        reason="abandoned interrupted publication",
        now=preview.eligible_at + timedelta(seconds=1),
    )

    assert record == repeated
    assert record.target_object_count == preview.object_count
    assert record.deleted_object_count == preview.object_count
    assert backing.head(transport.gc_record_key(record.gc_id)) is not None
    assert backing.list(f"{transport.release_root(receipt.archive_id)}/") == (
        transport.commit_key(receipt.archive_id),
    )
    assert transport.status(receipt.archive_id).state is RunArtifactReleaseState.GARBAGE_COLLECTED
    with pytest.raises(RunArtifactTransportError, match="tombstoned"):
        transport.publish(archive, expected_sha256=receipt.content_digest)


def test_gc_rejects_changed_state_and_resumes_after_delete_failure(tmp_path: Path) -> None:
    changed_id = "run_archive_" + "b" * 24
    changed_store = LocalBlobStore(tmp_path / "changed-remote")
    changed_transport = RunArtifactTransport(changed_store)
    changed_key = f"{changed_transport.release_root(changed_id)}/staged.bin"
    changed_store.put_if_absent(changed_key, b"before")
    initial = changed_transport.preview_garbage_collection(
        changed_id,
        min_age_seconds=1,
    )
    assert initial.eligible_at is not None
    preview = changed_transport.preview_garbage_collection(
        changed_id,
        min_age_seconds=1,
        now=initial.eligible_at,
    )
    (changed_store.root / changed_key).write_bytes(b"after")
    with pytest.raises(RunArtifactTransportError, match="state digest"):
        changed_transport.garbage_collect(
            changed_id,
            min_age_seconds=1,
            confirm_state_digest=preview.state_digest,
            operator="release-operator",
            reason="state changed before deletion",
            now=initial.eligible_at + timedelta(seconds=1),
        )
    assert changed_store.head(changed_transport.commit_key(changed_id)) is None

    archive, receipt = build_archive(tmp_path, "gc-resume")
    backing = LocalBlobStore(tmp_path / "gc-resume-remote")
    failing_commit = InstrumentedStore(
        backing,
        fail_put=lambda key: key.endswith("/commit.json"),
    )
    with pytest.raises(RuntimeError, match="injected upload failure"):
        RunArtifactTransport(failing_commit, chunk_size_bytes=128).publish(
            archive,
            expected_sha256=receipt.content_digest,
        )
    transport = RunArtifactTransport(backing)
    early = transport.preview_garbage_collection(receipt.archive_id, min_age_seconds=1)
    assert early.eligible_at is not None
    eligible = transport.preview_garbage_collection(
        receipt.archive_id,
        min_age_seconds=1,
        now=early.eligible_at,
    )
    delete_fault = DeleteFaultStore(backing, fail_after=1)
    with pytest.raises(RuntimeError, match="injected delete failure"):
        RunArtifactTransport(delete_fault).garbage_collect(
            receipt.archive_id,
            min_age_seconds=1,
            confirm_state_digest=eligible.state_digest,
            operator="release-operator",
            reason="resume interrupted garbage collection",
            now=early.eligible_at,
        )
    assert transport.status(receipt.archive_id).state is RunArtifactReleaseState.GC_IN_PROGRESS
    completed = transport.garbage_collect(
        receipt.archive_id,
        min_age_seconds=1,
        confirm_state_digest=eligible.state_digest,
        operator="release-operator",
        reason="resume interrupted garbage collection",
        now=early.eligible_at + timedelta(seconds=1),
    )
    assert completed.already_missing_object_count == 1
    assert transport.status(receipt.archive_id).state is RunArtifactReleaseState.GARBAGE_COLLECTED


def test_commit_and_gc_tombstone_are_mutually_exclusive(tmp_path: Path) -> None:
    archive, receipt = build_archive(tmp_path, "gc-race")
    backing = LocalBlobStore(tmp_path / "gc-race-remote")
    staged_transport = RunArtifactTransport(backing, chunk_size_bytes=128)
    failing = InstrumentedStore(
        backing,
        fail_put=lambda key: key.endswith("/commit.json"),
    )
    with pytest.raises(RuntimeError, match="injected upload failure"):
        RunArtifactTransport(failing, chunk_size_bytes=128).publish(
            archive,
            expected_sha256=receipt.content_digest,
        )
    manifest_key = next(
        key
        for key in backing.list(f"{staged_transport.release_root(receipt.archive_id)}/manifests")
        if key.endswith(".json")
    )
    manifest_payload = backing.get(manifest_key)
    manifest = RunArtifactTransportManifest.model_validate_json(manifest_payload)
    manifest_ref = RunArtifactTransportObjectRef(
        key=manifest_key,
        media_type="application/json",
        size_bytes=len(manifest_payload),
        content_digest=hashlib.sha256(manifest_payload).hexdigest(),
    )
    commit = RunArtifactTransportCommit(
        commit_id=RunArtifactTransportCommit.expected_commit_id(
            archive_id=receipt.archive_id,
            transport_id=manifest.transport_id,
            manifest=manifest_ref,
        ),
        archive_id=receipt.archive_id,
        transport_id=manifest.transport_id,
        manifest=manifest_ref,
    )
    early = staged_transport.preview_garbage_collection(
        receipt.archive_id,
        min_age_seconds=1,
    )
    assert early.eligible_at is not None
    preview = staged_transport.preview_garbage_collection(
        receipt.archive_id,
        min_age_seconds=1,
        now=early.eligible_at,
    )
    barrier = threading.Barrier(2)

    def write_commit() -> str:
        barrier.wait()
        backing.put_if_absent(
            staged_transport.commit_key(receipt.archive_id),
            commit.canonical_bytes() + b"\n",
        )
        return "commit"

    def run_gc() -> str:
        barrier.wait()
        staged_transport.garbage_collect(
            receipt.archive_id,
            min_age_seconds=1,
            confirm_state_digest=preview.state_digest,
            operator="release-operator",
            reason="race abandoned publication cleanup",
            now=early.eligible_at,
        )
        return "gc"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(write_commit), executor.submit(run_gc))
        successes = []
        failures = []
        for future in futures:
            try:
                successes.append(future.result())
            except (BlobConflictError, RunArtifactTransportError) as error:
                failures.append(error)

    assert len(successes) == 1
    assert len(failures) == 1
    state = staged_transport.status(receipt.archive_id).state
    assert state in {RunArtifactReleaseState.COMMITTED, RunArtifactReleaseState.GARBAGE_COLLECTED}


def test_transport_status_gc_validation_and_incomplete_identities(tmp_path: Path) -> None:
    archive_id = "run_archive_" + "e" * 24
    backing = LocalBlobStore(tmp_path / "identity-remote")
    key = f"{RunArtifactTransport.release_root(archive_id)}/staged.bin"
    backing.put_if_absent(key, b"staged")

    missing_time = RunArtifactTransport(
        IdentityMaskingStore(backing, hide_last_modified=True)
    ).status(archive_id)
    assert missing_time.state is RunArtifactReleaseState.INVALID

    no_digest = RunArtifactTransport(IdentityMaskingStore(backing, hide_digest=True))
    preliminary = no_digest.preview_garbage_collection(archive_id, min_age_seconds=1)
    assert preliminary.objects[0].content_sha256 == hashlib.sha256(b"staged").hexdigest()
    incomplete = RunArtifactTransport(IdentityMaskingStore(backing, hide_etag=True))
    assert (
        incomplete.preview_garbage_collection(archive_id, min_age_seconds=1).reason
        is RunArtifactGcReason.INVALID
    )
    wrong_size = RunArtifactTransport(IdentityMaskingStore(backing, hide_digest=True, size_delta=1))
    assert (
        wrong_size.preview_garbage_collection(archive_id, min_age_seconds=1).reason
        is RunArtifactGcReason.INVALID
    )

    transport = RunArtifactTransport(backing)
    with pytest.raises(ValueError, match="minimum age"):
        transport.preview_garbage_collection(archive_id, min_age_seconds=0)
    with pytest.raises(ValueError, match="minimum age"):
        transport.garbage_collect(
            archive_id,
            min_age_seconds=0,
            confirm_state_digest="0" * 64,
            operator="operator",
            reason="valid reason",
        )
    for kwargs, message in (
        ({"confirm_state_digest": "bad"}, "64-character"),
        ({"operator": ""}, "operator cannot be empty"),
        ({"reason": "short"}, "at least eight"),
    ):
        parameters = {
            "min_age_seconds": 1,
            "confirm_state_digest": "0" * 64,
            "operator": "operator",
            "reason": "valid reason",
            **kwargs,
        }
        with pytest.raises(ValueError, match=message):
            transport.garbage_collect(archive_id, **parameters)

    absent_id = "run_archive_" + "f" * 24
    with pytest.raises(RunArtifactTransportError, match="not eligible"):
        transport.garbage_collect(
            absent_id,
            min_age_seconds=1,
            confirm_state_digest="0" * 64,
            operator="operator",
            reason="absent release cleanup",
        )


def test_gc_tombstone_detects_late_and_mutated_objects(tmp_path: Path) -> None:
    archive_id = "run_archive_" + "1" * 24
    backing = LocalBlobStore(tmp_path / "late-object-remote")
    transport = RunArtifactTransport(backing)
    target_key = f"{transport.release_root(archive_id)}/target.bin"
    backing.put_if_absent(target_key, b"target")
    early = transport.preview_garbage_collection(archive_id, min_age_seconds=1)
    assert early.eligible_at is not None
    preview = transport.preview_garbage_collection(
        archive_id,
        min_age_seconds=1,
        now=early.eligible_at,
    )
    fault = DeleteFaultStore(backing, fail_after=0)
    with pytest.raises(RuntimeError, match="injected delete failure"):
        RunArtifactTransport(fault).garbage_collect(
            archive_id,
            min_age_seconds=1,
            confirm_state_digest=preview.state_digest,
            operator="operator",
            reason="late object race validation",
            now=early.eligible_at,
        )

    late_key = f"{transport.release_root(archive_id)}/late.bin"
    late = backing.put_if_absent(late_key, b"late").blob
    with pytest.raises(RunArtifactTransportError, match="unconfirmed objects"):
        transport.garbage_collect(
            archive_id,
            min_age_seconds=1,
            confirm_state_digest=preview.state_digest,
            operator="operator",
            reason="late object race validation",
            now=early.eligible_at + timedelta(seconds=1),
        )
    assert backing.delete_if_match(late_key, late)
    (backing.root / target_key).write_bytes(b"change")
    with pytest.raises(RunArtifactTransportError, match="changed after GC preview"):
        transport.garbage_collect(
            archive_id,
            min_age_seconds=1,
            confirm_state_digest=preview.state_digest,
            operator="operator",
            reason="late object race validation",
            now=early.eligible_at + timedelta(seconds=1),
        )
    with pytest.raises(RunArtifactTransportError, match="minimum age differs"):
        transport.garbage_collect(
            archive_id,
            min_age_seconds=2,
            confirm_state_digest=preview.state_digest,
            operator="operator",
            reason="late object race validation",
            now=early.eligible_at + timedelta(seconds=1),
        )
    with pytest.raises(RunArtifactTransportError, match="state digest differs"):
        transport.garbage_collect(
            archive_id,
            min_age_seconds=1,
            confirm_state_digest="0" * 64,
            operator="operator",
            reason="late object race validation",
            now=early.eligible_at + timedelta(seconds=1),
        )
    with pytest.raises(RunArtifactTransportError, match="operator or reason differs"):
        transport.garbage_collect(
            archive_id,
            min_age_seconds=1,
            confirm_state_digest=preview.state_digest,
            operator="another-operator",
            reason="late object race validation",
            now=early.eligible_at + timedelta(seconds=1),
        )


def test_gc_counts_disappeared_targets_and_status_rejects_post_record_objects(
    tmp_path: Path,
) -> None:
    archive_id = "run_archive_" + "2" * 24
    backing = LocalBlobStore(tmp_path / "false-delete-remote")
    key = f"{RunArtifactTransport.release_root(archive_id)}/target.bin"
    backing.put_if_absent(key, b"target")
    transport = RunArtifactTransport(backing)
    early = transport.preview_garbage_collection(archive_id, min_age_seconds=1)
    assert early.eligible_at is not None
    preview = transport.preview_garbage_collection(
        archive_id,
        min_age_seconds=1,
        now=early.eligible_at,
    )
    record = RunArtifactTransport(FalseDeleteStore(backing)).garbage_collect(
        archive_id,
        min_age_seconds=1,
        confirm_state_digest=preview.state_digest,
        operator="operator",
        reason="target disappeared during delete",
        now=early.eligible_at,
    )
    assert record.deleted_object_count == 0
    assert record.already_missing_object_count == 1

    backing.put_if_absent(f"{transport.release_root(archive_id)}/unexpected.bin", b"unexpected")
    invalid = transport.status(archive_id)
    assert invalid.state is RunArtifactReleaseState.INVALID
    assert "target objects remain" in invalid.detail


def test_store_inventory_and_retention_plan_are_stable_and_exact(tmp_path: Path) -> None:
    backing = LocalBlobStore(tmp_path / "inventory-remote")
    transport = RunArtifactTransport(backing)
    old_ids = ("run_archive_" + "3" * 24, "run_archive_" + "4" * 24)
    recent_id = "run_archive_" + "5" * 24
    action_time = datetime.now(timezone.utc)
    for archive_id in (*old_ids, recent_id):
        key = f"{transport.release_root(archive_id)}/staged.bin"
        backing.put_if_absent(key, archive_id.encode("ascii"))
        timestamp = (
            action_time - timedelta(hours=2) if archive_id in old_ids else action_time
        ).timestamp()
        os.utime(backing.root / key, (timestamp, timestamp))

    inventory = transport.inventory(now=action_time)
    repeated = transport.inventory(now=action_time + timedelta(seconds=1))
    plan = transport.plan_garbage_collection(
        min_age_seconds=60,
        now=action_time,
    )
    repeated_plan = transport.plan_garbage_collection(
        min_age_seconds=60,
        now=action_time + timedelta(seconds=1),
    )

    assert inventory.release_count == 3
    assert inventory.state_counts[RunArtifactReleaseState.STAGED.value] == 3
    assert inventory.state_digest == repeated.state_digest
    assert tuple(item.archive_id for item in plan.candidates) == old_ids
    assert plan.reclaimable_bytes == sum(item.total_bytes for item in plan.candidates)
    assert plan.plan_id == repeated_plan.plan_id
    assert plan.inventory.state_digest == inventory.state_digest


def test_gc_batch_execution_is_bounded_idempotent_and_evidenced(tmp_path: Path) -> None:
    backing = LocalBlobStore(tmp_path / "batch-remote")
    transport = RunArtifactTransport(backing)
    archive_ids = ("run_archive_" + "6" * 24, "run_archive_" + "7" * 24)
    action_time = datetime.now(timezone.utc)
    for archive_id in archive_ids:
        for index in range(2):
            key = f"{transport.release_root(archive_id)}/chunk-{index}.bin"
            backing.put_if_absent(key, f"{archive_id}-{index}".encode("ascii"))
            timestamp = (action_time - timedelta(hours=2)).timestamp()
            os.utime(backing.root / key, (timestamp, timestamp))
    plan = transport.plan_garbage_collection(min_age_seconds=60, now=action_time)
    concurrent = BatchDeleteConcurrencyStore(backing)
    batch_transport = RunArtifactTransport(concurrent)

    with pytest.raises(RunArtifactTransportError, match="plan ID"):
        batch_transport.execute_garbage_collection_plan(
            plan,
            confirm_plan_id="run_gc_plan_" + "0" * 24,
            operator="batch-operator",
            reason="reviewed expired staging batch",
            max_workers=2,
            now=action_time,
        )
    batch = batch_transport.execute_garbage_collection_plan(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-operator",
        reason="reviewed expired staging batch",
        max_workers=2,
        now=action_time,
    )
    repeated = transport.execute_garbage_collection_plan(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-operator",
        reason="reviewed expired staging batch",
        max_workers=1,
        now=action_time + timedelta(seconds=1),
    )

    assert batch == repeated
    assert batch.candidate_count == 2
    assert batch.deleted_object_count == 4
    assert 2 <= concurrent.max_active_deletes <= 2
    assert backing.head(transport.gc_batch_intent_key(batch.intent.batch_id)) is not None
    assert backing.head(transport.gc_batch_record_key(batch.intent.batch_id)) is not None
    assert all(
        transport.status(archive_id).state is RunArtifactReleaseState.GARBAGE_COLLECTED
        for archive_id in archive_ids
    )


def test_gc_batch_resumes_partial_completion_and_rejects_unsafe_inventory(
    tmp_path: Path,
) -> None:
    backing = LocalBlobStore(tmp_path / "batch-resume-remote")
    transport = RunArtifactTransport(backing)
    archive_ids = ("run_archive_" + "a1" * 12, "run_archive_" + "b2" * 12)
    action_time = datetime.now(timezone.utc)
    for archive_id in archive_ids:
        key = f"{transport.release_root(archive_id)}/staged.bin"
        backing.put_if_absent(key, archive_id.encode("ascii"))
        timestamp = (action_time - timedelta(hours=2)).timestamp()
        os.utime(backing.root / key, (timestamp, timestamp))
    plan = transport.plan_garbage_collection(min_age_seconds=60, now=action_time)
    faulty = RunArtifactTransport(SelectiveDeleteFaultStore(backing, archive_ids[1]))

    with pytest.raises(RuntimeError, match="injected batch delete failure"):
        faulty.execute_garbage_collection_plan(
            plan,
            confirm_plan_id=plan.plan_id,
            operator="batch-operator",
            reason="resume partially completed batch",
            max_workers=2,
            now=action_time,
        )
    batch_id = RunArtifactGcBatchIntent.expected_batch_id(
        plan_id=plan.plan_id,
        operator="batch-operator",
        reason="resume partially completed batch",
    )
    assert backing.head(transport.gc_batch_intent_key(batch_id)) is not None
    assert backing.head(transport.gc_batch_record_key(batch_id)) is None

    completed = transport.execute_garbage_collection_plan(
        plan,
        confirm_plan_id=plan.plan_id,
        operator="batch-operator",
        reason="resume partially completed batch",
        max_workers=1,
        now=action_time + timedelta(seconds=1),
    )
    assert completed.candidate_count == 2
    assert backing.head(transport.gc_batch_record_key(batch_id)) is not None

    unsafe = LocalBlobStore(tmp_path / "unsafe-inventory")
    unsafe.put_if_absent("run-releases/not-an-archive/object.bin", b"unsafe")
    unsafe_transport = RunArtifactTransport(unsafe)
    inventory = unsafe_transport.inventory(now=action_time)
    assert inventory.invalid_keys == ("run-releases/not-an-archive/object.bin",)
    with pytest.raises(RunArtifactTransportError, match="unclassified keys"):
        unsafe_transport.plan_garbage_collection(min_age_seconds=60, now=action_time)


def test_transport_status_and_gc_contracts_reject_inconsistent_evidence() -> None:
    archive_id = "run_archive_" + "c" * 24
    observed_at = datetime(2026, 9, 18, 16, 0, tzinfo=timezone.utc)
    eligible_at = observed_at + timedelta(seconds=60)
    info = BlobInfo(
        key=f"run-releases/{archive_id}/staged.bin",
        size_bytes=7,
        etag="etag-1",
        content_sha256=hashlib.sha256(b"staged").hexdigest(),
        last_modified=observed_at,
    )
    digest = RunArtifactGcPreview.expected_state_digest(
        archive_id=archive_id,
        min_age_seconds=60.0,
        eligible_at=eligible_at,
        objects=(info,),
    )
    preview_payload: dict[str, Any] = {
        "archive_id": archive_id,
        "eligible": True,
        "reason": RunArtifactGcReason.ELIGIBLE,
        "observed_at": eligible_at,
        "min_age_seconds": 60.0,
        "eligible_at": eligible_at,
        "objects": (info,),
        "object_count": 1,
        "total_bytes": 7,
        "state_digest": digest,
    }
    preview = RunArtifactGcPreview.model_validate(preview_payload)

    for update, message in (
        ({"observed_at": observed_at.replace(tzinfo=None)}, "observation time"),
        ({"eligible_at": eligible_at.replace(tzinfo=None)}, "eligibility time"),
        ({"object_count": 2}, "object count"),
        ({"total_bytes": 8}, "byte count"),
        (
            {
                "objects": (
                    info.model_copy(update={"key": f"run-releases/run_archive_{'d' * 24}/x"}),
                )
            },
            "another release",
        ),
        ({"objects": (info.model_copy(update={"etag": None}),)}, "complete object"),
        ({"eligible": False}, "eligibility does not match"),
        ({"eligible_at": None}, "require an eligibility time"),
        ({"state_digest": "0" * 64}, "state digest"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactGcPreview.model_validate({**preview_payload, **update})

    status_payload: dict[str, Any] = {
        "archive_id": archive_id,
        "state": RunArtifactReleaseState.STAGED,
        "observed_at": observed_at,
        "object_count": 1,
        "total_bytes": 7,
        "oldest_modified_at": observed_at,
        "newest_modified_at": observed_at,
        "detail": "staged release",
    }
    for update, message in (
        ({"observed_at": observed_at.replace(tzinfo=None)}, "timezone-aware"),
        ({"object_count": 0}, "empty transport status"),
        ({"oldest_modified_at": None}, "requires modification times"),
        (
            {
                "oldest_modified_at": observed_at + timedelta(seconds=1),
                "newest_modified_at": observed_at,
            },
            "reversed",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactTransportStatus.model_validate({**status_payload, **update})

    gc_id = RunArtifactGcTombstone.expected_gc_id(
        state_digest=preview.state_digest,
        operator="operator",
        reason="expired staged release",
        created_at=eligible_at,
    )
    tombstone_payload: dict[str, Any] = {
        "gc_id": gc_id,
        "preview": preview,
        "operator": "operator",
        "reason": "expired staged release",
        "created_at": eligible_at,
    }
    with pytest.raises(ValueError, match="eligible preview"):
        RunArtifactGcTombstone.model_validate(
            {
                **tombstone_payload,
                "preview": RunArtifactGcPreview.model_validate(
                    {
                        **preview_payload,
                        "eligible": False,
                        "reason": RunArtifactGcReason.TOO_RECENT,
                    }
                ),
            }
        )
    with pytest.raises(ValueError, match="time must be timezone-aware"):
        RunArtifactGcTombstone.model_validate(
            {**tombstone_payload, "created_at": eligible_at.replace(tzinfo=None)}
        )
    with pytest.raises(ValueError, match="ID does not match"):
        RunArtifactGcTombstone.model_validate({**tombstone_payload, "gc_id": "run_gc_" + "0" * 24})

    record_payload: dict[str, Any] = {
        "gc_id": gc_id,
        "archive_id": archive_id,
        "state_digest": preview.state_digest,
        "operator": "operator",
        "reason": "expired staged release",
        "started_at": eligible_at,
        "completed_at": eligible_at + timedelta(seconds=1),
        "target_object_count": 1,
        "deleted_object_count": 1,
        "already_missing_object_count": 0,
        "target_bytes": 7,
    }
    for update, message in (
        ({"started_at": eligible_at.replace(tzinfo=None)}, "times must be timezone-aware"),
        ({"completed_at": observed_at}, "completion precedes"),
        ({"deleted_object_count": 0}, "counts do not cover"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactGcRecord.model_validate({**record_payload, **update})

    plan_status = RunArtifactTransportStatus.model_validate(
        {**status_payload, "observed_at": eligible_at}
    )
    state_counts = {
        state.value: int(state is RunArtifactReleaseState.STAGED)
        for state in RunArtifactReleaseState
    }
    inventory_digest = RunArtifactStoreInventory.expected_state_digest(
        releases=(plan_status,),
        invalid_keys=(),
    )
    inventory_payload: dict[str, Any] = {
        "observed_at": eligible_at,
        "releases": (plan_status,),
        "invalid_keys": (),
        "release_count": 1,
        "object_count": 1,
        "total_bytes": 7,
        "state_counts": state_counts,
        "state_digest": inventory_digest,
    }
    inventory = RunArtifactStoreInventory.model_validate(inventory_payload)
    with pytest.raises(ValueError, match="count does not match"):
        RunArtifactStoreInventory.model_validate({**inventory_payload, "release_count": 2})
    with pytest.raises(ValueError, match="state counts"):
        RunArtifactStoreInventory.model_validate({**inventory_payload, "state_counts": {}})
    with pytest.raises(ValueError, match="state digest"):
        RunArtifactStoreInventory.model_validate({**inventory_payload, "state_digest": "0" * 64})

    plan_id = RunArtifactGcPlan.expected_plan_id(
        inventory_state_digest=inventory.state_digest,
        min_age_seconds=60.0,
        candidates=(preview,),
    )
    plan_payload: dict[str, Any] = {
        "plan_id": plan_id,
        "inventory": inventory,
        "min_age_seconds": 60.0,
        "candidates": (preview,),
        "candidate_count": 1,
        "reclaimable_bytes": 7,
    }
    plan = RunArtifactGcPlan.model_validate(plan_payload)
    for update, message in (
        ({"candidate_count": 2}, "candidate count"),
        ({"reclaimable_bytes": 8}, "reclaimable bytes"),
        ({"plan_id": "run_gc_plan_" + "0" * 24}, "plan ID"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactGcPlan.model_validate({**plan_payload, **update})

    batch_id = RunArtifactGcBatchIntent.expected_batch_id(
        plan_id=plan.plan_id,
        operator="operator",
        reason="expired staged release",
    )
    intent_payload: dict[str, Any] = {
        "batch_id": batch_id,
        "plan": plan,
        "operator": "operator",
        "reason": "expired staged release",
        "created_at": eligible_at,
    }
    intent = RunArtifactGcBatchIntent.model_validate(intent_payload)
    with pytest.raises(ValueError, match="time must be timezone-aware"):
        RunArtifactGcBatchIntent.model_validate(
            {**intent_payload, "created_at": eligible_at.replace(tzinfo=None)}
        )
    with pytest.raises(ValueError, match="batch ID"):
        RunArtifactGcBatchIntent.model_validate(
            {**intent_payload, "batch_id": "run_gc_batch_" + "0" * 24}
        )

    member_record = RunArtifactGcRecord.model_validate(record_payload)
    batch_record_payload: dict[str, Any] = {
        "intent": intent,
        "completed_at": eligible_at + timedelta(seconds=1),
        "records": (member_record,),
        "candidate_count": 1,
        "deleted_object_count": 1,
        "already_missing_object_count": 0,
        "target_bytes": 7,
    }
    RunArtifactGcBatchRecord.model_validate(batch_record_payload)
    for update, message in (
        ({"completed_at": observed_at}, "completion precedes"),
        ({"candidate_count": 2}, "candidate count"),
        ({"records": ()}, "do not cover"),
        ({"target_bytes": 8}, "byte count"),
    ):
        with pytest.raises(ValueError, match=message):
            RunArtifactGcBatchRecord.model_validate({**batch_record_payload, **update})

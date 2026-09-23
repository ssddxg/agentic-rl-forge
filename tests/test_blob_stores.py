from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agentic_rl_forge.storage import (
    BlobConflictError,
    LocalBlobStore,
    S3ConditionalBlobStore,
)


class FakeS3Error(Exception):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str], str, datetime]] = {}
        self.conflicts_remaining = 0
        self.last_put: dict[str, Any] = {}
        self.last_delete: dict[str, Any] = {}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.last_put = kwargs
        if self.conflicts_remaining:
            self.conflicts_remaining -= 1
            raise FakeS3Error(409, "ConditionalRequestConflict")
        key = str(kwargs["Key"])
        if key in self.objects:
            raise FakeS3Error(412, "PreconditionFailed")
        data = bytes(kwargs["Body"])
        metadata = {str(k): str(v) for k, v in kwargs["Metadata"].items()}
        etag = f"etag-{len(self.objects) + 1}"
        last_modified = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=len(self.objects)
        )
        self.objects[key] = (data, metadata, etag, last_modified)
        return {"ETag": f'"{etag}"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise FakeS3Error(404, "NoSuchKey")
        data, _, _, _ = self.objects[key]
        return {"Body": io.BytesIO(data)}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise FakeS3Error(404, "NotFound")
        data, metadata, etag, last_modified = self.objects[key]
        return {
            "ContentLength": len(data),
            "Metadata": metadata,
            "ETag": f'"{etag}"',
            "LastModified": last_modified,
        }

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = str(kwargs["Prefix"])
        return {
            "Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(prefix)],
            "IsTruncated": False,
        }

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.last_delete = kwargs
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise FakeS3Error(404, "NoSuchKey")
        _, _, etag, _ = self.objects[key]
        if kwargs.get("IfMatch") != etag:
            raise FakeS3Error(412, "PreconditionFailed")
        del self.objects[key]
        return {}


def test_local_blob_store_is_conditional_and_path_safe(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    created = store.put_if_absent("runs/1/data.json", b"payload")
    repeated = store.put_if_absent("runs/1/data.json", b"payload")

    assert created.created
    assert not repeated.created
    assert store.get("runs/1/data.json") == b"payload"
    assert store.head("runs/1/data.json") == created.blob
    assert created.blob.etag is not None
    assert created.blob.last_modified is not None
    assert store.list("runs/1") == ("runs/1/data.json",)
    assert store.head("missing") is None
    with pytest.raises(KeyError):
        store.get("missing")
    with pytest.raises(BlobConflictError):
        store.put_if_absent("runs/1/data.json", b"different")
    with pytest.raises(ValueError, match="relative path"):
        store.put_if_absent("../escape", b"unsafe")
    expected = store.head("runs/1/data.json")
    assert expected is not None
    with pytest.raises(BlobConflictError, match="changed before deletion"):
        store.delete_if_match(
            "runs/1/data.json",
            expected.model_copy(update={"size_bytes": expected.size_bytes + 1}),
        )
    assert store.delete_if_match("runs/1/data.json", expected)
    assert not store.delete_if_match("runs/1/data.json", expected)


def test_s3_blob_store_uses_if_none_match_and_detects_conflicts() -> None:
    client = FakeS3Client()
    client.conflicts_remaining = 1
    store = S3ConditionalBlobStore(
        client,
        bucket="bucket",
        prefix="project",
        conflict_retries=2,
    )

    created = store.put_if_absent(
        "runs/1/data.json",
        b"payload",
        metadata={"run": 1},
    )
    repeated = store.put_if_absent("runs/1/data.json", b"payload")

    assert created.created
    assert not repeated.created
    assert client.last_put["IfNoneMatch"] == "*"
    assert client.last_put["Key"] == "project/runs/1/data.json"
    assert store.get("runs/1/data.json") == b"payload"
    assert store.head("runs/1/data.json") is not None
    assert store.list("runs") == ("runs/1/data.json",)
    assert store.head("missing") is None
    with pytest.raises(KeyError):
        store.get("missing")
    with pytest.raises(BlobConflictError):
        store.put_if_absent("runs/1/data.json", b"different")
    expected = store.head("runs/1/data.json")
    assert expected is not None
    assert store.delete_if_match("runs/1/data.json", expected)
    assert client.last_delete["IfMatch"] == expected.etag
    assert not store.delete_if_match("runs/1/data.json", expected)


def test_s3_blob_store_reports_persistent_conditional_conflicts() -> None:
    client = FakeS3Client()
    client.conflicts_remaining = 3
    store = S3ConditionalBlobStore(
        client,
        bucket="bucket",
        conflict_retries=1,
    )

    with pytest.raises(RuntimeError, match="remained in conflict"):
        store.put_if_absent("data.json", b"payload")

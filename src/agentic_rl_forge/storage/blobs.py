from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from agentic_rl_forge.contracts import BlobInfo, BlobPutResult, JsonObject


class BlobConflictError(ValueError):
    pass


class ConditionalBlobStore(Protocol):
    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: JsonObject | None = None,
    ) -> BlobPutResult: ...

    def get(self, key: str) -> bytes: ...

    def head(self, key: str) -> BlobInfo | None: ...

    def list(self, prefix: str = "") -> tuple[str, ...]: ...

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool: ...


class LocalBlobStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._filesystem_path(self.root).mkdir(parents=True, exist_ok=True)

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: JsonObject | None = None,
    ) -> BlobPutResult:
        del metadata
        normalized = self._normalize_key(key)
        target = self._filesystem_path(self.root / normalized)
        target.parent.mkdir(parents=True, exist_ok=True)
        content_sha256 = hashlib.sha256(data).hexdigest()
        if target.exists():
            if target.read_bytes() != data:
                raise BlobConflictError(f"blob {normalized!r} already exists with other content")
            info = self.head(normalized)
            if info is None:
                raise RuntimeError(f"blob {normalized!r} disappeared during conditional write")
            return BlobPutResult(created=False, blob=info)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            # Keep the temporary basename deliberately short.  Blob keys often
            # contain a digest already and repeating that basename here can push
            # otherwise valid destinations beyond the legacy Windows MAX_PATH
            # limit before the atomic hard-link is attempted.
            prefix=".arf-blob-",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != data:
                    raise BlobConflictError(
                        f"blob {normalized!r} already exists with other content"
                    ) from None
                created = False
            else:
                created = True
        finally:
            if temporary.exists():
                temporary.unlink()
        info = self.head(normalized)
        if info is None:
            raise RuntimeError(f"blob {normalized!r} disappeared after conditional write")
        if info.content_sha256 != content_sha256:
            raise RuntimeError(f"blob {normalized!r} changed after conditional write")
        return BlobPutResult(created=created, blob=info)

    def get(self, key: str) -> bytes:
        path = self._filesystem_path(self.root / self._normalize_key(key))
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def head(self, key: str) -> BlobInfo | None:
        normalized = self._normalize_key(key)
        path = self._filesystem_path(self.root / normalized)
        if not path.is_file():
            return None
        data = path.read_bytes()
        status = path.stat()
        content_digest = hashlib.sha256(data).hexdigest()
        return BlobInfo(
            key=normalized,
            size_bytes=len(data),
            etag=self._local_etag(status, content_digest),
            content_sha256=content_digest,
            last_modified=datetime.fromtimestamp(status.st_mtime, timezone.utc),
        )

    def list(self, prefix: str = "") -> tuple[str, ...]:
        normalized_prefix = self._normalize_prefix(prefix)
        root = self._filesystem_path(self.root)
        scan_root = root
        if prefix.endswith("/"):
            scan_root = self._filesystem_path(self.root / normalized_prefix)
            if not scan_root.is_dir():
                return ()
        return tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in scan_root.rglob("*")
                if path.is_file()
                and not path.name.startswith(".")
                and path.relative_to(root).as_posix().startswith(normalized_prefix)
            )
        )

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        normalized = self._normalize_key(key)
        if expected.key != normalized:
            raise BlobConflictError("blob delete identity belongs to another key")
        target = self._filesystem_path(self.root / normalized)
        if target.is_symlink():
            raise BlobConflictError(f"blob {normalized!r} became a symbolic link")
        current = self.head(normalized)
        if current is None:
            return False
        if not _blob_identity_matches(current, expected):
            raise BlobConflictError(f"blob {normalized!r} changed before deletion")
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"blob {normalized!r} remained after deletion")
        return True

    @staticmethod
    def _normalize_key(key: str) -> str:
        path = PurePosixPath(key)
        if not key or path.is_absolute() or ".." in path.parts or path.as_posix() in {".", ""}:
            raise ValueError("blob key must be a non-empty relative path")
        return path.as_posix()

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        if not prefix:
            return ""
        return LocalBlobStore._normalize_key(prefix).rstrip("/")

    @staticmethod
    def _filesystem_path(path: Path) -> Path:
        """Return a Windows extended-length path without changing public keys."""
        if os.name != "nt":
            return path
        absolute = os.path.abspath(path)
        if absolute.startswith("\\\\?\\"):
            return Path(absolute)
        if absolute.startswith("\\\\"):
            return Path(f"\\\\?\\UNC\\{absolute[2:]}")
        return Path(f"\\\\?\\{absolute}")

    @staticmethod
    def _local_etag(status: os.stat_result, content_digest: str) -> str:
        identity = (
            f"{status.st_dev}:{status.st_ino}:{status.st_mtime_ns}:"
            f"{status.st_size}:{content_digest}"
        )
        return hashlib.sha256(identity.encode("ascii")).hexdigest()


class S3Client(Protocol):
    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...

    def delete_object(self, **kwargs: Any) -> dict[str, Any]: ...


class S3ConditionalBlobStore:
    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        prefix: str = "",
        conflict_retries: int = 3,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket cannot be empty")
        if conflict_retries < 0:
            raise ValueError("conflict_retries cannot be negative")
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._conflict_retries = conflict_retries

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        metadata: JsonObject | None = None,
    ) -> BlobPutResult:
        normalized = LocalBlobStore._normalize_key(key)
        object_key = self._object_key(normalized)
        content_sha256 = hashlib.sha256(data).hexdigest()
        string_metadata = {str(name): str(value) for name, value in (metadata or {}).items()}
        string_metadata["arf-sha256"] = content_sha256
        for attempt in range(self._conflict_retries + 1):
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=object_key,
                    Body=data,
                    ContentLength=len(data),
                    IfNoneMatch="*",
                    Metadata=string_metadata,
                )
                info = self.head(normalized)
                if info is None:
                    raise RuntimeError(
                        f"blob {normalized!r} disappeared after conditional S3 write"
                    )
                return BlobPutResult(created=True, blob=info)
            except Exception as error:
                status, code = self._error_identity(error)
                if status == 409 or code == "ConditionalRequestConflict":
                    if attempt < self._conflict_retries:
                        continue
                    raise RuntimeError(
                        f"conditional S3 write for {normalized!r} remained in conflict"
                    ) from error
                if status == 412 or code in {"PreconditionFailed", "412"}:
                    existing = self.get(normalized)
                    if existing != data:
                        raise BlobConflictError(
                            f"blob {normalized!r} already exists with other content"
                        ) from error
                    info = self.head(normalized)
                    if info is None:
                        raise RuntimeError(
                            f"blob {normalized!r} disappeared after precondition failure"
                        ) from error
                    return BlobPutResult(created=False, blob=info)
                raise
        raise AssertionError("unreachable conditional write loop")

    def get(self, key: str) -> bytes:
        normalized = LocalBlobStore._normalize_key(key)
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=self._object_key(normalized),
            )
        except Exception as error:
            status, code = self._error_identity(error)
            if status == 404 or code in {"NoSuchKey", "NotFound", "404"}:
                raise KeyError(normalized) from error
            raise
        body = response["Body"]
        data = body.read()
        close = getattr(body, "close", None)
        if callable(close):
            close()
        return bytes(data)

    def head(self, key: str) -> BlobInfo | None:
        normalized = LocalBlobStore._normalize_key(key)
        try:
            response = self._client.head_object(
                Bucket=self._bucket,
                Key=self._object_key(normalized),
            )
        except Exception as error:
            status, code = self._error_identity(error)
            if status == 404 or code in {"NoSuchKey", "NotFound", "404"}:
                return None
            raise
        metadata = {str(key): str(value) for key, value in response.get("Metadata", {}).items()}
        last_modified = response.get("LastModified")
        return BlobInfo(
            key=normalized,
            size_bytes=int(response.get("ContentLength", 0)),
            etag=self._clean_etag(response.get("ETag")),
            content_sha256=metadata.get("arf-sha256"),
            last_modified=last_modified if isinstance(last_modified, datetime) else None,
            metadata=metadata,
        )

    def list(self, prefix: str = "") -> tuple[str, ...]:
        normalized_prefix = LocalBlobStore._normalize_prefix(prefix)
        object_prefix = self._object_key(normalized_prefix)
        keys = []
        continuation: str | None = None
        while True:
            request: dict[str, Any] = {
                "Bucket": self._bucket,
                "Prefix": object_prefix,
            }
            if continuation is not None:
                request["ContinuationToken"] = continuation
            response = self._client.list_objects_v2(**request)
            for item in response.get("Contents", []):
                object_key = str(item["Key"])
                keys.append(self._strip_prefix(object_key))
            if not response.get("IsTruncated"):
                break
            continuation = str(response["NextContinuationToken"])
        return tuple(sorted(keys))

    def delete_if_match(self, key: str, expected: BlobInfo) -> bool:
        normalized = LocalBlobStore._normalize_key(key)
        if expected.key != normalized:
            raise BlobConflictError("blob delete identity belongs to another key")
        current = self.head(normalized)
        if current is None:
            return False
        if not _blob_identity_matches(current, expected):
            raise BlobConflictError(f"blob {normalized!r} changed before deletion")
        if current.etag is None:
            raise RuntimeError(f"blob {normalized!r} has no ETag for conditional deletion")
        try:
            self._client.delete_object(
                Bucket=self._bucket,
                Key=self._object_key(normalized),
                IfMatch=current.etag,
            )
        except Exception as error:
            status, code = self._error_identity(error)
            if status == 404 or code in {"NoSuchKey", "NotFound", "404"}:
                return False
            if status == 412 or code in {"PreconditionFailed", "412"}:
                raise BlobConflictError(
                    f"blob {normalized!r} changed during conditional deletion"
                ) from error
            raise
        if self.head(normalized) is not None:
            raise RuntimeError(f"blob {normalized!r} remained after conditional deletion")
        return True

    def _object_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _strip_prefix(self, key: str) -> str:
        if not self._prefix:
            return key
        expected = f"{self._prefix}/"
        if not key.startswith(expected):
            raise ValueError(f"S3 key {key!r} is outside configured prefix")
        return key[len(expected) :]

    @staticmethod
    def _clean_etag(value: object) -> str | None:
        return str(value).strip('"') if value is not None else None

    @staticmethod
    def _error_identity(error: Exception) -> tuple[int | None, str | None]:
        response = getattr(error, "response", None)
        if not isinstance(response, dict):
            return None, None
        error_payload = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})
        code = error_payload.get("Code") if isinstance(error_payload, dict) else None
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
        return int(status) if status is not None else None, str(code) if code else None


def create_s3_blob_store(
    *,
    bucket: str,
    prefix: str = "",
    endpoint_url: str | None = None,
    region_name: str | None = None,
) -> S3ConditionalBlobStore:
    try:
        boto3 = importlib.import_module("boto3")
    except ImportError as error:
        raise RuntimeError("install agentic-rl-forge[object-store] for S3 support") from error
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
    )
    return S3ConditionalBlobStore(client, bucket=bucket, prefix=prefix)


def _blob_identity_matches(current: BlobInfo, expected: BlobInfo) -> bool:
    return (
        current.key == expected.key
        and current.size_bytes == expected.size_bytes
        and current.content_sha256 == expected.content_sha256
        and current.etag == expected.etag
        and current.last_modified == expected.last_modified
    )

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import shutil
import tarfile
import tempfile
import zlib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO, NamedTuple

from agentic_rl_forge.contracts import (
    RunArtifactArchiveReceipt,
    RunArtifactManifest,
    TrajectoryShardManifest,
)
from agentic_rl_forge.storage.blobs import LocalBlobStore
from agentic_rl_forge.storage.run_artifacts import RunArtifactBundle


class RunArtifactArchiveError(ValueError):
    pass


class _ArchiveEntry(NamedTuple):
    relative_path: str
    source_path: Path | None
    payload: bytes | None
    size_bytes: int
    content_digest: str


class RunArtifactArchive:
    _CHUNK_SIZE = 1024 * 1024
    _MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
    _MAX_SHARD_MANIFEST_BYTES = 128 * 1024 * 1024
    _RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def pack(
        self,
        manifest: RunArtifactManifest,
        output: Path | str,
    ) -> RunArtifactArchiveReceipt:
        output_path = Path(output)
        if not output_path.name.endswith(".tar.gz"):
            raise RunArtifactArchiveError("run artifact archive must use the .tar.gz suffix")
        manifest_path = self.root / RunArtifactBundle.manifest_relative_path(manifest.run_id)
        canonical_manifest = manifest.canonical_bytes() + b"\n"
        if not manifest_path.is_file() or manifest_path.read_bytes() != canonical_manifest:
            raise RunArtifactArchiveError(
                "stored run artifact manifest is missing or not canonical"
            )
        verification = RunArtifactBundle(self.root).verify(manifest)
        if not verification.valid:
            raise RunArtifactArchiveError("cannot pack an invalid run artifact bundle")

        entries = self._source_entries(manifest, canonical_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            # Do not duplicate a potentially long archive name in the staging
            # path; this otherwise makes valid output paths fail on Windows.
            prefix=".arf-archive-",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            self._write_archive(temporary, entries)
            temporary.chmod(0o644)
            content_digest, _ = self._digest_file(temporary)
            receipt = self.inspect(temporary, expected_sha256=content_digest)
            self._validate_existing_checksum(output_path, content_digest)
            self._publish_file(temporary, output_path, content_digest)
            LocalBlobStore(output_path.parent).put_if_absent(
                self.checksum_path(output_path).name,
                f"{content_digest}\n".encode("ascii"),
            )
            output_path.chmod(0o644)
            self.checksum_path(output_path).chmod(0o644)
            return receipt
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def inspect(
        cls,
        archive: Path | str,
        *,
        expected_sha256: str,
    ) -> RunArtifactArchiveReceipt:
        archive_path = Path(archive)
        expected = cls.normalize_digest(expected_sha256)
        content_digest, size_bytes = cls._digest_file(archive_path)
        if content_digest != expected:
            raise RunArtifactArchiveError("run artifact archive SHA-256 does not match")
        manifest, member_count = cls._read_archive(archive_path, destination=None)
        return RunArtifactArchiveReceipt(
            archive_id=RunArtifactArchiveReceipt.expected_archive_id(content_digest),
            run_id=manifest.run_id,
            manifest_id=manifest.manifest_id,
            content_digest=content_digest,
            size_bytes=size_bytes,
            member_count=member_count,
        )

    @classmethod
    def unpack(
        cls,
        archive: Path | str,
        destination: Path | str,
        *,
        expected_sha256: str,
    ) -> RunArtifactArchiveReceipt:
        archive_path = Path(archive)
        destination_path = Path(destination)
        if destination_path.exists():
            raise RunArtifactArchiveError("archive destination already exists")
        expected = cls.normalize_digest(expected_sha256)
        content_digest, size_bytes = cls._digest_file(archive_path)
        if content_digest != expected:
            raise RunArtifactArchiveError("run artifact archive SHA-256 does not match")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                dir=destination_path.parent,
                prefix=".arf-unpack-",
            )
        )
        try:
            manifest, member_count = cls._read_archive(archive_path, destination=temporary)
            verification = RunArtifactBundle(temporary).verify(manifest)
            if not verification.valid:
                raise RunArtifactArchiveError(
                    "extracted run artifact bundle failed semantic verification"
                )
            for directory in (path for path in temporary.rglob("*") if path.is_dir()):
                directory.chmod(0o755)
            temporary.chmod(0o755)
            if destination_path.exists():
                raise RunArtifactArchiveError("archive destination appeared during extraction")
            os.rename(temporary, destination_path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        return RunArtifactArchiveReceipt(
            archive_id=RunArtifactArchiveReceipt.expected_archive_id(content_digest),
            run_id=manifest.run_id,
            manifest_id=manifest.manifest_id,
            content_digest=content_digest,
            size_bytes=size_bytes,
            member_count=member_count,
        )

    @staticmethod
    def checksum_path(archive: Path | str) -> Path:
        path = Path(archive)
        return path.with_name(f"{path.name}.sha256")

    @classmethod
    def read_checksum(cls, path: Path | str) -> str:
        checksum_path = Path(path)
        try:
            payload = checksum_path.read_bytes()
        except OSError as error:
            raise RunArtifactArchiveError(f"cannot read archive checksum: {error}") from error
        if len(payload) != 65 or not payload.endswith(b"\n"):
            raise RunArtifactArchiveError("archive checksum file is not canonical")
        try:
            digest = payload[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise RunArtifactArchiveError("archive checksum is not ASCII") from error
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RunArtifactArchiveError("archive checksum file is not canonical")
        return digest

    @staticmethod
    def normalize_digest(value: str) -> str:
        digest = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RunArtifactArchiveError("expected archive SHA-256 must contain 64 hex digits")
        return digest

    def _source_entries(
        self,
        manifest: RunArtifactManifest,
        canonical_manifest: bytes,
    ) -> tuple[_ArchiveEntry, ...]:
        manifest_path = RunArtifactBundle.manifest_relative_path(manifest.run_id)
        entries = [
            _ArchiveEntry(
                relative_path=manifest_path,
                source_path=None,
                payload=canonical_manifest,
                size_bytes=len(canonical_manifest),
                content_digest=hashlib.sha256(canonical_manifest).hexdigest(),
            )
        ]
        for artifact in sorted(manifest.artifacts, key=lambda item: item.relative_path):
            entries.append(
                _ArchiveEntry(
                    relative_path=artifact.relative_path,
                    source_path=self.root / artifact.relative_path,
                    payload=None,
                    size_bytes=artifact.size_bytes,
                    content_digest=artifact.content_digest,
                )
            )
        shard_manifest_ref = next(
            item for item in manifest.artifacts if item.name == "shard_manifest"
        )
        shard_manifest = TrajectoryShardManifest.model_validate_json(
            (self.root / shard_manifest_ref.relative_path).read_bytes()
        )
        shard_prefix = PurePosixPath("shards", "runs", manifest.run_id)
        for shard in sorted(shard_manifest.shards, key=lambda item: item.relative_path):
            relative_path = (shard_prefix / shard.relative_path).as_posix()
            entries.append(
                _ArchiveEntry(
                    relative_path=relative_path,
                    source_path=self.root / relative_path,
                    payload=None,
                    size_bytes=shard.size_bytes,
                    content_digest=shard.content_digest,
                )
            )
        names = [entry.relative_path for entry in entries]
        if len(names) != len(set(names)):
            raise RunArtifactArchiveError("run artifact archive member paths are not unique")
        return tuple(entries)

    @classmethod
    def _write_archive(cls, output: Path, entries: Sequence[_ArchiveEntry]) -> None:
        try:
            with (
                output.open("wb") as raw_output,
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=9,
                    fileobj=raw_output,
                    mtime=0,
                ) as compressed,
                tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive,
            ):
                for entry in entries:
                    info = cls._canonical_tar_info(entry.relative_path, entry.size_bytes)
                    if entry.payload is not None:
                        archive.addfile(info, io.BytesIO(entry.payload))
                    elif entry.source_path is not None:
                        with entry.source_path.open("rb") as source:
                            archive.addfile(info, source)
                    else:
                        raise AssertionError("archive entry has no payload source")
                raw_output.flush()
                os.fsync(raw_output.fileno())
        except (OSError, tarfile.TarError, ValueError) as error:
            raise RunArtifactArchiveError(f"cannot build canonical run archive: {error}") from error

    @staticmethod
    def _canonical_tar_info(name: str, size_bytes: int) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name=name)
        info.size = size_bytes
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.type = tarfile.REGTYPE
        return info

    @classmethod
    def _read_archive(
        cls,
        archive_path: Path,
        *,
        destination: Path | None,
    ) -> tuple[RunArtifactManifest, int]:
        try:
            with tarfile.open(archive_path, mode="r|gz") as archive:
                expected_offset = 0
                first = archive.next()
                if first is None:
                    raise RunArtifactArchiveError("run artifact archive is empty")
                cls._validate_member_metadata(first, expected_offset=expected_offset)
                expected_offset = cls._next_offset(first)
                manifest_run_id = cls._run_id_from_manifest_path(first.name)
                if first.size > cls._MAX_ARTIFACT_MANIFEST_BYTES:
                    raise RunArtifactArchiveError("run artifact manifest exceeds the safety limit")
                manifest_payload = cls._consume_member(
                    archive,
                    first,
                    expected_name=first.name,
                    expected_size=first.size,
                    expected_digest=None,
                    destination=destination,
                    capture=True,
                )
                if manifest_payload is None:
                    raise AssertionError("manifest payload was not captured")
                manifest = RunArtifactManifest.model_validate_json(manifest_payload)
                expected_manifest_path = RunArtifactBundle.manifest_relative_path(manifest.run_id)
                if first.name != expected_manifest_path or manifest.run_id != manifest_run_id:
                    raise RunArtifactArchiveError(
                        "archive manifest path does not match its run identity"
                    )
                if manifest_payload != manifest.canonical_bytes() + b"\n":
                    raise RunArtifactArchiveError("archive manifest bytes are not canonical")

                member_count = 1
                shard_manifest_payload: bytes | None = None
                for artifact in sorted(manifest.artifacts, key=lambda item: item.relative_path):
                    member = archive.next()
                    if member is None:
                        raise RunArtifactArchiveError(
                            f"archive is missing member {artifact.relative_path!r}"
                        )
                    cls._validate_member_metadata(member, expected_offset=expected_offset)
                    expected_offset = cls._next_offset(member)
                    capture = artifact.name == "shard_manifest"
                    if capture and artifact.size_bytes > cls._MAX_SHARD_MANIFEST_BYTES:
                        raise RunArtifactArchiveError("shard manifest exceeds the safety limit")
                    payload = cls._consume_member(
                        archive,
                        member,
                        expected_name=artifact.relative_path,
                        expected_size=artifact.size_bytes,
                        expected_digest=artifact.content_digest,
                        destination=destination,
                        capture=capture,
                    )
                    if capture:
                        shard_manifest_payload = payload
                    member_count += 1

                if shard_manifest_payload is None:
                    raise RunArtifactArchiveError("archive did not provide its shard manifest")
                shard_manifest = TrajectoryShardManifest.model_validate_json(shard_manifest_payload)
                if (
                    shard_manifest.run_id != manifest.run_id
                    or shard_manifest.manifest_id != manifest.shard_manifest_id
                ):
                    raise RunArtifactArchiveError(
                        "archive shard manifest does not match artifact manifest"
                    )
                shard_prefix = PurePosixPath("shards", "runs", manifest.run_id)
                for shard in sorted(
                    shard_manifest.shards,
                    key=lambda item: item.relative_path,
                ):
                    expected_name = (shard_prefix / shard.relative_path).as_posix()
                    member = archive.next()
                    if member is None:
                        raise RunArtifactArchiveError(
                            f"archive is missing member {expected_name!r}"
                        )
                    cls._validate_member_metadata(member, expected_offset=expected_offset)
                    expected_offset = cls._next_offset(member)
                    cls._consume_member(
                        archive,
                        member,
                        expected_name=expected_name,
                        expected_size=shard.size_bytes,
                        expected_digest=shard.content_digest,
                        destination=destination,
                        capture=False,
                    )
                    member_count += 1

                unexpected = archive.next()
                if unexpected is not None:
                    raise RunArtifactArchiveError(
                        f"archive contains unexpected member {unexpected.name!r}"
                    )
                return manifest, member_count
        except RunArtifactArchiveError:
            raise
        except (OSError, EOFError, tarfile.TarError, ValueError, zlib.error) as error:
            raise RunArtifactArchiveError(f"invalid run artifact archive: {error}") from error

    @classmethod
    def _consume_member(
        cls,
        archive: tarfile.TarFile,
        member: tarfile.TarInfo,
        *,
        expected_name: str,
        expected_size: int,
        expected_digest: str | None,
        destination: Path | None,
        capture: bool,
    ) -> bytes | None:
        if member.name != expected_name:
            raise RunArtifactArchiveError(
                f"archive member order mismatch: expected {expected_name!r}, got {member.name!r}"
            )
        if member.size != expected_size:
            raise RunArtifactArchiveError(f"archive member {expected_name!r} has the wrong size")
        source = archive.extractfile(member)
        if source is None:
            raise RunArtifactArchiveError(f"archive member {expected_name!r} has no file data")
        target = cls._target_path(destination, expected_name) if destination is not None else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
        captured = bytearray() if capture else None
        content_hash = hashlib.sha256()
        size = 0
        output: BinaryIO | None = target.open("xb") if target is not None else None
        try:
            while True:
                chunk = source.read(cls._CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                content_hash.update(chunk)
                if output is not None:
                    output.write(chunk)
                if captured is not None:
                    captured.extend(chunk)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output is not None:
                output.close()
            source.close()
        if size != expected_size:
            raise RunArtifactArchiveError(f"archive member {expected_name!r} is truncated")
        digest = content_hash.hexdigest()
        if expected_digest is not None and digest != expected_digest:
            raise RunArtifactArchiveError(
                f"archive member {expected_name!r} failed SHA-256 verification"
            )
        if target is not None:
            target.chmod(0o644)
        return bytes(captured) if captured is not None else None

    @classmethod
    def _validate_member_metadata(
        cls,
        member: tarfile.TarInfo,
        *,
        expected_offset: int,
    ) -> None:
        if not member.isreg() or member.type != tarfile.REGTYPE:
            raise RunArtifactArchiveError("archive members must be regular files")
        if (
            member.mode != 0o644
            or member.uid != 0
            or member.gid != 0
            or member.uname != ""
            or member.gname != ""
            or member.mtime != 0
            or member.linkname != ""
            or member.pax_headers
        ):
            raise RunArtifactArchiveError(
                f"archive member {member.name!r} has noncanonical metadata"
            )
        if member.offset != expected_offset or member.offset_data != member.offset + 512:
            raise RunArtifactArchiveError(
                f"archive member {member.name!r} has a noncanonical header layout"
            )

    @staticmethod
    def _next_offset(member: tarfile.TarInfo) -> int:
        blocks = (member.size + 511) // 512
        return member.offset_data + blocks * 512

    @classmethod
    def _run_id_from_manifest_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or path.is_absolute()
            or len(path.parts) != 3
            or path.parts[0] != "runs"
            or path.parts[2] != "artifact-manifest.json"
            or not cls._RUN_ID_PATTERN.fullmatch(path.parts[1])
        ):
            raise RunArtifactArchiveError(
                "first archive member must be runs/<run_id>/artifact-manifest.json"
            )
        return path.parts[1]

    @staticmethod
    def _target_path(root: Path, relative_path: str) -> Path:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise RunArtifactArchiveError("archive member path escapes the destination")
        return root.joinpath(*path.parts)

    @staticmethod
    def _digest_file(path: Path) -> tuple[str, int]:
        content_hash = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(RunArtifactArchive._CHUNK_SIZE):
                    size += len(chunk)
                    content_hash.update(chunk)
        except OSError as error:
            raise RunArtifactArchiveError(f"cannot read run artifact archive: {error}") from error
        return content_hash.hexdigest(), size

    @classmethod
    def _publish_file(cls, source: Path, target: Path, content_digest: str) -> None:
        try:
            os.link(source, target)
        except FileExistsError:
            if not target.is_file() or cls._digest_file(target)[0] != content_digest:
                raise RunArtifactArchiveError(
                    "archive output already exists with different content"
                ) from None
        except OSError as error:
            raise RunArtifactArchiveError(
                f"cannot publish run artifact archive: {error}"
            ) from error

    @classmethod
    def _validate_existing_checksum(cls, archive: Path, content_digest: str) -> None:
        path = cls.checksum_path(archive)
        if path.exists() and cls.read_checksum(path) != content_digest:
            raise RunArtifactArchiveError(
                "archive checksum output already exists with different content"
            )

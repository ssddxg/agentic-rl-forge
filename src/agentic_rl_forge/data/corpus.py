from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import orjson
from pydantic import Field

from agentic_rl_forge.contracts import ContractModel

DEFAULT_TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".rst")
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


@dataclass(frozen=True, slots=True)
class _CorpusSummary:
    document_count: int
    source_count: int
    total_characters: int
    min_characters: int
    max_characters: int
    average_characters: float


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    changed_at_ns: int


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    document_id: str
    contents: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.contents, str) or not self.contents.strip():
            raise ValueError("contents must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


class CorpusInspection(ContractModel):
    path: str = Field(min_length=1)
    file_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_sha256: str = Field(pattern=_SHA256_PATTERN)
    document_count: int = Field(ge=1)
    source_count: int = Field(ge=1)
    total_characters: int = Field(ge=1)
    min_characters: int = Field(ge=1)
    max_characters: int = Field(ge=1)
    average_characters: float = Field(gt=0)


class CorpusBuildReport(ContractModel):
    source: str = Field(min_length=1)
    output: str = Field(min_length=1)
    file_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_sha256: str = Field(pattern=_SHA256_PATTERN)
    document_count: int = Field(ge=1)
    source_count: int = Field(ge=1)
    total_characters: int = Field(ge=1)
    min_characters: int = Field(ge=1)
    max_characters: int = Field(ge=1)
    average_characters: float = Field(gt=0)
    scanned_file_count: int = Field(ge=1)
    included_file_count: int = Field(ge=1)
    skipped_empty_file_count: int = Field(ge=0)
    duplicate_chunk_count: int = Field(ge=0)


def load_corpus_jsonl(path: Path) -> tuple[CorpusDocument, ...]:
    payload = _read_bytes(path)
    text = _decode_utf8(payload, path)
    documents: list[CorpusDocument] = []
    seen_ids: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = orjson.loads(line)
        except orjson.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        if "id" not in value:
            raise ValueError(f"{path}:{line_number} does not contain 'id'")
        document_id = _document_id(value["id"], path=path, line_number=line_number)
        if document_id in seen_ids:
            first_line = seen_ids[document_id]
            raise ValueError(
                f"{path}:{line_number} duplicates document id {document_id!r} "
                f"from line {first_line}"
            )
        if "contents" not in value:
            raise ValueError(f"{path}:{line_number} does not contain 'contents'")
        contents = value["contents"]
        if not isinstance(contents, str) or not contents.strip():
            raise ValueError(f"{path}:{line_number} has non-empty string 'contents' requirement")
        metadata = {key: item for key, item in value.items() if key not in {"id", "contents"}}
        try:
            document = CorpusDocument(document_id, contents, metadata)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} has invalid metadata: {error}") from error
        documents.append(document)
        seen_ids[document_id] = line_number
    if not documents:
        raise ValueError(f"{path}:1 corpus is empty")
    return tuple(documents)


def corpus_digest(documents: Iterable[CorpusDocument]) -> str:
    records: list[tuple[str, bytes]] = []
    seen_ids: set[str] = set()
    for document in documents:
        if document.document_id in seen_ids:
            raise ValueError(f"duplicate document id {document.document_id!r}")
        seen_ids.add(document.document_id)
        record = {
            "contents": document.contents,
            "id": document.document_id,
            "metadata": _thaw_json(document.metadata),
        }
        records.append(
            (
                document.document_id,
                orjson.dumps(record, option=orjson.OPT_SORT_KEYS),
            )
        )
    if not records:
        raise ValueError("corpus requires at least one document")
    digest = hashlib.sha256()
    for _, record_bytes in sorted(records, key=lambda item: (item[0], item[1])):
        digest.update(record_bytes)
        digest.update(b"\n")
    return digest.hexdigest()


def inspect_corpus(path: Path) -> CorpusInspection:
    payload = _read_bytes(path)
    documents = load_corpus_jsonl(path)
    summary = _summarize(documents)
    return CorpusInspection(
        path=str(path.resolve()),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        corpus_sha256=corpus_digest(documents),
        document_count=summary.document_count,
        source_count=summary.source_count,
        total_characters=summary.total_characters,
        min_characters=summary.min_characters,
        max_characters=summary.max_characters,
        average_characters=summary.average_characters,
    )


def build_text_corpus(
    source: Path,
    output: Path,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
    extensions: Sequence[str] = DEFAULT_TEXT_EXTENSIONS,
    force: bool = False,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> CorpusBuildReport:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    allowed_extensions = _normalize_extensions(extensions)
    source_path = source.resolve(strict=True)
    output_path = output.parent.resolve() / output.name
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")

    candidates = _text_files(source_path, allowed_extensions, output_path=output_path)
    if not candidates:
        raise ValueError(f"no supported text files found under {source_path}")

    documents: list[CorpusDocument] = []
    seen_contents: set[str] = set()
    skipped_empty = 0
    duplicate_chunks = 0
    included_sources: set[str] = set()
    root = source_path if source_path.is_dir() else source_path.parent
    for path in candidates:
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(f"{path} exceeds max_file_bytes ({size} > {max_file_bytes})")
        text = _decode_utf8(_read_bytes(path), path)
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            skipped_empty += 1
            continue
        relative_path = path.relative_to(root).as_posix()
        chunks = _chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, contents in enumerate(chunks):
            if contents in seen_contents:
                duplicate_chunks += 1
                continue
            seen_contents.add(contents)
            chunk_identity = hashlib.sha256()
            chunk_identity.update(relative_path.encode("utf-8"))
            chunk_identity.update(b"\0")
            chunk_identity.update(str(chunk_index).encode("ascii"))
            chunk_identity.update(b"\0")
            chunk_identity.update(contents.encode("utf-8"))
            documents.append(
                CorpusDocument(
                    document_id=f"chunk_{chunk_identity.hexdigest()[:24]}",
                    contents=contents,
                    metadata={
                        "source": relative_path,
                        "title": path.stem,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                    },
                )
            )
            included_sources.add(relative_path)
    if not documents:
        raise ValueError(f"no non-empty unique text found under {source_path}")

    output_payload = b"".join(_document_jsonl(document) for document in documents)
    _atomic_publish(output_path, output_payload, force=force)
    inspection = inspect_corpus(output_path)
    return CorpusBuildReport(
        source=str(source_path),
        output=str(output_path),
        file_sha256=inspection.file_sha256,
        corpus_sha256=inspection.corpus_sha256,
        document_count=inspection.document_count,
        source_count=inspection.source_count,
        total_characters=inspection.total_characters,
        min_characters=inspection.min_characters,
        max_characters=inspection.max_characters,
        average_characters=inspection.average_characters,
        scanned_file_count=len(candidates),
        included_file_count=len(included_sources),
        skipped_empty_file_count=skipped_empty,
        duplicate_chunk_count=duplicate_chunks,
    )


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}:1 cannot be read: {error}") from error


def _decode_utf8(payload: bytes, path: Path) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        line_number = payload[: error.start].count(b"\n") + 1
        raise ValueError(f"{path}:{line_number} is not valid UTF-8") from error


def _document_id(value: Any, *, path: Path, line_number: int) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{path}:{line_number} has an empty document id")
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return orjson.dumps(value).decode("ascii")
    raise ValueError(f"{path}:{line_number} has a non-scalar document id")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        frozen[key] = _freeze_json(item)
    return MappingProxyType(frozen)


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError(f"metadata contains unsupported value type {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _summarize(documents: Sequence[CorpusDocument]) -> _CorpusSummary:
    lengths = [len(document.contents) for document in documents]
    sources = {_source_identity(document) for document in documents}
    total = sum(lengths)
    return _CorpusSummary(
        document_count=len(documents),
        source_count=len(sources),
        total_characters=total,
        min_characters=min(lengths),
        max_characters=max(lengths),
        average_characters=total / len(lengths),
    )


def _source_identity(document: CorpusDocument) -> str:
    source = document.metadata.get("source")
    if isinstance(source, str) and source.strip():
        return f"source:{source}"
    if isinstance(source, bool | int | float):
        return f"source:{source}"
    return f"document:{document.document_id}"


def _normalize_extensions(extensions: Sequence[str]) -> frozenset[str]:
    normalized = {
        (extension if extension.startswith(".") else f".{extension}").casefold()
        for extension in extensions
        if extension.strip()
    }
    if not normalized:
        raise ValueError("extensions must contain at least one file extension")
    return frozenset(normalized)


def _text_files(
    source: Path,
    extensions: frozenset[str],
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    if source.is_symlink() or _is_hidden(source.name):
        return ()
    if source.is_file():
        if source.suffix.casefold() not in extensions or source == output_path:
            return ()
        return (source,)
    if not source.is_dir():
        raise ValueError(f"source is not a file or directory: {source}")
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _is_hidden(name) and not (directory_path / name).is_symlink()
        )
        for name in sorted(file_names):
            path = directory_path / name
            if (
                _is_hidden(name)
                or path.is_symlink()
                or path.suffix.casefold() not in extensions
                or path.resolve() == output_path
            ):
                continue
            files.append(path.resolve(strict=True))
    return tuple(sorted(files, key=lambda path: path.relative_to(source).as_posix()))


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _chunks(text: str, *, chunk_size: int, chunk_overlap: int) -> tuple[str, ...]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - chunk_overlap
    return tuple(chunks)


def _document_jsonl(document: CorpusDocument) -> bytes:
    record = {
        "id": document.document_id,
        "contents": document.contents,
        **_thaw_json(document.metadata),
    }
    return orjson.dumps(record, option=orjson.OPT_SORT_KEYS) + b"\n"


def _atomic_publish(path: Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".corpus-", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        if force:
            os.replace(temporary_path, path)
        else:
            _publish_without_overwrite(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_without_overwrite(temporary_path: Path, path: Path) -> None:
    try:
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"output already exists: {path}") from error
    except OSError:
        _publish_with_placeholder(temporary_path, path)
    else:
        temporary_path.unlink()


def _publish_with_placeholder(temporary_path: Path, path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"output already exists: {path}") from error

    try:
        placeholder_identity = _file_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)

    try:
        os.replace(temporary_path, path)
    except BaseException:
        _unlink_if_unchanged(path, placeholder_identity)
        raise


def _file_identity(status: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        changed_at_ns=status.st_ctime_ns,
    )


def _unlink_if_unchanged(path: Path, expected: _FileIdentity) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(current) != expected:
            return
        path.unlink()
    except OSError:
        return

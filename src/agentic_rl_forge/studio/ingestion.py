from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import BinaryIO, ClassVar
from uuid import uuid4

from agentic_rl_forge.data import CorpusDocument
from agentic_rl_forge.studio.models import Source
from agentic_rl_forge.studio.paths import StudioPaths
from agentic_rl_forge.studio.repository import StudioRepository

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 120


class OptionalDocumentDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExtractedText:
    title: str
    text: str


class _ReadableHTMLParser(HTMLParser):
    _BLOCK_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            "address",
            "article",
            "aside",
            "br",
            "div",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "li",
            "main",
            "p",
            "section",
            "td",
            "th",
            "tr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif normalized == "title":
            self._in_title = True
        if normalized in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif normalized == "title":
            self._in_title = False
        if normalized in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        self.parts.append(data)
        if self._in_title:
            self.title_parts.append(data)


class DocumentIngestor:
    _ALLOWED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf", ".docx"}
    )

    def __init__(
        self,
        paths: StudioPaths,
        repository: StudioRepository,
        *,
        max_file_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_extracted_characters: int = 8_000_000,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        if max_extracted_characters < 1:
            raise ValueError("max_extracted_characters must be positive")
        if max_total_bytes < 1:
            raise ValueError("max_total_bytes must be positive")
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        self._paths = paths.ensure()
        self._repository = repository
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._max_extracted_characters = max_extracted_characters
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def allowed_extensions(self) -> frozenset[str]:
        return self._ALLOWED_EXTENSIONS

    @property
    def max_file_bytes(self) -> int:
        return self._max_file_bytes

    @property
    def max_total_bytes(self) -> int:
        return self._max_total_bytes

    def store_upload(
        self,
        knowledge_base_id: str,
        filename: str,
        chunks: Iterable[bytes],
        media_type: str = "application/octet-stream",
    ) -> Source:
        self._repository.require_knowledge_base(knowledge_base_id)
        safe_name = safe_upload_filename(filename)
        extension = Path(safe_name).suffix.casefold()
        if extension not in self._ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(self._ALLOWED_EXTENSIONS))
            raise ValueError(
                f"unsupported document type {extension or '(none)'}; allowed: {allowed}"
            )
        stored_name = f"{uuid4().hex}{extension}"
        directory = self._paths.source_directory(knowledge_base_id)
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".upload-", suffix=".tmp", dir=directory
        )
        temporary_path = Path(temporary_name)
        final_path = self._paths.source_path(knowledge_base_id, stored_name)
        size = 0
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("upload chunks must be bytes")
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self._max_file_bytes:
                        raise ValueError(
                            f"file exceeds the {self._max_file_bytes} byte upload limit"
                        )
                    destination.write(chunk)
                    digest.update(chunk)
                if size == 0:
                    raise ValueError("uploaded file is empty")
                destination.flush()
                os.fsync(destination.fileno())
            content_sha256 = digest.hexdigest()
            if self._repository.find_source_by_digest(knowledge_base_id, content_sha256):
                raise ValueError("this file is already present in the knowledge base")
            used_bytes = self._repository.total_source_bytes()
            if used_bytes + size > self._max_total_bytes:
                raise ValueError(
                    f"Studio storage limit exceeded ({used_bytes + size} > "
                    f"{self._max_total_bytes} bytes)"
                )
            os.replace(temporary_path, final_path)
            try:
                return self._repository.create_source(
                    knowledge_base_id,
                    original_name=safe_name,
                    stored_name=stored_name,
                    media_type=media_type or "application/octet-stream",
                    size_bytes=size,
                    content_sha256=content_sha256,
                    max_total_bytes=self._max_total_bytes,
                )
            except Exception:
                final_path.unlink(missing_ok=True)
                raise
        finally:
            temporary_path.unlink(missing_ok=True)

    def store_upload_file(
        self,
        knowledge_base_id: str,
        filename: str,
        stream: BinaryIO,
        *,
        media_type: str = "application/octet-stream",
        read_size: int = 64 * 1024,
    ) -> Source:
        if read_size < 1:
            raise ValueError("read_size must be positive")

        def chunks() -> Iterator[bytes]:
            while payload := stream.read(read_size):
                yield payload

        return self.store_upload(
            knowledge_base_id,
            filename,
            chunks(),
            media_type=media_type,
        )

    def delete_source_file(self, source: Source) -> None:
        path = self._paths.source_path(source.knowledge_base_id, source.stored_name)
        path.unlink(missing_ok=True)

    def delete_knowledge_base_files(self, knowledge_base_id: str) -> None:
        directory = self._paths.source_directory(knowledge_base_id)
        _remove_managed_directory(self._paths.uploads, directory)

    def extract(self, path: Path) -> ExtractedText:
        extension = path.suffix.casefold()
        if extension in {".txt", ".md", ".markdown", ".rst"}:
            result = ExtractedText(path.stem, _decode_text(path.read_bytes(), path))
        elif extension in {".html", ".htm"}:
            result = _extract_html(path)
        elif extension == ".pdf":
            result = _extract_pdf(path, character_limit=self._max_extracted_characters)
        elif extension == ".docx":
            result = _extract_docx(path, character_limit=self._max_extracted_characters)
        else:
            raise ValueError(f"unsupported document type: {extension or '(none)'}")
        cleaned = _normalize_extracted_text(result.text)
        if not cleaned:
            raise ValueError(f"document contains no readable text: {path.name}")
        if len(cleaned) > self._max_extracted_characters:
            raise ValueError(
                "extracted document exceeds the "
                f"{self._max_extracted_characters} character safety limit"
            )
        return ExtractedText(result.title.strip() or path.stem, cleaned)

    def ingest_source(self, source_id: str) -> tuple[CorpusDocument, ...]:
        source = self._repository.require_source(source_id)
        self._repository.update_source_status(source_id, "processing")
        path = self._paths.source_path(source.knowledge_base_id, source.stored_name)
        try:
            extracted = self.extract(path)
            chunks = _chunks(
                extracted.text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            documents = tuple(
                _corpus_document(source, extracted.title, index, len(chunks), contents)
                for index, contents in enumerate(chunks)
            )
            self._repository.replace_source_documents(source_id, documents)
            return documents
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:2000]
            self._repository.update_source_status(
                source_id,
                "failed",
                error=message,
            )
            raise


def safe_upload_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKC", filename).replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    basename = "".join(
        character for character in basename if not unicodedata.category(character).startswith("C")
    )
    basename = re.sub(r"\s+", " ", basename).strip(" .")
    if not basename or basename in {".", ".."}:
        raise ValueError("filename is empty or unsafe")
    if len(basename) > 255:
        raise ValueError("filename must not exceed 255 characters")
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if Path(basename).stem.casefold() in reserved:
        raise ValueError("filename is reserved by Windows")
    return basename


def _decode_text(payload: bytes, path: Path) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} is not valid UTF-8") from error


def _extract_html(path: Path) -> ExtractedText:
    parser = _ReadableHTMLParser()
    parser.feed(_decode_text(path.read_bytes(), path))
    return ExtractedText(" ".join(parser.title_parts).strip() or path.stem, " ".join(parser.parts))


def _extract_pdf(path: Path, *, character_limit: int) -> ExtractedText:
    try:
        pypdf = importlib.import_module("pypdf")
    except ImportError as error:
        raise OptionalDocumentDependencyError(
            "PDF import requires the optional 'pypdf' package; install pypdf to enable it"
        ) from error
    try:
        reader = pypdf.PdfReader(str(path))
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            total += len(page_text)
            if total > character_limit:
                raise ValueError("PDF extracted text exceeds the character safety limit")
            parts.append(page_text)
        text = "\n\n".join(parts)
        title = str((reader.metadata or {}).get("/Title") or path.stem)
    except Exception as error:
        raise ValueError(f"could not read PDF document {path.name}: {error}") from error
    return ExtractedText(title, text)


def _extract_docx(path: Path, *, character_limit: int) -> ExtractedText:
    try:
        docx = importlib.import_module("docx")
    except ImportError as error:
        raise OptionalDocumentDependencyError(
            "DOCX import requires the optional 'python-docx' package; "
            "install python-docx to enable it"
        ) from error
    try:
        document = docx.Document(str(path))
        paragraphs: list[str] = []
        total = 0
        for paragraph in document.paragraphs:
            paragraph_text = str(paragraph.text)
            total += len(paragraph_text)
            if total > character_limit:
                raise ValueError("DOCX extracted text exceeds the character safety limit")
            paragraphs.append(paragraph_text)
        for table in document.tables:
            for row in table.rows:
                row_text = "\t".join(str(cell.text) for cell in row.cells)
                total += len(row_text)
                if total > character_limit:
                    raise ValueError("DOCX extracted text exceeds the character safety limit")
                paragraphs.append(row_text)
        title = str(document.core_properties.title or path.stem)
    except Exception as error:
        raise ValueError(f"could not read DOCX document {path.name}: {error}") from error
    return ExtractedText(title, "\n".join(paragraphs))


def _normalize_extracted_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r", "\n").split("\n")]
    compact: list[str] = []
    previous_blank = True
    for line in lines:
        if line:
            compact.append(line)
            previous_blank = False
        elif not previous_blank:
            compact.append("")
            previous_blank = True
    return "\n".join(compact).strip()


def _chunks(text: str, *, chunk_size: int, chunk_overlap: int) -> tuple[str, ...]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - chunk_overlap)
    return tuple(chunks)


def _corpus_document(
    source: Source,
    title: str,
    chunk_index: int,
    chunk_count: int,
    contents: str,
) -> CorpusDocument:
    digest = hashlib.sha256()
    digest.update(source.id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(chunk_index).encode("ascii"))
    digest.update(b"\0")
    digest.update(contents.encode("utf-8"))
    return CorpusDocument(
        document_id=f"doc_{digest.hexdigest()[:32]}",
        contents=contents,
        metadata={
            "source_id": source.id,
            "source": source.original_name,
            "source_name": source.original_name,
            "title": title,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
        },
    )


def _remove_managed_directory(parent: Path, directory: Path) -> None:
    parent = parent.resolve()
    resolved = directory.resolve()
    if resolved.parent != parent:
        raise ValueError("managed directory escapes the Studio data directory")
    if directory.exists():
        shutil.rmtree(directory)

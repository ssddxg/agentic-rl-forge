from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import orjson

from agentic_rl_forge.contracts import new_id, utc_now
from agentic_rl_forge.data import CorpusDocument
from agentic_rl_forge.studio.models import (
    IndexStats,
    IngestionJob,
    JobKind,
    JobStatus,
    KnowledgeBase,
    ModelSettings,
    Source,
    SourceStatus,
)

_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "qwen2.5:7b"
_INTERRUPTED_WORK_ERROR = "上次运行中断, 请重新处理。"


class StudioRepository:
    """Thread-safe local SQLite state for Studio.

    The optional API key is stored only in this local database. Public settings methods
    return a boolean flag instead of the secret itself.
    """

    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
        with suppress(OSError):
            database.chmod(0o600)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> StudioRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBase:
        clean_name = _required_text(name, "name", maximum=120)
        clean_description = _optional_text(description, "description", maximum=2000)
        now = utc_now()
        knowledge_base = KnowledgeBase(
            id=new_id("kb"),
            name=clean_name,
            description=clean_description,
            revision=0,
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO knowledge_bases
                (id, name, description, revision, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    knowledge_base.id,
                    knowledge_base.name,
                    knowledge_base.description,
                    knowledge_base.revision,
                    _datetime_text(now),
                    _datetime_text(now),
                ),
            )
        return knowledge_base

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBase | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM knowledge_bases WHERE id = ?", (knowledge_base_id,)
            ).fetchone()
        return _knowledge_base(row) if row is not None else None

    def require_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBase:
        knowledge_base = self.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise KeyError(f"knowledge base not found: {knowledge_base_id}")
        return knowledge_base

    def list_knowledge_bases(self) -> tuple[KnowledgeBase, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM knowledge_bases ORDER BY updated_at DESC, id"
            ).fetchall()
        return tuple(_knowledge_base(row) for row in rows)

    def update_knowledge_base(
        self,
        knowledge_base_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBase:
        current = self.require_knowledge_base(knowledge_base_id)
        next_name = current.name if name is None else _required_text(name, "name", maximum=120)
        next_description = (
            current.description
            if description is None
            else _optional_text(description, "description", maximum=2000)
        )
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE knowledge_bases SET name = ?, description = ?, updated_at = ? WHERE id = ?",
                (next_name, next_description, _datetime_text(now), knowledge_base_id),
            )
        return current.model_copy(
            update={"name": next_name, "description": next_description, "updated_at": now}
        )

    def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM knowledge_bases WHERE id = ?", (knowledge_base_id,)
            )
        return cursor.rowcount > 0

    def create_source(
        self,
        knowledge_base_id: str,
        *,
        original_name: str,
        stored_name: str,
        media_type: str,
        size_bytes: int,
        content_sha256: str | None = None,
        max_total_bytes: int | None = None,
    ) -> Source:
        self.require_knowledge_base(knowledge_base_id)
        if size_bytes < 1:
            raise ValueError("size_bytes must be positive")
        if content_sha256 is not None and (
            len(content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in content_sha256)
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if max_total_bytes is not None and max_total_bytes < 1:
            raise ValueError("max_total_bytes must be positive")
        if Path(stored_name).name != stored_name or "/" in stored_name or "\\" in stored_name:
            raise ValueError("stored_name must be a plain filename")
        now = utc_now()
        source = Source(
            id=new_id("src"),
            knowledge_base_id=knowledge_base_id,
            original_name=_required_text(original_name, "original_name", maximum=255),
            stored_name=_required_text(stored_name, "stored_name", maximum=255),
            media_type=_required_text(media_type, "media_type", maximum=255),
            size_bytes=size_bytes,
            content_sha256=content_sha256,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        try:
            with self._lock, self._connection:
                if max_total_bytes is not None:
                    used = int(
                        self._connection.execute(
                            "SELECT COALESCE(SUM(size_bytes), 0) FROM sources"
                        ).fetchone()[0]
                    )
                    if used + size_bytes > max_total_bytes:
                        raise ValueError(
                            f"Studio storage limit exceeded ({used + size_bytes} > "
                            f"{max_total_bytes} bytes)"
                        )
                self._connection.execute(
                    """INSERT INTO sources
                    (id, knowledge_base_id, original_name, stored_name, media_type, size_bytes,
                     content_sha256, status, error, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                    (
                        source.id,
                        source.knowledge_base_id,
                        source.original_name,
                        source.stored_name,
                        source.media_type,
                        source.size_bytes,
                        source.content_sha256,
                        source.status,
                        _datetime_text(now),
                        _datetime_text(now),
                    ),
                )
        except sqlite3.IntegrityError as error:
            if content_sha256 is not None and self.find_source_by_digest(
                knowledge_base_id, content_sha256
            ):
                raise ValueError("this file is already present in the knowledge base") from error
            raise
        return source

    def total_source_bytes(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM sources"
            ).fetchone()
        return int(row["total"])

    def find_source_by_digest(self, knowledge_base_id: str, content_sha256: str) -> Source | None:
        with self._lock:
            row = self._connection.execute(
                _SOURCE_SELECT
                + " WHERE sources.knowledge_base_id = ? AND sources.content_sha256 = ?",
                (knowledge_base_id, content_sha256),
            ).fetchone()
        return _source(row) if row is not None else None

    def get_source(self, source_id: str) -> Source | None:
        with self._lock:
            row = self._connection.execute(
                _SOURCE_SELECT + " WHERE sources.id = ?", (source_id,)
            ).fetchone()
        return _source(row) if row is not None else None

    def require_source(self, source_id: str) -> Source:
        source = self.get_source(source_id)
        if source is None:
            raise KeyError(f"source not found: {source_id}")
        return source

    def list_sources(self, knowledge_base_id: str) -> tuple[Source, ...]:
        self.require_knowledge_base(knowledge_base_id)
        with self._lock:
            rows = self._connection.execute(
                _SOURCE_SELECT + " WHERE sources.knowledge_base_id = ?"
                " ORDER BY sources.created_at DESC, sources.id",
                (knowledge_base_id,),
            ).fetchall()
        return tuple(_source(row) for row in rows)

    def update_source_status(
        self,
        source_id: str,
        status: SourceStatus,
        *,
        error: str | None = None,
    ) -> Source:
        if status not in {"pending", "processing", "ready", "failed"}:
            raise ValueError("invalid source status")
        source = self.require_source(source_id)
        clean_error = None if error is None else _optional_text(error, "error", maximum=2000)
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE sources SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, clean_error, _datetime_text(now), source_id),
            )
        return source.model_copy(update={"status": status, "error": clean_error, "updated_at": now})

    def delete_source(self, source_id: str) -> Source | None:
        source = self.get_source(source_id)
        if source is None:
            return None
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            self._connection.execute(
                "UPDATE knowledge_bases SET revision = revision + 1, updated_at = ? WHERE id = ?",
                (_datetime_text(now), source.knowledge_base_id),
            )
        return source

    def replace_source_documents(
        self, source_id: str, documents: Sequence[CorpusDocument]
    ) -> Source:
        source = self.require_source(source_id)
        for document in documents:
            metadata_source = document.metadata.get("source_id")
            if metadata_source != source_id:
                raise ValueError("every document must contain the matching source_id metadata")
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
            self._connection.executemany(
                """INSERT INTO documents
                (id, knowledge_base_id, source_id, contents, metadata_json)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    (
                        document.document_id,
                        source.knowledge_base_id,
                        source_id,
                        document.contents,
                        orjson.dumps(dict(document.metadata)).decode("utf-8"),
                    )
                    for document in documents
                ),
            )
            self._connection.execute(
                "UPDATE sources SET status = 'ready', error = NULL, updated_at = ? WHERE id = ?",
                (_datetime_text(now), source_id),
            )
            self._connection.execute(
                "UPDATE knowledge_bases SET revision = revision + 1, updated_at = ? WHERE id = ?",
                (_datetime_text(now), source.knowledge_base_id),
            )
        return source.model_copy(
            update={
                "status": "ready",
                "error": None,
                "document_count": len(documents),
                "updated_at": now,
            }
        )

    def list_documents(self, knowledge_base_id: str) -> tuple[CorpusDocument, ...]:
        self.require_knowledge_base(knowledge_base_id)
        with self._lock:
            rows = self._connection.execute(
                """SELECT id, contents, metadata_json FROM documents
                WHERE knowledge_base_id = ? ORDER BY id""",
                (knowledge_base_id,),
            ).fetchall()
        return tuple(_document(row) for row in rows)

    def index_input(self, knowledge_base_id: str) -> tuple[int, tuple[CorpusDocument, ...]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT revision FROM knowledge_bases WHERE id = ?", (knowledge_base_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"knowledge base not found: {knowledge_base_id}")
            documents = self._connection.execute(
                """SELECT id, contents, metadata_json FROM documents
                WHERE knowledge_base_id = ? ORDER BY id""",
                (knowledge_base_id,),
            ).fetchall()
        return int(row["revision"]), tuple(_document(item) for item in documents)

    def create_job(
        self,
        knowledge_base_id: str,
        *,
        kind: JobKind,
        source_id: str | None = None,
    ) -> IngestionJob:
        if kind not in {"ingest", "reindex"}:
            raise ValueError("invalid job kind")
        self.require_knowledge_base(knowledge_base_id)
        if source_id is not None:
            source = self.require_source(source_id)
            if source.knowledge_base_id != knowledge_base_id:
                raise ValueError("source does not belong to the knowledge base")
        now = utc_now()
        job = IngestionJob(
            id=new_id("job"),
            knowledge_base_id=knowledge_base_id,
            source_id=source_id,
            kind=kind,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO jobs
                (id, knowledge_base_id, source_id, kind, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    job.id,
                    job.knowledge_base_id,
                    job.source_id,
                    job.kind,
                    job.status,
                    _datetime_text(now),
                    _datetime_text(now),
                ),
            )
        return job

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job(row) if row is not None else None

    def list_jobs(
        self, knowledge_base_id: str | None = None, *, limit: int = 100
    ) -> tuple[IngestionJob, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._lock:
            if knowledge_base_id is None:
                rows = self._connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC, id LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """SELECT * FROM jobs WHERE knowledge_base_id = ?
                    ORDER BY created_at DESC, id LIMIT ?""",
                    (knowledge_base_id, limit),
                ).fetchall()
        return tuple(_job(row) for row in rows)

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
    ) -> IngestionJob:
        if status not in {"queued", "running", "succeeded", "failed"}:
            raise ValueError("invalid job status")
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(f"job not found: {job_id}")
        clean_error = None if error is None else _optional_text(error, "error", maximum=2000)
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, clean_error, _datetime_text(now), job_id),
            )
        return current.model_copy(
            update={"status": status, "error": clean_error, "updated_at": now}
        )

    def recover_interrupted_work(self) -> tuple[int, int]:
        """Fail work that cannot continue after the Studio process has restarted.

        Jobs and sources are updated in one transaction so callers never observe a recovered job
        while its source is still permanently shown as processing. Calling this method again is
        safe and leaves already-terminal work unchanged.
        """
        now = _datetime_text(utc_now())
        with self._lock, self._connection:
            jobs = self._connection.execute(
                """UPDATE jobs SET status = 'failed', error = ?, updated_at = ?
                WHERE status IN ('queued', 'running')""",
                (_INTERRUPTED_WORK_ERROR, now),
            )
            sources = self._connection.execute(
                """UPDATE sources SET status = 'failed', error = ?, updated_at = ?
                WHERE status IN ('pending', 'processing')""",
                (_INTERRUPTED_WORK_ERROR, now),
            )
        return int(jobs.rowcount), int(sources.rowcount)

    def save_model_settings(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        clear_api_key: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ModelSettings:
        if clear_api_key and api_key is not None:
            raise ValueError("api_key and clear_api_key cannot be used together")
        now = utc_now()
        validated = ModelSettings(
            base_url=_model_base_url(base_url),
            model=_required_text(model, "model", maximum=255),
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_configured=api_key is not None and bool(api_key),
            updated_at=now,
        )
        with self._lock, self._connection:
            current = self._connection.execute(
                "SELECT api_key FROM model_settings WHERE singleton = 1"
            ).fetchone()
            secret = None if current is None else cast(str | None, current["api_key"])
            if clear_api_key:
                secret = None
            elif api_key is not None:
                secret = _optional_text(api_key, "api_key", maximum=8192) or None
            self._connection.execute(
                """INSERT INTO model_settings
                (singleton, base_url, model, api_key, temperature, max_tokens, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    base_url = excluded.base_url,
                    model = excluded.model,
                    api_key = excluded.api_key,
                    temperature = excluded.temperature,
                    max_tokens = excluded.max_tokens,
                    updated_at = excluded.updated_at""",
                (
                    validated.base_url,
                    validated.model,
                    secret,
                    validated.temperature,
                    validated.max_tokens,
                    _datetime_text(now),
                ),
            )
        return validated.model_copy(update={"api_key_configured": bool(secret)})

    def get_model_settings(self) -> ModelSettings:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM model_settings WHERE singleton = 1"
            ).fetchone()
        if row is None:
            return ModelSettings(
                base_url=_DEFAULT_BASE_URL,
                model=_DEFAULT_MODEL,
                updated_at=utc_now(),
            )
        return ModelSettings(
            base_url=str(row["base_url"]),
            model=str(row["model"]),
            temperature=float(row["temperature"]),
            max_tokens=int(row["max_tokens"]),
            api_key_configured=bool(row["api_key"]),
            updated_at=_datetime(row["updated_at"]),
        )

    def get_model_api_key(self) -> str | None:
        """Return the secret only to the local chat client, never to an API response."""
        with self._lock:
            row = self._connection.execute(
                "SELECT api_key FROM model_settings WHERE singleton = 1"
            ).fetchone()
        return None if row is None else cast(str | None, row["api_key"])

    def get_api_key(self) -> str | None:
        """Compatibility alias for local callers; do not expose this value in API output."""
        return self.get_model_api_key()

    def get_index_state(self, knowledge_base_id: str) -> IndexStats | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM index_states WHERE knowledge_base_id = ?", (knowledge_base_id,)
            ).fetchone()
        return _index_stats(row) if row is not None else None

    def save_index_state(self, stats: IndexStats) -> bool:
        """Save state only if the indexed source revision is still current."""
        with self._lock, self._connection:
            current = self._connection.execute(
                "SELECT revision FROM knowledge_bases WHERE id = ?", (stats.knowledge_base_id,)
            ).fetchone()
            if current is None or int(current["revision"]) != stats.source_revision:
                return False
            self._connection.execute(
                """INSERT INTO index_states
                (knowledge_base_id, source_revision, generation, document_count,
                 corpus_sha256, built_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(knowledge_base_id) DO UPDATE SET
                    source_revision = excluded.source_revision,
                    generation = excluded.generation,
                    document_count = excluded.document_count,
                    corpus_sha256 = excluded.corpus_sha256,
                    built_at = excluded.built_at""",
                (
                    stats.knowledge_base_id,
                    stats.source_revision,
                    stats.generation,
                    stats.document_count,
                    stats.corpus_sha256,
                    _datetime_text(stats.built_at),
                ),
            )
        return True

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL UNIQUE,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                content_sha256 TEXT,
                status TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                contents TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS documents_kb_idx ON documents(knowledge_base_id);
            CREATE INDEX IF NOT EXISTS documents_source_idx ON documents(source_id);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                source_id TEXT REFERENCES sources(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_kb_idx ON jobs(knowledge_base_id, created_at);
            CREATE TABLE IF NOT EXISTS model_settings (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                base_url TEXT NOT NULL,
                model TEXT NOT NULL,
                api_key TEXT,
                temperature REAL NOT NULL,
                max_tokens INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS index_states (
                knowledge_base_id TEXT PRIMARY KEY REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                source_revision INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                document_count INTEGER NOT NULL,
                corpus_sha256 TEXT,
                built_at TEXT NOT NULL
            );
            """
        )
        source_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(sources)").fetchall()
        }
        if "content_sha256" not in source_columns:
            self._connection.execute("ALTER TABLE sources ADD COLUMN content_sha256 TEXT")
        self._connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS sources_kb_digest_unique
            ON sources(knowledge_base_id, content_sha256)
            WHERE content_sha256 IS NOT NULL"""
        )
        now = utc_now()
        self._connection.execute(
            """INSERT OR IGNORE INTO model_settings
            (singleton, base_url, model, api_key, temperature, max_tokens, updated_at)
            VALUES (1, ?, ?, NULL, 0.2, 1024, ?)""",
            (_DEFAULT_BASE_URL, _DEFAULT_MODEL, _datetime_text(now)),
        )


_SOURCE_SELECT = """SELECT sources.*,
    (SELECT COUNT(*) FROM documents WHERE documents.source_id = sources.id) AS document_count
    FROM sources"""


def _required_text(value: str, label: str, *, maximum: int) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{label} must not exceed {maximum} characters")
    return cleaned


def _optional_text(value: str, label: str, *, maximum: int) -> str:
    cleaned = value.strip()
    if len(cleaned) > maximum:
        raise ValueError(f"{label} must not exceed {maximum} characters")
    return cleaned


def _model_base_url(value: str) -> str:
    cleaned = _required_text(value, "base_url", maximum=2048).rstrip("/")
    try:
        parsed = urlsplit(cleaned)
        port = parsed.port
    except ValueError as error:
        raise ValueError("base_url must be a valid HTTP or HTTPS URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("base_url must be a valid HTTP or HTTPS URL without credentials")
    return cleaned


def _datetime_text(value: datetime) -> str:
    return value.isoformat()


def _datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


def _knowledge_base(row: sqlite3.Row) -> KnowledgeBase:
    return KnowledgeBase(
        id=str(row["id"]),
        name=str(row["name"]),
        description=str(row["description"]),
        revision=int(row["revision"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _source(row: sqlite3.Row) -> Source:
    return Source(
        id=str(row["id"]),
        knowledge_base_id=str(row["knowledge_base_id"]),
        original_name=str(row["original_name"]),
        stored_name=str(row["stored_name"]),
        media_type=str(row["media_type"]),
        size_bytes=int(row["size_bytes"]),
        content_sha256=(None if row["content_sha256"] is None else str(row["content_sha256"])),
        status=cast(SourceStatus, str(row["status"])),
        error=None if row["error"] is None else str(row["error"]),
        document_count=int(row["document_count"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _job(row: sqlite3.Row) -> IngestionJob:
    return IngestionJob(
        id=str(row["id"]),
        knowledge_base_id=str(row["knowledge_base_id"]),
        source_id=None if row["source_id"] is None else str(row["source_id"]),
        kind=cast(JobKind, str(row["kind"])),
        status=cast(JobStatus, str(row["status"])),
        error=None if row["error"] is None else str(row["error"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _document(row: sqlite3.Row) -> CorpusDocument:
    metadata = orjson.loads(str(row["metadata_json"]))
    if not isinstance(metadata, dict):
        raise ValueError("stored document metadata must be a JSON object")
    return CorpusDocument(str(row["id"]), str(row["contents"]), metadata)


def _index_stats(row: sqlite3.Row) -> IndexStats:
    return IndexStats(
        knowledge_base_id=str(row["knowledge_base_id"]),
        source_revision=int(row["source_revision"]),
        generation=int(row["generation"]),
        document_count=int(row["document_count"]),
        corpus_sha256=None if row["corpus_sha256"] is None else str(row["corpus_sha256"]),
        built_at=_datetime(row["built_at"]),
    )

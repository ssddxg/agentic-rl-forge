from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agentic_rl_forge.data.corpus import (
    CorpusDocument,
    corpus_digest,
    load_corpus_jsonl,
)
from agentic_rl_forge.search.tokenization import TOKENIZER_VERSION, tokenize_text
from agentic_rl_forge.services.observability import MetricsRegistry, instrument_fastapi

Document = CorpusDocument
load_jsonl_documents = load_corpus_jsonl

QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[QueryText] = Field(min_length=1, max_length=64)
    topk: int = Field(default=3, ge=1, le=100)
    return_scores: bool = True


class RetrieverReloadStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["loaded", "reloaded", "unchanged", "failed"]
    attempted_at: datetime
    error: str | None = None


class RetrieverStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: int = Field(ge=1)
    loaded_at: datetime
    last_reload: RetrieverReloadStatus | None
    document_count: int = Field(ge=1)
    unique_term_count: int = Field(ge=0)
    total_term_count: int = Field(ge=0)
    average_document_length: float = Field(ge=0.0)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_version: str


class BM25Index:
    def __init__(
        self,
        documents: tuple[Document, ...],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("retrieval index requires at least one document")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")

        self._documents = documents
        self._k1 = k1
        self._b = b
        document_lengths: list[int] = []
        mutable_postings: dict[str, list[tuple[int, int]]] = {}

        for document_index, document in enumerate(documents):
            frequencies = Counter(tokenize_text(document.contents))
            document_lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                mutable_postings.setdefault(term, []).append((document_index, frequency))

        self._document_lengths = tuple(document_lengths)
        self._postings = {term: tuple(postings) for term, postings in mutable_postings.items()}
        self._total_term_count = sum(document_lengths)
        self._average_document_length = self._total_term_count / len(documents)
        total = len(documents)
        self._idf = {
            term: math.log(1.0 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }
        self._corpus_sha256 = corpus_digest(documents)
        index_payload = json.dumps(
            {
                "b": b,
                "corpus_sha256": self._corpus_sha256,
                "k1": k1,
                "tokenizer_version": TOKENIZER_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._index_sha256 = hashlib.sha256(index_payload).hexdigest()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def unique_term_count(self) -> int:
        return len(self._postings)

    @property
    def total_term_count(self) -> int:
        return self._total_term_count

    @property
    def average_document_length(self) -> float:
        return self._average_document_length

    @property
    def corpus_sha256(self) -> str:
        return self._corpus_sha256

    @property
    def index_sha256(self) -> str:
        return self._index_sha256

    @property
    def tokenizer_version(self) -> str:
        return TOKENIZER_VERSION

    def search(self, query: str, *, top_k: int) -> list[dict[str, object]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores: dict[int, float] = {}
        query_frequencies = Counter(tokenize_text(query))
        for term, query_frequency in query_frequencies.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for document_index, frequency in self._postings[term]:
                document_length = self._document_lengths[document_index]
                length_ratio = (
                    document_length / self._average_document_length
                    if self._average_document_length
                    else 0.0
                )
                denominator = frequency + self._k1 * (1.0 - self._b + self._b * length_ratio)
                contribution = idf * frequency * (self._k1 + 1.0) * query_frequency / denominator
                scores[document_index] = scores.get(document_index, 0.0) + contribution

        ranked = sorted(
            (
                (score, self._documents[document_index])
                for document_index, score in scores.items()
                if score > 0
            ),
            key=lambda item: (-item[0], item[1].document_id),
        )
        return [
            {
                "document": _document_payload(document),
                "score": round(score, 8),
            }
            for score, document in ranked[:top_k]
        ]

    def search_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
    ) -> list[list[dict[str, object]]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return [self.search(query, top_k=top_k) for query in queries]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return tokenize_text(text)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    return value


def _document_payload(document: Document) -> dict[str, object]:
    payload = {
        key: _json_compatible(value)
        for key, value in document.metadata.items()
        if key not in {"id", "contents"}
    }
    payload["id"] = document.document_id
    payload["contents"] = document.contents
    return payload


@dataclass(frozen=True, slots=True)
class _FileSignature:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _RetrieverSnapshot:
    index: BM25Index
    generation: int
    loaded_at: datetime
    last_reload: RetrieverReloadStatus | None


def _file_signature(path: Path) -> _FileSignature:
    stat_before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    stat_after = path.stat()
    stable_before = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
        stat_before.st_ctime_ns,
    )
    stable_after = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
        stat_after.st_ctime_ns,
    )
    if stable_before != stable_after:
        raise ValueError("corpus changed while its content fingerprint was being computed")
    return _FileSignature(
        device=stat_after.st_dev,
        inode=stat_after.st_ino,
        size=stat_after.st_size,
        modified_ns=stat_after.st_mtime_ns,
        changed_ns=stat_after.st_ctime_ns,
        content_sha256=digest.hexdigest(),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReloadableRetriever:
    """Atomically replace a BM25 index after a corpus file changes."""

    def __init__(
        self,
        corpus_path: Path,
        *,
        reload_interval: float = 0.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(reload_interval) or reload_interval < 0:
            raise ValueError("reload_interval must be a finite non-negative number")

        self._corpus_path = corpus_path.resolve()
        self._reload_interval = reload_interval
        self._k1 = k1
        self._b = b
        self._lock = threading.RLock()
        self._reload_lock = threading.Lock()

        signature = _file_signature(self._corpus_path)
        index = BM25Index(load_corpus_jsonl(self._corpus_path), k1=k1, b=b)
        if _file_signature(self._corpus_path) != signature:
            raise ValueError("corpus changed while the initial index was being built")

        loaded_at = _utc_now()
        self._index = index
        self._generation = 1
        self._loaded_at = loaded_at
        self._last_reload = RetrieverReloadStatus(
            status="loaded",
            attempted_at=loaded_at,
        )
        self._observed_signature = signature

    @property
    def reload_interval(self) -> float:
        return self._reload_interval

    @property
    def current_index(self) -> BM25Index:
        with self._lock:
            return self._index

    def snapshot(self) -> _RetrieverSnapshot:
        with self._lock:
            return _RetrieverSnapshot(
                index=self._index,
                generation=self._generation,
                loaded_at=self._loaded_at,
                last_reload=self._last_reload,
            )

    def stats(self) -> RetrieverStats:
        return _stats_from_snapshot(self.snapshot())

    def reload_if_changed(self) -> RetrieverReloadStatus | None:
        with self._reload_lock:
            try:
                signature = _file_signature(self._corpus_path)
            except Exception as error:
                return self._record_failure(error)

            with self._lock:
                if signature == self._observed_signature:
                    return None

            attempted_at = _utc_now()
            try:
                replacement = BM25Index(
                    load_corpus_jsonl(self._corpus_path),
                    k1=self._k1,
                    b=self._b,
                )
                if _file_signature(self._corpus_path) != signature:
                    raise ValueError("corpus changed while the replacement index was being built")
            except Exception as error:
                return self._record_failure(error, attempted_at=attempted_at)

            with self._lock:
                self._observed_signature = signature
                if replacement.index_sha256 == self._index.index_sha256:
                    status = RetrieverReloadStatus(
                        status="unchanged",
                        attempted_at=attempted_at,
                    )
                    self._last_reload = status
                    return status
                self._index = replacement
                self._generation += 1
                self._loaded_at = _utc_now()
                status = RetrieverReloadStatus(
                    status="reloaded",
                    attempted_at=attempted_at,
                )
                self._last_reload = status
                return status

    def _record_failure(
        self,
        error: Exception,
        *,
        attempted_at: datetime | None = None,
    ) -> RetrieverReloadStatus:
        status = RetrieverReloadStatus(
            status="failed",
            attempted_at=attempted_at or _utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        with self._lock:
            self._last_reload = status
        return status


def _stats_from_snapshot(snapshot: _RetrieverSnapshot) -> RetrieverStats:
    index = snapshot.index
    return RetrieverStats(
        generation=snapshot.generation,
        loaded_at=snapshot.loaded_at,
        last_reload=snapshot.last_reload,
        document_count=index.document_count,
        unique_term_count=index.unique_term_count,
        total_term_count=index.total_term_count,
        average_document_length=index.average_document_length,
        corpus_sha256=index.corpus_sha256,
        index_sha256=index.index_sha256,
        tokenizer_version=index.tokenizer_version,
    )


async def _search_without_releasing_early(
    index: BM25Index,
    queries: tuple[str, ...],
    *,
    top_k: int,
) -> list[list[dict[str, object]]]:
    worker = asyncio.create_task(asyncio.to_thread(index.search_many, queries, top_k=top_k))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        with suppress(Exception, asyncio.CancelledError):
            worker.result()
        raise


def create_retriever_app(
    index: BM25Index | ReloadableRetriever,
    *,
    metrics: MetricsRegistry | None = None,
    max_concurrent_searches: int = 8,
) -> Any:
    if max_concurrent_searches < 1:
        raise ValueError("max_concurrent_searches must be positive")
    try:
        from fastapi import FastAPI
    except ImportError as error:
        raise RuntimeError("install the server extra to run the retrieval service") from error

    registry = metrics or MetricsRegistry()
    search_slots = asyncio.Semaphore(max_concurrent_searches)
    if isinstance(index, ReloadableRetriever):
        manager: ReloadableRetriever | None = index
        static_snapshot: _RetrieverSnapshot | None = None
    else:
        manager = None
        static_snapshot = _RetrieverSnapshot(
            index=index,
            generation=1,
            loaded_at=_utc_now(),
            last_reload=None,
        )

    def current_snapshot() -> _RetrieverSnapshot:
        if manager is not None:
            return manager.snapshot()
        assert static_snapshot is not None
        return static_snapshot

    def update_index_gauges(snapshot: _RetrieverSnapshot) -> None:
        registry.set_gauge(
            "arf_retrieval_documents",
            float(snapshot.index.document_count),
            help_text="Documents in the active retrieval index.",
        )
        registry.set_gauge(
            "arf_retrieval_index_generation",
            float(snapshot.generation),
            help_text="Generation of the active retrieval index.",
        )

    update_index_gauges(current_snapshot())
    registry.increment(
        "arf_retrieval_queries_total",
        0.0,
        help_text="Queries processed by the retrieval service.",
    )
    registry.increment(
        "arf_retrieval_results_total",
        0.0,
        help_text="Ranked results returned by the retrieval service.",
    )
    registry.increment(
        "arf_retrieval_zero_hit_queries_total",
        0.0,
        help_text="Queries that returned no matching documents.",
    )
    registry.set_gauge(
        "arf_retrieval_concurrency_limit",
        float(max_concurrent_searches),
        help_text="Maximum retrieval batches evaluated concurrently.",
    )

    async def watch_reloads() -> None:
        assert manager is not None
        while True:
            await asyncio.sleep(manager.reload_interval)
            outcome = await asyncio.to_thread(manager.reload_if_changed)
            if outcome is None:
                continue
            registry.increment(
                "arf_retrieval_reloads_total",
                labels={"outcome": outcome.status},
                help_text="Corpus reload attempts by outcome.",
            )
            update_index_gauges(manager.snapshot())

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        reload_task: asyncio.Task[None] | None = None
        if manager is not None and manager.reload_interval > 0:
            reload_task = asyncio.create_task(watch_reloads())
        try:
            yield
        finally:
            if reload_task is not None:
                reload_task.cancel()
                with suppress(asyncio.CancelledError):
                    await reload_task

    app = FastAPI(title="AgenticRLForge Retriever", version="1.0", lifespan=lifespan)
    instrument_fastapi(app, registry, service="retriever")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict[str, object]:
        snapshot = current_snapshot()
        update_index_gauges(snapshot)
        return _stats_from_snapshot(snapshot).model_dump(mode="json")

    @app.post("/retrieve")
    async def retrieve(request: RetrievalRequest) -> dict[str, object]:
        async with search_slots:
            snapshot = current_snapshot()
            results = await _search_without_releasing_early(
                snapshot.index,
                tuple(request.queries),
                top_k=request.topk,
            )
        result_count = sum(len(query_results) for query_results in results)
        zero_hit_count = sum(not query_results for query_results in results)
        registry.increment(
            "arf_retrieval_queries_total",
            float(len(request.queries)),
            help_text="Queries processed by the retrieval service.",
        )
        registry.increment(
            "arf_retrieval_results_total",
            float(result_count),
            help_text="Ranked results returned by the retrieval service.",
        )
        registry.increment(
            "arf_retrieval_zero_hit_queries_total",
            float(zero_hit_count),
            help_text="Queries that returned no matching documents.",
        )
        if not request.return_scores:
            results = [
                [{"document": item["document"]} for item in query_results]
                for query_results in results
            ]
        return {"result": results}

    return app

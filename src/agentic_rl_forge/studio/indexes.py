from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import orjson

from agentic_rl_forge.contracts import utc_now
from agentic_rl_forge.data import CorpusDocument, corpus_digest, load_corpus_jsonl
from agentic_rl_forge.services.retriever import BM25Index
from agentic_rl_forge.studio.models import IndexStats, SearchHit, SearchMode
from agentic_rl_forge.studio.paths import StudioPaths
from agentic_rl_forge.studio.repository import StudioRepository


@dataclass(frozen=True, slots=True)
class _CachedIndex:
    stats: IndexStats
    documents: tuple[CorpusDocument, ...]
    index: BM25Index | None


class StudioIndexManager:
    """Persistent BM25 plus an explicitly named character n-gram hybrid ranker."""

    def __init__(self, paths: StudioPaths, repository: StudioRepository) -> None:
        self._paths = paths.ensure()
        self._repository = repository
        self._cache: dict[str, _CachedIndex] = {}
        self._lock = threading.RLock()

    def rebuild(self, knowledge_base_id: str) -> IndexStats:
        for _ in range(3):
            revision, documents = self._repository.index_input(knowledge_base_id)
            index = BM25Index(documents) if documents else None
            digest = corpus_digest(documents) if documents else None
            previous = self._repository.get_index_state(knowledge_base_id)
            generation = 1 if previous is None else previous.generation + 1
            stats = IndexStats(
                knowledge_base_id=knowledge_base_id,
                source_revision=revision,
                generation=generation,
                document_count=len(documents),
                corpus_sha256=digest,
                built_at=utc_now(),
            )
            self._publish_corpus(knowledge_base_id, documents)
            if not self._repository.save_index_state(stats):
                continue
            cached = _CachedIndex(stats=stats, documents=documents, index=index)
            with self._lock:
                self._cache[knowledge_base_id] = cached
            return stats
        raise RuntimeError("knowledge base changed repeatedly while its index was being rebuilt")

    def stats(self, knowledge_base_id: str) -> IndexStats:
        return self._current(knowledge_base_id).stats

    def search(
        self,
        knowledge_base_id: str,
        query: str,
        *,
        top_k: int = 5,
        mode: SearchMode = "bm25",
    ) -> tuple[SearchHit, ...]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if mode not in {"bm25", "hybrid_character"}:
            raise ValueError("mode must be 'bm25' or 'hybrid_character'")
        cached = self._current(knowledge_base_id)
        if cached.index is None:
            return ()
        ranked = cached.index.search(clean_query, top_k=max(top_k, len(cached.documents)))
        bm25_by_id: dict[str, float] = {}
        for item in ranked:
            document_payload = cast(dict[str, object], item["document"])
            score_value = item["score"]
            if not isinstance(score_value, int | float):
                raise ValueError("BM25 result contains an invalid score")
            bm25_by_id[str(document_payload["id"])] = float(score_value)
        if mode == "bm25":
            by_id = {document.document_id: document for document in cached.documents}
            return tuple(
                self._hit(
                    by_id[document_id],
                    score=score,
                    bm25_score=score,
                    character_score=0.0,
                    mode=mode,
                )
                for document_id, score in list(bm25_by_id.items())[:top_k]
            )

        maximum_bm25 = max(bm25_by_id.values(), default=0.0)
        candidates: list[tuple[float, float, float, CorpusDocument]] = []
        for document in cached.documents:
            raw_bm25 = bm25_by_id.get(document.document_id, 0.0)
            normalized_bm25 = raw_bm25 / maximum_bm25 if maximum_bm25 else 0.0
            character_score = character_ngram_similarity(clean_query, document.contents)
            combined = 0.75 * normalized_bm25 + 0.25 * character_score
            if combined > 0:
                candidates.append((combined, raw_bm25, character_score, document))
        candidates.sort(key=lambda item: (-item[0], item[3].document_id))
        return tuple(
            self._hit(
                document,
                score=combined,
                bm25_score=raw_bm25,
                character_score=character_score,
                mode=mode,
            )
            for combined, raw_bm25, character_score, document in candidates[:top_k]
        )

    def invalidate(self, knowledge_base_id: str) -> None:
        with self._lock:
            self._cache.pop(knowledge_base_id, None)

    def delete(self, knowledge_base_id: str) -> None:
        self.invalidate(knowledge_base_id)
        parent = self._paths.indexes.resolve()
        directory = self._paths.index_directory(knowledge_base_id)
        resolved = directory.resolve()
        if resolved.parent != parent:
            raise ValueError("index directory escapes the Studio data directory")
        if directory.exists():
            shutil.rmtree(directory)

    def _current(self, knowledge_base_id: str) -> _CachedIndex:
        knowledge_base = self._repository.require_knowledge_base(knowledge_base_id)
        with self._lock:
            cached = self._cache.get(knowledge_base_id)
        if cached is not None and cached.stats.source_revision == knowledge_base.revision:
            return cached
        restored = self._restore(knowledge_base_id, knowledge_base.revision)
        if restored is not None:
            with self._lock:
                self._cache[knowledge_base_id] = restored
            return restored
        self.rebuild(knowledge_base_id)
        with self._lock:
            return self._cache[knowledge_base_id]

    def _restore(self, knowledge_base_id: str, revision: int) -> _CachedIndex | None:
        state = self._repository.get_index_state(knowledge_base_id)
        if state is None or state.source_revision != revision:
            return None
        if state.document_count == 0:
            return _CachedIndex(stats=state, documents=(), index=None)
        path = self._corpus_path(knowledge_base_id)
        try:
            documents = load_corpus_jsonl(path)
            if (
                len(documents) != state.document_count
                or corpus_digest(documents) != state.corpus_sha256
            ):
                return None
            return _CachedIndex(stats=state, documents=documents, index=BM25Index(documents))
        except (OSError, ValueError):
            return None

    def _publish_corpus(
        self, knowledge_base_id: str, documents: tuple[CorpusDocument, ...]
    ) -> None:
        directory = self._paths.index_directory(knowledge_base_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = self._corpus_path(knowledge_base_id)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".index-", suffix=".tmp", dir=directory
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for document in documents:
                    payload = {
                        "id": document.document_id,
                        "contents": document.contents,
                        **_json_compatible(document.metadata),
                    }
                    destination.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _corpus_path(self, knowledge_base_id: str) -> Path:
        return self._paths.index_directory(knowledge_base_id) / "corpus.jsonl"

    @staticmethod
    def _hit(
        document: CorpusDocument,
        *,
        score: float,
        bm25_score: float,
        character_score: float,
        mode: SearchMode,
    ) -> SearchHit:
        source_id = document.metadata.get("source_id")
        source_name = document.metadata.get("source_name") or document.metadata.get("source")
        if not isinstance(source_id, str) or not isinstance(source_name, str):
            raise ValueError("indexed Studio documents require source_id and source_name metadata")
        return SearchHit(
            document_id=document.document_id,
            source_id=source_id,
            source_name=source_name,
            contents=document.contents,
            score=max(0.0, score),
            bm25_score=max(0.0, bm25_score),
            character_score=min(1.0, max(0.0, character_score)),
            scoring_method=mode,
            metadata=cast(dict[str, Any], _json_compatible(document.metadata)),
        )


def character_ngram_similarity(query: str, text: str) -> float:
    """Dice similarity over normalized character bigrams; this is not an embedding model."""
    query_grams = _character_ngrams(query)
    text_grams = _character_ngrams(text)
    if not query_grams or not text_grams:
        return 0.0
    return 2.0 * len(query_grams & text_grams) / (len(query_grams) + len(text_grams))


def _character_ngrams(value: str) -> set[str]:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKC", value).casefold()
        if character.isalnum()
    )
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metadata numbers must be finite")
    return value

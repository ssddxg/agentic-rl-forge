import asyncio
import json
import math
import os
import threading
from collections import Counter
from pathlib import Path

import httpx
import pytest

from agentic_rl_forge.data import CorpusDocument
from agentic_rl_forge.search import TOKENIZER_VERSION, tokenize_text
from agentic_rl_forge.services import (
    BM25Index,
    Document,
    ReloadableRetriever,
    create_retriever_app,
    load_jsonl_documents,
)


def _write_corpus(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _reference_search(
    documents: tuple[Document, ...],
    query: str,
    *,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    terms = [tokenize_text(document.contents) for document in documents]
    frequencies = [Counter(document_terms) for document_terms in terms]
    average_length = sum(map(len, terms)) / len(terms)
    document_frequency: Counter[str] = Counter()
    for document_terms in terms:
        document_frequency.update(set(document_terms))
    total = len(documents)
    inverse_document_frequency = {
        term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    scored: list[tuple[float, Document]] = []
    for document, term_frequency, document_terms in zip(documents, frequencies, terms, strict=True):
        score = 0.0
        for term in tokenize_text(query):
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (1.0 - b + b * len(document_terms) / average_length)
            score += inverse_document_frequency[term] * frequency * (k1 + 1.0) / denominator
        if score > 0:
            scored.append((score, document))
    scored.sort(key=lambda item: (-item[0], item[1].document_id))
    return [(document.document_id, round(score, 8)) for score, document in scored[:top_k]]


def test_tokenizer_normalizes_unicode_and_emits_cjk_bigrams() -> None:
    fullwidth = "\uff21\uff47\uff45\uff4e\uff54\uff49\uff43 \uff32\uff2c Café"
    assert tokenize_text(fullwidth) == ["agentic", "rl", "café"]
    assert tokenize_text("巴黎是法国的首都") == [
        "巴",
        "黎",
        "是",
        "法",
        "国",
        "的",
        "首",
        "都",
        "巴黎",
        "黎是",
        "是法",
        "法国",
        "国的",
        "的首",
        "首都",
    ]
    assert tokenize_text("中 API 文") == ["中", "api", "文"]
    assert "法" in tokenize_text("法国")


def test_bm25_index_searches_english_and_chinese_with_stable_ties() -> None:
    documents = (
        Document("zh-fr", "巴黎是法国的首都。"),
        Document("zh-de", "柏林是德国的首都。"),
        Document("en-fr", "Paris is the capital of France."),
    )
    index = BM25Index(documents)

    assert index.search("法国 首都", top_k=1)[0]["document"]["id"] == "zh-fr"
    assert index.search("法", top_k=1)[0]["document"]["id"] == "zh-fr"
    assert index.search("FRANCE CAPITAL", top_k=1)[0]["document"]["id"] == "en-fr"
    assert index.document_count == 3
    assert index.unique_term_count > 0
    assert index.total_term_count > 0
    assert index.average_document_length > 0
    assert len(index.corpus_sha256) == 64
    assert len(index.index_sha256) == 64
    assert index.tokenizer_version == TOKENIZER_VERSION

    tied = BM25Index((Document("b", "same text"), Document("a", "same text")))
    assert [item["document"]["id"] for item in tied.search("same", top_k=2)] == ["a", "b"]


def test_inverted_bm25_matches_reference_scoring_with_repeated_query_terms() -> None:
    documents = (
        Document("a", "alpha alpha beta"),
        Document("b", "alpha gamma"),
        Document("c", "beta gamma gamma"),
    )
    query = "alpha alpha gamma"

    actual = [
        (item["document"]["id"], item["score"])
        for item in BM25Index(documents).search(query, top_k=3)
    ]

    assert actual == _reference_search(documents, query, top_k=3)


@pytest.mark.parametrize(
    ("k1", "b"),
    [
        (0.0, 0.75),
        (-1.0, 0.75),
        (float("nan"), 0.75),
        (1.5, -0.1),
        (1.5, 1.1),
        (1.5, float("inf")),
    ],
)
def test_bm25_index_rejects_invalid_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25Index((Document("doc", "text"),), k1=k1, b=b)


def test_bm25_index_rejects_non_positive_top_k() -> None:
    index = BM25Index((Document("doc", "text"),))

    with pytest.raises(ValueError, match="requires at least one document"):
        BM25Index(())
    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search("text", top_k=0)
    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search_many(("text",), top_k=-1)


def test_bm25_scans_each_repeated_query_term_once(monkeypatch: pytest.MonkeyPatch) -> None:
    index = BM25Index((Document("doc", "repeated term"),))
    accesses = 0

    class CountingPostings(dict[str, tuple[tuple[int, int], ...]]):
        def __getitem__(self, key: str) -> tuple[tuple[int, int], ...]:
            nonlocal accesses
            accesses += 1
            return super().__getitem__(key)

    monkeypatch.setattr(index, "_postings", CountingPostings(index._postings))

    results = index.search(" ".join(["term"] * 2048), top_k=1)

    assert results[0]["document"]["id"] == "doc"
    assert accesses == 1


def test_retriever_app_rejects_invalid_concurrency_limit() -> None:
    index = BM25Index((Document("doc", "text"),))

    with pytest.raises(ValueError, match="max_concurrent_searches must be positive"):
        create_retriever_app(index, max_concurrent_searches=0)


def test_legacy_document_and_loader_names_remain_compatible(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [{"id": "one", "contents": "first", "source": "notes.md"}])

    documents = load_jsonl_documents(corpus)

    assert Document is CorpusDocument
    assert documents == (Document("one", "first", {"source": "notes.md"}),)


@pytest.mark.asyncio
async def test_retriever_api_preserves_protocol_and_exposes_stats_and_metrics() -> None:
    index = BM25Index(
        (
            Document(
                "fr",
                "Paris is the capital of France.",
                {
                    "contents": "metadata cannot replace contents",
                    "details": {"language": "en"},
                    "id": "metadata-cannot-replace-id",
                    "source": "france.md",
                    "tags": ["capital"],
                },
            ),
            Document("de", "Berlin is the capital of Germany."),
        )
    )
    app = create_retriever_app(index)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://retriever.test",
    ) as client:
        health = await client.get("/health")
        stats = await client.get("/stats")
        retrieved = await client.post(
            "/retrieve",
            json={"queries": ["  France capital  ", "missing"], "topk": 2},
        )
        without_scores = await client.post(
            "/retrieve",
            json={"queries": ["France"], "topk": 1, "return_scores": False},
        )
        metrics = await client.get("/metrics")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert stats.status_code == 200
    assert stats.json()["generation"] == 1
    assert stats.json()["document_count"] == 2
    assert stats.json()["tokenizer_version"] == TOKENIZER_VERSION
    assert stats.json()["last_reload"] is None
    assert retrieved.json()["result"][0][0]["document"] == {
        "id": "fr",
        "contents": "Paris is the capital of France.",
        "details": {"language": "en"},
        "source": "france.md",
        "tags": ["capital"],
    }
    assert retrieved.json()["result"][1] == []
    assert without_scores.json() == {
        "result": [
            [
                {
                    "document": {
                        "id": "fr",
                        "contents": "Paris is the capital of France.",
                        "details": {"language": "en"},
                        "source": "france.md",
                        "tags": ["capital"],
                    }
                }
            ]
        ]
    }
    assert "arf_retrieval_documents 2" in metrics.text
    assert "arf_retrieval_concurrency_limit 8" in metrics.text
    assert "arf_retrieval_queries_total 3" in metrics.text
    assert "arf_retrieval_results_total 3" in metrics.text
    assert "arf_retrieval_zero_hit_queries_total 1" in metrics.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"queries": []},
        {"queries": [""]},
        {"queries": ["   "]},
        {"queries": ["q"] * 65},
        {"queries": ["q" * 4097]},
        {"queries": ["q"], "topk": 0},
        {"queries": ["q"], "topk": 101},
        {"queries": ["q"], "unexpected": True},
    ],
)
async def test_retriever_api_rejects_unbounded_requests(payload: dict[str, object]) -> None:
    app = create_retriever_app(BM25Index((Document("doc", "query text"),)))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://retriever.test",
    ) as client:
        response = await client.post("/retrieve", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_retrieval_search_does_not_block_health_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = BM25Index((Document("doc", "query text"),))
    original_search_many = index.search_many
    started = threading.Event()
    release = threading.Event()

    def slow_search_many(
        queries: tuple[str, ...],
        *,
        top_k: int,
    ) -> list[list[dict[str, object]]]:
        started.set()
        if not release.wait(timeout=2):
            raise TimeoutError("test did not release retrieval worker")
        return original_search_many(queries, top_k=top_k)

    monkeypatch.setattr(index, "search_many", slow_search_many)
    app = create_retriever_app(index)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://retriever.test",
    ) as client:
        retrieval = asyncio.create_task(client.post("/retrieve", json={"queries": ["query"]}))
        assert await asyncio.to_thread(started.wait, 1)
        try:
            health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
        finally:
            release.set()
        result = await retrieval

    assert health.json() == {"status": "ok"}
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_cancelled_request_keeps_its_search_slot_until_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = BM25Index((Document("doc", "query text"),))
    original_search_many = index.search_many
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    call_count = 0

    def controlled_search_many(
        queries: tuple[str, ...],
        *,
        top_k: int,
    ) -> list[list[dict[str, object]]]:
        nonlocal active, call_count, max_active
        with lock:
            call_count += 1
            current_call = call_count
            active += 1
            max_active = max(max_active, active)
        try:
            if current_call == 1:
                first_started.set()
                if not release_first.wait(timeout=2):
                    raise TimeoutError("test did not release first retrieval worker")
            else:
                second_started.set()
            return original_search_many(queries, top_k=top_k)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(index, "search_many", controlled_search_many)
    app = create_retriever_app(index, max_concurrent_searches=1)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://retriever.test",
    ) as client:
        first = asyncio.create_task(client.post("/retrieve", json={"queries": ["query"]}))
        assert await asyncio.to_thread(first_started.wait, 1)
        first.cancel()
        second = asyncio.create_task(client.post("/retrieve", json={"queries": ["query"]}))
        await asyncio.sleep(0.05)
        assert not second_started.is_set()
        release_first.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        response = await second

    assert response.status_code == 200
    assert max_active == 1


@pytest.mark.asyncio
async def test_in_flight_batch_uses_one_index_generation_during_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [{"id": "old", "contents": "old generation evidence"}])
    retriever = ReloadableRetriever(corpus)
    old_index = retriever.current_index
    original_search_many = old_index.search_many
    started = threading.Event()
    release = threading.Event()

    def slow_search_many(
        queries: tuple[str, ...],
        *,
        top_k: int,
    ) -> list[list[dict[str, object]]]:
        started.set()
        if not release.wait(timeout=2):
            raise TimeoutError("test did not release retrieval worker")
        return original_search_many(queries, top_k=top_k)

    monkeypatch.setattr(old_index, "search_many", slow_search_many)
    app = create_retriever_app(retriever)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://retriever.test",
    ) as client:
        in_flight = asyncio.create_task(
            client.post("/retrieve", json={"queries": ["old generation"]})
        )
        assert await asyncio.to_thread(started.wait, 1)
        try:
            _write_corpus(corpus, [{"id": "new", "contents": "new generation evidence"}])
            reloaded = await asyncio.to_thread(retriever.reload_if_changed)
        finally:
            release.set()
        old_response = await in_flight
        new_response = await client.post("/retrieve", json={"queries": ["new generation"]})

    assert reloaded is not None
    assert reloaded.status == "reloaded"
    assert old_response.json()["result"][0][0]["document"]["id"] == "old"
    assert new_response.json()["result"][0][0]["document"]["id"] == "new"


def test_reloadable_retriever_swaps_valid_indexes_and_keeps_last_good_index(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [{"id": "old", "contents": "old searchable text"}])
    retriever = ReloadableRetriever(corpus)
    initial = retriever.stats()

    corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    unchanged = retriever.reload_if_changed()

    assert unchanged is not None
    assert unchanged.status == "unchanged"
    assert retriever.stats().generation == 1

    _write_corpus(corpus, [{"id": "new", "contents": "new replacement searchable text"}])
    successful = retriever.reload_if_changed()

    assert successful is not None
    assert successful.status == "reloaded"
    assert retriever.stats().generation == 2
    assert retriever.stats().loaded_at >= initial.loaded_at
    assert retriever.current_index.search("replacement", top_k=1)[0]["document"]["id"] == "new"

    corpus.write_text("not-json\n", encoding="utf-8")
    failed = retriever.reload_if_changed()

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error is not None
    assert retriever.stats().generation == 2
    assert retriever.current_index.search("replacement", top_k=1)[0]["document"]["id"] == "new"
    retried = retriever.reload_if_changed()
    assert retried is not None
    assert retried.status == "failed"

    _write_corpus(corpus, [{"id": "fixed", "contents": "fixed healthy searchable text"}])
    repaired = retriever.reload_if_changed()

    assert repaired is not None
    assert repaired.status == "reloaded"
    assert retriever.stats().generation == 3
    assert retriever.current_index.search("healthy", top_k=1)[0]["document"]["id"] == "fixed"

    with pytest.raises(ValueError, match="reload_interval"):
        ReloadableRetriever(corpus, reload_interval=-1)


def test_reload_detection_uses_content_when_size_and_mtime_are_unchanged(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [{"id": "one", "contents": "alpha token"}])
    original_stat = corpus.stat()
    retriever = ReloadableRetriever(corpus)

    _write_corpus(corpus, [{"id": "two", "contents": "bravo token"}])
    assert corpus.stat().st_size == original_stat.st_size
    os.utime(corpus, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    outcome = retriever.reload_if_changed()

    assert outcome is not None
    assert outcome.status == "reloaded"
    assert retriever.current_index.search("bravo", top_k=1)[0]["document"]["id"] == "two"


@pytest.mark.asyncio
async def test_retriever_app_watches_and_reports_corpus_reloads(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, [{"id": "one", "contents": "first searchable text"}])
    retriever = ReloadableRetriever(corpus, reload_interval=0.01)
    app = create_retriever_app(retriever)

    async with app.router.lifespan_context(app):
        _write_corpus(corpus, [{"id": "two", "contents": "second replacement searchable text"}])
        deadline = asyncio.get_running_loop().time() + 2
        while retriever.stats().generation == 1:
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("retriever did not reload the changed corpus")
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://retriever.test",
        ) as client:
            stats = await client.get("/stats")
            result = await client.post("/retrieve", json={"queries": ["replacement"]})
            metrics = await client.get("/metrics")

    assert stats.json()["generation"] == 2
    assert stats.json()["last_reload"]["status"] == "reloaded"
    assert result.json()["result"][0][0]["document"]["id"] == "two"
    assert 'arf_retrieval_reloads_total{outcome="reloaded"} 1' in metrics.text

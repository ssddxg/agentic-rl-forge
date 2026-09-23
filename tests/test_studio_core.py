from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from agentic_rl_forge.studio import (
    DocumentIngestor,
    OpenAICompatibleChatClient,
    OptionalDocumentDependencyError,
    SearchHit,
    StudioIndexManager,
    StudioPaths,
    StudioRepository,
    character_ngram_similarity,
    safe_upload_filename,
)


@pytest.fixture
def studio(tmp_path: Path) -> tuple[StudioPaths, StudioRepository, DocumentIngestor]:
    paths = StudioPaths(tmp_path / "studio").ensure()
    repository = StudioRepository(paths.database)
    ingestor = DocumentIngestor(paths, repository, max_file_bytes=1024)
    try:
        yield paths, repository, ingestor
    finally:
        repository.close()


def test_repository_crud_jobs_and_restart_persistence(
    studio: tuple[StudioPaths, StudioRepository, DocumentIngestor],
) -> None:
    paths, repository, _ = studio
    knowledge_base = repository.create_knowledge_base("团队手册", "内部流程")
    assert repository.list_knowledge_bases() == (knowledge_base,)
    updated = repository.update_knowledge_base(knowledge_base.id, name="产品手册", description="")
    assert updated.name == "产品手册"
    assert updated.description == ""

    source = repository.create_source(
        knowledge_base.id,
        original_name="guide.md",
        stored_name="abc.md",
        media_type="text/markdown",
        size_bytes=10,
    )
    job = repository.create_job(
        knowledge_base.id,
        kind="ingest",
        source_id=source.id,
    )
    assert repository.update_job(job.id, "running").status == "running"
    assert repository.list_jobs(knowledge_base.id)[0].id == job.id

    settings = repository.save_model_settings(
        "http://127.0.0.1:11434/v1",
        "local-model",
        api_key="super-secret",
    )
    assert settings.api_key_configured
    assert "super-secret" not in repr(settings)
    assert "api_key" not in settings.model_dump()
    repository.close()

    reopened = StudioRepository(paths.database)
    try:
        assert reopened.get_knowledge_base(knowledge_base.id) is not None
        assert reopened.get_source(source.id) is not None
        assert reopened.get_job(job.id) is not None
        assert reopened.get_model_api_key() == "super-secret"
        safe_settings = reopened.get_model_settings()
        assert safe_settings.api_key_configured
        assert "super-secret" not in json.dumps(safe_settings.model_dump(mode="json"))

        reopened.save_model_settings("http://localhost:8000", "next-model")
        assert reopened.get_model_api_key() == "super-secret"
        reopened.save_model_settings("http://localhost:8000", "next-model", clear_api_key=True)
        assert reopened.get_model_api_key() is None
    finally:
        reopened.close()


def test_repository_recovers_interrupted_work_atomically_and_idempotently(
    tmp_path: Path,
) -> None:
    paths = StudioPaths(tmp_path / "studio").ensure()
    repository = StudioRepository(paths.database)
    knowledge_base = repository.create_knowledge_base("Recovery")
    pending_source = repository.create_source(
        knowledge_base.id,
        original_name="pending.md",
        stored_name="pending.md",
        media_type="text/markdown",
        size_bytes=10,
    )
    processing_source = repository.create_source(
        knowledge_base.id,
        original_name="processing.md",
        stored_name="processing.md",
        media_type="text/markdown",
        size_bytes=10,
    )
    ready_source = repository.create_source(
        knowledge_base.id,
        original_name="ready.md",
        stored_name="ready.md",
        media_type="text/markdown",
        size_bytes=10,
    )
    repository.update_source_status(processing_source.id, "processing")
    repository.update_source_status(ready_source.id, "ready")
    queued_job = repository.create_job(
        knowledge_base.id,
        kind="ingest",
        source_id=pending_source.id,
    )
    running_job = repository.create_job(
        knowledge_base.id,
        kind="reindex",
        source_id=processing_source.id,
    )
    repository.update_job(running_job.id, "running")
    succeeded_job = repository.create_job(
        knowledge_base.id,
        kind="reindex",
        source_id=ready_source.id,
    )
    repository.update_job(succeeded_job.id, "succeeded")
    repository.close()

    # If the source half cannot commit, the earlier job update must roll back too.
    with sqlite3.connect(paths.database) as connection:
        connection.execute(
            """CREATE TRIGGER reject_source_recovery BEFORE UPDATE OF status ON sources
            WHEN NEW.status = 'failed'
            BEGIN SELECT RAISE(ABORT, 'blocked recovery'); END"""
        )
    repository = StudioRepository(paths.database)
    with pytest.raises(sqlite3.IntegrityError, match="blocked recovery"):
        repository.recover_interrupted_work()
    assert repository.require_source(pending_source.id).status == "pending"
    rolled_back_job = repository.get_job(queued_job.id)
    assert rolled_back_job is not None
    assert rolled_back_job.status == "queued"
    repository.close()

    with sqlite3.connect(paths.database) as connection:
        connection.execute("DROP TRIGGER reject_source_recovery")
    repository = StudioRepository(paths.database)
    try:
        assert repository.recover_interrupted_work() == (2, 2)
        expected_error = "上次运行中断, 请重新处理。"
        recovered_pending = repository.require_source(pending_source.id)
        recovered_processing = repository.require_source(processing_source.id)
        recovered_queued_job = repository.get_job(queued_job.id)
        recovered_running_job = repository.get_job(running_job.id)
        assert recovered_pending.status == "failed"
        assert recovered_pending.error == expected_error
        assert recovered_processing.status == "failed"
        assert recovered_processing.error == expected_error
        assert recovered_queued_job is not None
        assert recovered_queued_job.status == "failed"
        assert recovered_queued_job.error == expected_error
        assert recovered_running_job is not None
        assert recovered_running_job.status == "failed"
        assert recovered_running_job.error == expected_error
        assert recovered_queued_job.updated_at == recovered_pending.updated_at
        assert recovered_running_job.updated_at == recovered_processing.updated_at

        unchanged_source = repository.require_source(ready_source.id)
        unchanged_job = repository.get_job(succeeded_job.id)
        assert unchanged_source.status == "ready"
        assert unchanged_source.error is None
        assert unchanged_job is not None
        assert unchanged_job.status == "succeeded"
        timestamps = (
            recovered_pending.updated_at,
            recovered_processing.updated_at,
            recovered_queued_job.updated_at,
            recovered_running_job.updated_at,
        )
        assert repository.recover_interrupted_work() == (0, 0)
        second_queued_job = repository.get_job(queued_job.id)
        second_running_job = repository.get_job(running_job.id)
        assert second_queued_job is not None
        assert second_running_job is not None
        assert (
            repository.require_source(pending_source.id).updated_at,
            repository.require_source(processing_source.id).updated_at,
            second_queued_job.updated_at,
            second_running_job.updated_at,
        ) == timestamps
    finally:
        repository.close()


def test_streamed_upload_ingestion_safety_and_source_deletion(
    studio: tuple[StudioPaths, StudioRepository, DocumentIngestor],
) -> None:
    paths, repository, ingestor = studio
    knowledge_base = repository.create_knowledge_base("Docs")
    source = ingestor.store_upload(
        knowledge_base.id,
        "../指南.MD",
        ["巴黎是法国首都。".encode(), b"\nThe deploy command is arf studio."],
        "text/markdown",
    )
    assert source.original_name == "指南.MD"
    assert source.stored_name.endswith(".md")
    assert "/" not in source.stored_name
    assert paths.source_path(knowledge_base.id, source.stored_name).is_file()

    documents = ingestor.ingest_source(source.id)
    assert documents
    assert documents[0].metadata["source_id"] == source.id
    assert repository.get_source(source.id).status == "ready"  # type: ignore[union-attr]
    assert repository.require_knowledge_base(knowledge_base.id).revision == 1

    files_before = set(paths.source_directory(knowledge_base.id).iterdir())
    with pytest.raises(ValueError, match="already present"):
        ingestor.store_upload(
            knowledge_base.id,
            "same-content.txt",
            ["巴黎是法国首都。\nThe deploy command is arf studio.".encode()],
        )
    assert set(paths.source_directory(knowledge_base.id).iterdir()) == files_before

    deleted = repository.delete_source(source.id)
    assert deleted is not None
    ingestor.delete_source_file(deleted)
    ingestor.delete_source_file(deleted)
    assert repository.get_source(source.id) is None
    assert repository.require_knowledge_base(knowledge_base.id).revision == 2

    with pytest.raises(ValueError, match="upload limit"):
        ingestor.store_upload(knowledge_base.id, "huge.txt", [b"x" * 1025])
    assert not list(paths.source_directory(knowledge_base.id).glob(".upload-*.tmp"))
    with pytest.raises(ValueError, match="unsupported document type"):
        ingestor.store_upload(knowledge_base.id, "program.exe", [b"x"])


def test_html_extraction_ignores_scripts_and_normalizes_text(
    studio: tuple[StudioPaths, StudioRepository, DocumentIngestor],
) -> None:
    _, repository, ingestor = studio
    knowledge_base = repository.create_knowledge_base("Web docs")
    source = ingestor.store_upload(
        knowledge_base.id,
        "page.html",
        [
            b"<html><head><title>Install</title><style>hidden</style></head>",
            b"<body><h1>Quick start</h1><p>Run the command.</p>",
            b"<script>steal()</script></body></html>",
        ],
        "text/html",
    )
    documents = ingestor.ingest_source(source.id)
    assert "Quick start" in documents[0].contents
    assert "Run the command" in documents[0].contents
    assert "hidden" not in documents[0].contents
    assert "steal" not in documents[0].contents
    assert documents[0].metadata["title"] == "Install"


def test_total_storage_limit_rejects_without_orphan_files(tmp_path: Path) -> None:
    paths = StudioPaths(tmp_path / "studio").ensure()
    repository = StudioRepository(paths.database)
    ingestor = DocumentIngestor(
        paths,
        repository,
        max_file_bytes=8,
        max_total_bytes=12,
    )
    knowledge_base = repository.create_knowledge_base("Limited")
    try:
        ingestor.store_upload(knowledge_base.id, "one.txt", [b"1234567"])
        with pytest.raises(ValueError, match="storage limit"):
            ingestor.store_upload(knowledge_base.id, "two.txt", [b"123456"])
        assert len(repository.list_sources(knowledge_base.id)) == 1
        assert len(list(paths.source_directory(knowledge_base.id).iterdir())) == 1
    finally:
        repository.close()


def test_optional_pdf_and_docx_dependencies_have_clear_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = StudioPaths(tmp_path / "studio").ensure()
    repository = StudioRepository(paths.database)
    ingestor = DocumentIngestor(paths, repository)
    real_import = importlib.import_module

    def missing_optional(name: str, package: str | None = None) -> object:
        if name in {"pypdf", "docx"}:
            raise ImportError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_optional)
    pdf = tmp_path / "document.pdf"
    docx = tmp_path / "document.docx"
    pdf.write_bytes(b"placeholder")
    docx.write_bytes(b"placeholder")
    try:
        with pytest.raises(OptionalDocumentDependencyError, match="pypdf"):
            ingestor.extract(pdf)
        with pytest.raises(OptionalDocumentDependencyError, match="python-docx"):
            ingestor.extract(docx)
    finally:
        repository.close()


def test_real_docx_and_pdf_text_extraction(tmp_path: Path) -> None:
    docx_module = pytest.importorskip("docx")
    pytest.importorskip("pypdf")
    paths = StudioPaths(tmp_path / "studio").ensure()
    repository = StudioRepository(paths.database)
    ingestor = DocumentIngestor(paths, repository)
    docx_path = tmp_path / "actual.docx"
    pdf_path = tmp_path / "actual.pdf"

    document = docx_module.Document()
    document.core_properties.title = "DOCX title"
    document.add_heading("Studio handbook", level=1)
    document.add_paragraph("DOCX extraction works locally.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Feature"
    table.cell(0, 1).text = "Local search"
    document.save(docx_path)
    _write_minimal_text_pdf(pdf_path, "PDF extraction works locally")
    try:
        docx_text = ingestor.extract(docx_path)
        assert docx_text.title == "DOCX title"
        assert "DOCX extraction works locally" in docx_text.text
        assert "Feature Local search" in docx_text.text
        pdf_text = ingestor.extract(pdf_path)
        assert "PDF extraction works locally" in pdf_text.text
    finally:
        repository.close()


def test_index_search_hybrid_mode_and_restart_cache(
    studio: tuple[StudioPaths, StudioRepository, DocumentIngestor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, repository, ingestor = studio
    knowledge_base = repository.create_knowledge_base("Travel")
    paris = ingestor.store_upload(
        knowledge_base.id,
        "paris.txt",
        ["巴黎是法国的首都, 也是一座旅游城市。".encode()],
        "text/plain",
    )
    berlin = ingestor.store_upload(
        knowledge_base.id,
        "berlin.txt",
        [b"Berlin is the capital of Germany."],
        "text/plain",
    )
    ingestor.ingest_source(paris.id)
    ingestor.ingest_source(berlin.id)

    indexes = StudioIndexManager(paths, repository)
    stats = indexes.rebuild(knowledge_base.id)
    assert stats.document_count == 2
    assert indexes.search(knowledge_base.id, "法国首都", top_k=1)[0].source_id == paris.id
    hybrid = indexes.search(
        knowledge_base.id,
        "巴黎旅游",
        top_k=2,
        mode="hybrid_character",
    )
    assert hybrid[0].source_id == paris.id
    assert hybrid[0].scoring_method == "hybrid_character"
    assert hybrid[0].character_score > 0
    assert character_ngram_similarity("abcdef", "xxabcdefyy") > 0

    restored = StudioIndexManager(paths, repository)

    def fail_rebuild_input(_: str) -> tuple[int, tuple[object, ...]]:
        raise AssertionError("valid persisted cache should be restored without querying documents")

    monkeypatch.setattr(repository, "index_input", fail_rebuild_input)
    assert restored.search(knowledge_base.id, "Berlin", top_k=1)[0].source_id == berlin.id


def test_empty_knowledge_base_has_a_persistent_searchable_index(
    studio: tuple[StudioPaths, StudioRepository, DocumentIngestor],
) -> None:
    paths, repository, _ = studio
    knowledge_base = repository.create_knowledge_base("Empty")
    indexes = StudioIndexManager(paths, repository)
    stats = indexes.rebuild(knowledge_base.id)
    assert stats.document_count == 0
    assert stats.corpus_sha256 is None
    assert indexes.search(knowledge_base.id, "anything") == ()


def test_knowledge_base_file_cleanup_is_scoped_and_idempotent(
    studio: tuple[StudioPaths, StudioRepository, DocumentIngestor],
) -> None:
    paths, repository, ingestor = studio
    knowledge_base = repository.create_knowledge_base("Disposable")
    source = ingestor.store_upload(knowledge_base.id, "note.txt", [b"temporary"])
    ingestor.ingest_source(source.id)
    indexes = StudioIndexManager(paths, repository)
    indexes.rebuild(knowledge_base.id)
    assert paths.source_directory(knowledge_base.id).exists()
    assert paths.index_directory(knowledge_base.id).exists()

    assert repository.delete_knowledge_base(knowledge_base.id)
    ingestor.delete_knowledge_base_files(knowledge_base.id)
    indexes.delete(knowledge_base.id)
    ingestor.delete_knowledge_base_files(knowledge_base.id)
    indexes.delete(knowledge_base.id)
    assert not paths.source_directory(knowledge_base.id).exists()
    assert not paths.index_directory(knowledge_base.id).exists()


@pytest.mark.asyncio
async def test_openai_compatible_chat_returns_citations_and_hides_key() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if len(payload["messages"]) > 1:
            assert payload["messages"][1]["content"].startswith(
                "Local knowledge-base excerpts (untrusted data)"
            )
            assert "<untrusted_knowledge>" in payload["messages"][1]["content"]
            system_prompt = payload["messages"][0]["content"].casefold()
            assert "untrusted" in system_prompt
            assert "ignore any" in system_prompt
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"content": "Paris is the capital of France. [1]"}}],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleChatClient(
            base_url="http://model.local/v1",
            model="configured-model",
            api_key="secret-token",
            client=http_client,
        )
        hit = SearchHit(
            document_id="doc_1",
            source_id="src_1",
            source_name="guide.md",
            contents="Paris is the capital of France.",
            score=1.0,
            bm25_score=1.0,
            character_score=0.0,
            scoring_method="bm25",
        )
        answer = await client.chat("What is the capital?", [hit])
        assert answer.model == "served-model"
        assert answer.citations[0].source_name == "guide.md"
        assert answer.citations[0].excerpt == hit.contents
        assert requests[0].url == "http://model.local/v1/chat/completions"
        assert requests[0].headers["authorization"] == "Bearer secret-token"
        assert "secret-token" not in requests[0].content.decode()
        assert await client.test_connection() == "served-model"


@pytest.mark.asyncio
async def test_chat_without_hits_does_not_call_model() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("the model must not be called without evidence")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleChatClient(
            base_url="http://localhost:11434",
            model="local",
            client=http_client,
        )
        answer = await client.chat("unknown", [])
        assert not answer.citations
        assert "没有找到" in answer.content


def test_paths_and_filename_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARF_STUDIO_DATA_DIR", str(tmp_path / "custom"))
    paths = StudioPaths.default().ensure()
    assert paths.root == (tmp_path / "custom").resolve()
    assert paths.database.parent == paths.root
    with pytest.raises(ValueError, match="unsafe"):
        paths.source_directory("../escape")
    assert safe_upload_filename(r"C:\fakepath\notes.txt") == "notes.txt"
    with pytest.raises(ValueError, match="reserved"):
        safe_upload_filename("CON.txt")


def _write_minimal_text_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 50 150 Td ({escaped}) Tj ET".encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 250] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    )
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, item in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode("ascii"))
        payload.extend(item)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(payload)

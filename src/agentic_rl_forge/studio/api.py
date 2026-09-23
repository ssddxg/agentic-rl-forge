from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast
from urllib.parse import urlsplit

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from agentic_rl_forge import __version__

from .security import LocalSessionGuard, close_maybe_async

LOGGER = logging.getLogger(__name__)

KnowledgeBaseName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2_000)]
Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_096),
]
ModelName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
SearchMode = Literal["bm25", "hybrid_character"]


class StudioAPIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class KnowledgeBaseCreate(_StrictModel):
    name: KnowledgeBaseName
    description: Description = ""


class KnowledgeBaseUpdate(_StrictModel):
    name: KnowledgeBaseName | None = None
    description: Description | None = None

    @model_validator(mode="after")
    def changed_fields_must_not_be_null(self) -> KnowledgeBaseUpdate:
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null")
        return self


class SearchRequest(_StrictModel):
    query: Question
    top_k: int = Field(default=5, alias="topK", ge=1, le=20)
    mode: SearchMode = "hybrid_character"


class AnswerRequest(_StrictModel):
    question: Question
    top_k: int = Field(default=5, alias="topK", ge=1, le=20)
    mode: SearchMode = "hybrid_character"


class ModelSettingsUpdate(_StrictModel):
    base_url: str = Field(alias="baseUrl", min_length=1, max_length=2_048)
    model: ModelName
    api_key: SecretStr | None = Field(default=None, alias="apiKey", max_length=8_192)
    clear_api_key: bool = Field(default=False, alias="clearApiKey")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, alias="maxTokens", ge=1, le=32_768)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        try:
            parsed = urlsplit(normalized)
            port = parsed.port
        except ValueError as error:
            raise ValueError("baseUrl must be a valid HTTP(S) URL") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65_535)
        ):
            raise ValueError("baseUrl must be a valid HTTP(S) URL without credentials")
        return normalized

    @field_validator("api_key")
    @classmethod
    def reject_empty_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("apiKey cannot be empty; use clearApiKey instead")
        return value


class _ChatClient(Protocol):
    async def chat(
        self,
        question: str,
        hits: Any,
    ) -> Any: ...

    async def test_connection(self) -> str: ...


ChatClientFactory = Callable[[Any, str | None], _ChatClient]


@dataclass(slots=True)
class StudioServices:
    repository: Any
    ingestor: Any
    indexes: Any
    chat_client_factory: ChatClientFactory
    session_guard: LocalSessionGuard
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    job_runner: StudioJobRunner | None = None
    rl_workspace: Any = None
    rl_job_runner: Any = None


class StudioJobRunner:
    """Run ingestion jobs without tying their lifetime to an HTTP connection."""

    def __init__(self, services: StudioServices) -> None:
        self._services = services
        self._tasks: set[asyncio.Task[None]] = set()
        self._active_sources: set[str] = set()
        self._active_knowledge_bases: dict[str, int] = {}
        self._closing = False

    def enqueue(
        self, knowledge_base_id: str, jobs_and_sources: tuple[tuple[Any, Any], ...]
    ) -> None:
        if self._closing:
            raise RuntimeError("Studio is shutting down")
        source_ids = {str(_value(source, "id")) for _, source in jobs_and_sources}
        self._active_sources.update(source_ids)
        self._active_knowledge_bases[knowledge_base_id] = (
            self._active_knowledge_bases.get(knowledge_base_id, 0) + 1
        )
        task = asyncio.create_task(
            self._run(knowledge_base_id, jobs_and_sources),
            name=f"studio-ingest-{knowledge_base_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)

    def source_is_active(self, source_id: str) -> bool:
        return source_id in self._active_sources

    def knowledge_base_is_active(self, knowledge_base_id: str) -> bool:
        return self._active_knowledge_bases.get(knowledge_base_id, 0) > 0

    async def close(self) -> None:
        self._closing = True
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            LOGGER.exception("Unexpected Studio ingestion task failure")

    async def _run(
        self,
        knowledge_base_id: str,
        jobs_and_sources: tuple[tuple[Any, Any], ...],
    ) -> None:
        successful: list[tuple[Any, Any]] = []
        try:
            async with self._services.operation_lock:
                for job, source in jobs_and_sources:
                    job_id = str(_value(job, "id"))
                    try:
                        await asyncio.to_thread(
                            self._services.repository.update_job,
                            job_id,
                            status="running",
                            error=None,
                        )
                        await asyncio.to_thread(
                            self._services.ingestor.ingest_source,
                            str(_value(source, "id")),
                        )
                        successful.append((job, source))
                    except Exception:
                        LOGGER.exception("Studio failed to ingest source %s", _value(source, "id"))
                        await self._mark_failed(job_id, "文件处理失败, 请检查格式后重试。")

                if successful:
                    try:
                        await asyncio.to_thread(self._services.indexes.rebuild, knowledge_base_id)
                    except Exception:
                        LOGGER.exception(
                            "Studio failed to rebuild knowledge base %s", knowledge_base_id
                        )
                        for job, _ in successful:
                            await self._mark_failed(
                                str(_value(job, "id")),
                                "索引构建失败, 请稍后重试。",
                            )
                    else:
                        for job, _ in successful:
                            await asyncio.to_thread(
                                self._services.repository.update_job,
                                str(_value(job, "id")),
                                status="succeeded",
                                error=None,
                            )
        finally:
            for _, source in jobs_and_sources:
                self._active_sources.discard(str(_value(source, "id")))
            remaining = self._active_knowledge_bases.get(knowledge_base_id, 1) - 1
            if remaining <= 0:
                self._active_knowledge_bases.pop(knowledge_base_id, None)
            else:
                self._active_knowledge_bases[knowledge_base_id] = remaining

    async def _mark_failed(self, job_id: str, message: str) -> None:
        try:
            await asyncio.to_thread(
                self._services.repository.update_job,
                job_id,
                status="failed",
                error=message,
            )
        except Exception:
            LOGGER.exception("Studio could not mark job %s as failed", job_id)


router = APIRouter()


def _services(request: Request) -> StudioServices:
    try:
        return cast(StudioServices, request.app.state.studio)
    except AttributeError as error:
        raise RuntimeError("Studio lifespan has not started") from error


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _knowledge_base_payload(item: Any) -> dict[str, object]:
    return {
        "id": str(_value(item, "id")),
        "name": str(_value(item, "name")),
        "description": str(_value(item, "description", "")),
        "createdAt": _iso(_value(item, "created_at")),
        "updatedAt": _iso(_value(item, "updated_at")),
    }


def _source_payload(item: Any) -> dict[str, object]:
    return {
        "id": str(_value(item, "id")),
        "knowledgeBaseId": str(_value(item, "knowledge_base_id")),
        "originalName": str(_value(item, "original_name")),
        "mediaType": str(_value(item, "media_type", "application/octet-stream")),
        "sizeBytes": int(_value(item, "size_bytes", 0)),
        "status": str(_value(item, "status")),
        "error": _value(item, "error"),
        "documentCount": int(_value(item, "document_count", 0)),
        "createdAt": _iso(_value(item, "created_at")),
        "updatedAt": _iso(_value(item, "updated_at")),
    }


def _job_payload(item: Any) -> dict[str, object]:
    source_id = _value(item, "source_id")
    return {
        "id": str(_value(item, "id")),
        "knowledgeBaseId": str(_value(item, "knowledge_base_id")),
        "sourceId": str(source_id) if source_id is not None else None,
        "kind": str(_value(item, "kind")),
        "status": str(_value(item, "status")),
        "error": _value(item, "error"),
        "createdAt": _iso(_value(item, "created_at")),
        "updatedAt": _iso(_value(item, "updated_at")),
    }


def _search_hit_payload(item: Any) -> dict[str, object]:
    return {
        "documentId": str(_value(item, "document_id")),
        "sourceId": str(_value(item, "source_id")),
        "sourceName": str(_value(item, "source_name")),
        "contents": str(_value(item, "contents")),
        "score": float(_value(item, "score", 0.0)),
        "bm25Score": float(_value(item, "bm25_score", 0.0)),
        "characterScore": float(_value(item, "character_score", 0.0)),
        "scoringMethod": str(_value(item, "scoring_method", "bm25")),
    }


def _citation_payload(item: Any) -> dict[str, object]:
    return {
        "index": int(_value(item, "index", 0)),
        "documentId": str(_value(item, "document_id")),
        "sourceId": str(_value(item, "source_id")),
        "sourceName": str(_value(item, "source_name")),
        "excerpt": str(_value(item, "excerpt")),
    }


def _model_settings_payload(item: Any) -> dict[str, object]:
    # Deliberately enumerate safe fields. Never serialize a backend model wholesale here.
    return {
        "baseUrl": str(_value(item, "base_url", "")),
        "model": str(_value(item, "model", "")),
        "temperature": float(_value(item, "temperature", 0.2)),
        "maxTokens": int(_value(item, "max_tokens", 1024)),
        "apiKeyConfigured": bool(_value(item, "api_key_configured", False)),
        "updatedAt": _iso(_value(item, "updated_at")),
    }


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


def _generic_payload(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return _generic_payload(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _generic_payload(asdict(value))
    if isinstance(value, dict):
        return {_camel(str(key)): _generic_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_generic_payload(item) for item in value]
    return value


async def _get_knowledge_base(services: StudioServices, knowledge_base_id: str) -> Any:
    item = await asyncio.to_thread(services.repository.get_knowledge_base, knowledge_base_id)
    if item is None:
        raise StudioAPIError(404, "knowledge_base_not_found", "找不到这个知识库。")
    return item


async def _get_source(services: StudioServices, knowledge_base_id: str, source_id: str) -> Any:
    item = await asyncio.to_thread(services.repository.get_source, source_id)
    if item is None or str(_value(item, "knowledge_base_id")) != knowledge_base_id:
        raise StudioAPIError(404, "source_not_found", "找不到这个文件。")
    return item


def _safe_filename(filename: str | None) -> str:
    normalized = (filename or "").replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1].strip()
    if not basename or basename in {".", ".."} or "\x00" in basename:
        raise StudioAPIError(422, "invalid_filename", "请选择名称有效的文件。")
    return basename


def _upload_chunks(upload: UploadFile, maximum_bytes: int) -> Iterable[bytes]:
    total = 0
    while chunk := upload.file.read(1024 * 1024):
        total += len(chunk)
        if total > maximum_bytes:
            raise StudioAPIError(
                413,
                "file_too_large",
                f"文件不能超过 {maximum_bytes // (1024 * 1024)} MB。",
            )
        yield chunk


@router.get("/bootstrap")
async def bootstrap(request: Request, response: Response) -> dict[str, object]:
    services = _services(request)
    existing = request.cookies.get(services.session_guard.cookie_name)
    session_id, csrf_token = services.session_guard.bootstrap(existing)
    response.set_cookie(
        services.session_guard.cookie_name,
        session_id,
        max_age=services.session_guard.ttl_seconds,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    settings = await asyncio.to_thread(services.repository.get_model_settings)
    used_storage_bytes = await asyncio.to_thread(services.repository.total_source_bytes)
    allowed = sorted(str(item).lower() for item in services.ingestor.allowed_extensions)
    return {
        "app": {"name": "AgenticRLForge RL Studio", "version": __version__},
        "security": {"csrfToken": csrf_token},
        "capabilities": {
            "maxUploadBytes": int(services.ingestor.max_file_bytes),
            "maxTotalBytes": int(services.ingestor.max_total_bytes),
            "usedStorageBytes": int(used_storage_bytes),
            "supportedExtensions": allowed,
            "searchModes": ["hybrid_character", "bm25"],
        },
        "modelSettings": _model_settings_payload(settings),
    }


@router.get("/knowledge-bases")
async def list_knowledge_bases(request: Request) -> dict[str, object]:
    services = _services(request)
    items = await asyncio.to_thread(services.repository.list_knowledge_bases)
    return {"items": [_knowledge_base_payload(item) for item in items]}


@router.post("/knowledge-bases", status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    request: Request,
    payload: KnowledgeBaseCreate,
) -> dict[str, object]:
    services = _services(request)
    item = await asyncio.to_thread(
        services.repository.create_knowledge_base,
        payload.name,
        payload.description,
    )
    return _knowledge_base_payload(item)


@router.get("/knowledge-bases/{knowledge_base_id}")
async def get_knowledge_base(request: Request, knowledge_base_id: str) -> dict[str, object]:
    return _knowledge_base_payload(await _get_knowledge_base(_services(request), knowledge_base_id))


@router.patch("/knowledge-bases/{knowledge_base_id}")
async def update_knowledge_base(
    request: Request,
    knowledge_base_id: str,
    payload: KnowledgeBaseUpdate,
) -> dict[str, object]:
    services = _services(request)
    await _get_knowledge_base(services, knowledge_base_id)
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        raise StudioAPIError(422, "no_changes", "请至少修改一个字段。")
    item = await asyncio.to_thread(
        services.repository.update_knowledge_base,
        knowledge_base_id,
        **changes,
    )
    if item is None:
        raise StudioAPIError(404, "knowledge_base_not_found", "找不到这个知识库。")
    return _knowledge_base_payload(item)


@router.delete("/knowledge-bases/{knowledge_base_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(request: Request, knowledge_base_id: str) -> Response:
    services = _services(request)
    runner = services.job_runner
    async with services.operation_lock:
        await _get_knowledge_base(services, knowledge_base_id)
        if runner is not None and runner.knowledge_base_is_active(knowledge_base_id):
            raise StudioAPIError(
                409,
                "knowledge_base_busy",
                "知识库正在处理文件, 请稍后再删除。",
            )
        sources = await asyncio.to_thread(services.repository.list_sources, knowledge_base_id)
        deleted = await asyncio.to_thread(
            services.repository.delete_knowledge_base,
            knowledge_base_id,
        )
        if not deleted:
            raise StudioAPIError(404, "knowledge_base_not_found", "找不到这个知识库。")
        for source in sources:
            await _remove_source_file(services, source)
        await _cleanup_knowledge_base_files(services, knowledge_base_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/knowledge-bases/{knowledge_base_id}/sources")
async def list_sources(request: Request, knowledge_base_id: str) -> dict[str, object]:
    services = _services(request)
    await _get_knowledge_base(services, knowledge_base_id)
    items = await asyncio.to_thread(services.repository.list_sources, knowledge_base_id)
    return {"items": [_source_payload(item) for item in items]}


@router.post(
    "/knowledge-bases/{knowledge_base_id}/sources",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_source(
    request: Request,
    knowledge_base_id: str,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, object]:
    services = _services(request)
    await _get_knowledge_base(services, knowledge_base_id)
    filename = _safe_filename(file.filename)
    extension = Path(filename).suffix.lower()
    allowed = {str(item).lower() for item in services.ingestor.allowed_extensions}
    if extension not in allowed:
        raise StudioAPIError(
            415,
            "unsupported_file_type",
            f"暂不支持 {extension or '无扩展名'} 文件。",
            details={"supportedExtensions": sorted(allowed)},
        )
    maximum_bytes = int(services.ingestor.max_file_bytes)
    if file.size is not None and file.size > maximum_bytes:
        raise StudioAPIError(
            413,
            "file_too_large",
            f"文件不能超过 {maximum_bytes // (1024 * 1024)} MB。",
        )

    source: Any = None
    job: Any = None
    try:
        async with services.operation_lock:
            await _get_knowledge_base(services, knowledge_base_id)
            try:
                source = await asyncio.to_thread(
                    services.ingestor.store_upload,
                    knowledge_base_id,
                    filename,
                    _upload_chunks(file, maximum_bytes),
                    file.content_type or "application/octet-stream",
                )
            except StudioAPIError:
                raise
            except ValueError as error:
                normalized = str(error).lower()
                if "already present" in normalized:
                    raise StudioAPIError(
                        409,
                        "duplicate_source",
                        "这个文件已经在知识库中。",
                    ) from None
                if "storage limit" in normalized:
                    raise StudioAPIError(
                        507,
                        "storage_limit",
                        "Studio 存储空间已达到上限, 请删除不用的文件后重试。",
                    ) from None
                raise StudioAPIError(
                    422,
                    "invalid_source",
                    "文件无法保存, 请检查文件内容后重试。",
                ) from None

            try:
                job = await asyncio.to_thread(
                    services.repository.create_job,
                    knowledge_base_id,
                    kind="ingest",
                    source_id=str(_value(source, "id")),
                )
            except Exception:
                LOGGER.exception("Studio could not create ingestion job")
                deleted = await asyncio.to_thread(
                    services.repository.delete_source,
                    str(_value(source, "id")),
                )
                if deleted is not None:
                    await _remove_source_file(services, deleted)
                raise StudioAPIError(
                    500,
                    "job_create_failed",
                    "无法开始处理文件, 请重试。",
                ) from None

            runner = services.job_runner
            if runner is None:
                raise RuntimeError("Studio job runner is not configured")
            runner.enqueue(knowledge_base_id, ((job, source),))
    finally:
        await file.close()
    return {"source": _source_payload(source), "job": _job_payload(job)}


@router.delete(
    "/knowledge-bases/{knowledge_base_id}/sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_source(request: Request, knowledge_base_id: str, source_id: str) -> Response:
    services = _services(request)
    runner = services.job_runner
    async with services.operation_lock:
        await _get_source(services, knowledge_base_id, source_id)
        if runner is not None and runner.source_is_active(source_id):
            raise StudioAPIError(409, "source_busy", "文件正在处理中, 请稍后再删除。")
        deleted = await asyncio.to_thread(services.repository.delete_source, source_id)
        if deleted is None:
            raise StudioAPIError(404, "source_not_found", "找不到这个文件。")
        await _remove_source_file(services, deleted)
        try:
            await asyncio.to_thread(services.indexes.rebuild, knowledge_base_id)
        except Exception:
            LOGGER.exception("Studio could not eagerly rebuild after deleting source %s", source_id)
            services.indexes.invalidate(knowledge_base_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/knowledge-bases/{knowledge_base_id}/sources/{source_id}/reindex")
async def reindex_source(
    request: Request,
    knowledge_base_id: str,
    source_id: str,
) -> dict[str, object]:
    services = _services(request)
    runner = services.job_runner
    if runner is None:
        raise RuntimeError("Studio job runner is not configured")
    async with services.operation_lock:
        source = await _get_source(services, knowledge_base_id, source_id)
        if runner.source_is_active(source_id):
            raise StudioAPIError(409, "source_busy", "文件已经在处理。")
        job = await asyncio.to_thread(
            services.repository.create_job,
            knowledge_base_id,
            kind="reindex",
            source_id=source_id,
        )
        runner.enqueue(knowledge_base_id, ((job, source),))
    return {"job": _job_payload(job)}


@router.post("/knowledge-bases/{knowledge_base_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_knowledge_base(
    request: Request,
    knowledge_base_id: str,
) -> dict[str, object]:
    services = _services(request)
    runner = services.job_runner
    if runner is None:
        raise RuntimeError("Studio job runner is not configured")
    async with services.operation_lock:
        await _get_knowledge_base(services, knowledge_base_id)
        if runner.knowledge_base_is_active(knowledge_base_id):
            raise StudioAPIError(409, "knowledge_base_busy", "知识库已经在处理文件。")
        sources = await asyncio.to_thread(services.repository.list_sources, knowledge_base_id)
        if not sources:
            raise StudioAPIError(409, "knowledge_base_empty", "请先上传文件。")
        jobs_and_sources: list[tuple[Any, Any]] = []
        for source in sources:
            job = await asyncio.to_thread(
                services.repository.create_job,
                knowledge_base_id,
                kind="reindex",
                source_id=str(_value(source, "id")),
            )
            jobs_and_sources.append((job, source))
        runner.enqueue(knowledge_base_id, tuple(jobs_and_sources))
    return {"jobs": [_job_payload(job) for job, _ in jobs_and_sources]}


@router.get("/jobs")
async def list_jobs(
    request: Request,
    knowledge_base_id: Annotated[str | None, Query(alias="knowledgeBaseId")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, object]:
    services = _services(request)
    if knowledge_base_id is not None:
        await _get_knowledge_base(services, knowledge_base_id)
    items = await asyncio.to_thread(
        services.repository.list_jobs,
        knowledge_base_id,
        limit=limit,
    )
    return {"items": [_job_payload(item) for item in items]}


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict[str, object]:
    item = await asyncio.to_thread(_services(request).repository.get_job, job_id)
    if item is None:
        raise StudioAPIError(404, "job_not_found", "找不到这个任务。")
    return _job_payload(item)


@router.post("/knowledge-bases/{knowledge_base_id}/search")
async def search(
    request: Request,
    knowledge_base_id: str,
    payload: SearchRequest,
) -> dict[str, object]:
    services = _services(request)
    await _get_knowledge_base(services, knowledge_base_id)
    hits = await asyncio.to_thread(
        services.indexes.search,
        knowledge_base_id,
        payload.query,
        top_k=payload.top_k,
        mode=payload.mode,
    )
    return {"query": payload.query, "items": [_search_hit_payload(item) for item in hits]}


@router.post("/knowledge-bases/{knowledge_base_id}/answer")
async def answer(
    request: Request,
    knowledge_base_id: str,
    payload: AnswerRequest,
) -> dict[str, object]:
    services = _services(request)
    await _get_knowledge_base(services, knowledge_base_id)
    hits = await asyncio.to_thread(
        services.indexes.search,
        knowledge_base_id,
        payload.question,
        top_k=payload.top_k,
        mode=payload.mode,
    )
    if not hits:
        raise StudioAPIError(422, "no_search_results", "知识库中没有找到相关内容。")
    settings = await asyncio.to_thread(services.repository.get_model_settings)
    if (
        not str(_value(settings, "base_url", "")).strip()
        or not str(_value(settings, "model", "")).strip()
    ):
        raise StudioAPIError(409, "model_not_configured", "请先配置问答模型。")
    api_key = await asyncio.to_thread(services.repository.get_model_api_key)
    client = services.chat_client_factory(settings, api_key)
    try:
        result = await client.chat(payload.question, hits)
    except Exception:
        LOGGER.exception("Studio model request failed")
        raise StudioAPIError(
            502,
            "model_request_failed",
            "模型服务暂时不可用, 请检查地址、模型名称和密钥。",
        ) from None
    finally:
        await close_maybe_async(client)
    return {
        "content": str(_value(result, "content")),
        "citations": [_citation_payload(item) for item in _value(result, "citations", ())],
        "model": str(_value(result, "model")),
        "searchHits": [_search_hit_payload(item) for item in hits],
    }


@router.get("/model-settings")
async def get_model_settings(request: Request) -> dict[str, object]:
    item = await asyncio.to_thread(_services(request).repository.get_model_settings)
    return _model_settings_payload(item)


@router.put("/model-settings")
async def update_model_settings(
    request: Request,
    payload: ModelSettingsUpdate,
) -> dict[str, object]:
    if payload.clear_api_key and payload.api_key is not None:
        raise StudioAPIError(
            422,
            "conflicting_api_key_update",
            "不能同时填写新密钥和清除密钥。",
        )
    services = _services(request)
    current = await asyncio.to_thread(services.repository.get_model_settings)
    if payload.clear_api_key:
        api_key: str | None = None
    elif payload.api_key is not None:
        api_key = payload.api_key.get_secret_value().strip()
    else:
        api_key = None
    item = await asyncio.to_thread(
        services.repository.save_model_settings,
        base_url=payload.base_url,
        model=payload.model,
        api_key=api_key,
        clear_api_key=payload.clear_api_key,
        temperature=(
            payload.temperature
            if payload.temperature is not None
            else float(_value(current, "temperature", 0.2))
        ),
        max_tokens=(
            payload.max_tokens
            if payload.max_tokens is not None
            else int(_value(current, "max_tokens", 1024))
        ),
    )
    return _model_settings_payload(item)


@router.post("/model-settings/test")
async def test_model_settings(request: Request) -> dict[str, object]:
    services = _services(request)
    settings = await asyncio.to_thread(services.repository.get_model_settings)
    if (
        not str(_value(settings, "base_url", "")).strip()
        or not str(_value(settings, "model", "")).strip()
    ):
        raise StudioAPIError(409, "model_not_configured", "请先保存模型配置。")
    api_key = await asyncio.to_thread(services.repository.get_model_api_key)
    client = services.chat_client_factory(settings, api_key)
    try:
        model = await client.test_connection()
    except Exception:
        LOGGER.exception("Studio model connection test failed")
        raise StudioAPIError(
            502,
            "model_connection_failed",
            "连接失败, 请检查地址、模型名称和密钥。",
        ) from None
    finally:
        await close_maybe_async(client)
    return {"ok": True, "model": model, "message": "连接成功。"}


async def _remove_source_file(services: StudioServices, source: Any) -> None:
    remover = getattr(services.ingestor, "delete_source_file", None)
    if remover is None:
        return
    try:
        await asyncio.to_thread(remover, source)
    except FileNotFoundError:
        return
    except Exception:
        LOGGER.exception("Studio could not remove source file %s", _value(source, "id"))


async def _cleanup_knowledge_base_files(
    services: StudioServices,
    knowledge_base_id: str,
) -> None:
    cleanup_steps = (
        getattr(services.ingestor, "delete_knowledge_base_files", None),
        getattr(services.indexes, "delete", None),
    )
    for cleanup in cleanup_steps:
        if cleanup is None:
            continue
        try:
            await asyncio.to_thread(cleanup, knowledge_base_id)
        except Exception:
            LOGGER.exception(
                "Studio could not remove files for knowledge base %s",
                knowledge_base_id,
            )

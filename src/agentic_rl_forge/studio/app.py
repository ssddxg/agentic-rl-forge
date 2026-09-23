from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentic_rl_forge import __version__

from .api import StudioAPIError, StudioJobRunner, StudioServices, router
from .chat import OpenAICompatibleChatClient
from .indexes import StudioIndexManager
from .ingestion import DEFAULT_MAX_UPLOAD_BYTES, DocumentIngestor
from .paths import StudioPaths
from .repository import StudioRepository
from .rl import MAX_DATASET_FILE_BYTES, RLJobRunner, RLWorkspace
from .rl import router as rl_router
from .security import (
    LOCAL_HOSTS,
    LocalAPISecurityMiddleware,
    LocalSessionGuard,
    RequestContextMiddleware,
    UploadBodyLimitMiddleware,
    close_maybe_async,
)

LOGGER = logging.getLogger(__name__)
_MULTIPART_OVERHEAD_ALLOWANCE = 1024 * 1024
_STUDIO_CSP = "; ".join(
    (
        "default-src 'self'",
        "base-uri 'none'",
        "connect-src 'self'",
        "font-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "img-src 'self' data:",
        "object-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
    )
)


class StudioBrowserHeadersMiddleware:
    """Apply browser hardening and explicit cache behavior to Studio responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault(
                    "Permissions-Policy",
                    "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
                )
                headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
                if path.startswith("/api/") or path == "/api":
                    headers.setdefault("Cache-Control", "no-store")
                else:
                    headers.setdefault("Cache-Control", "no-cache")
                    headers.setdefault("Content-Security-Policy", _STUDIO_CSP)
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _static_directory() -> Path:
    directory = Path(str(resources.files("agentic_rl_forge.studio").joinpath("static")))
    if not directory.is_dir():
        raise RuntimeError("Studio static assets are missing from the installation")
    return directory


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    return request_id if isinstance(request_id, str) else "unknown"


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: object | None = None,
) -> JSONResponse:
    error: dict[str, object] = {
        "code": code,
        "message": message,
        "requestId": _request_id(request),
    }
    if details is not None:
        error["details"] = details
    return JSONResponse(
        {"error": error},
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def _validation_details(error: RequestValidationError) -> list[dict[str, object]]:
    # Do not include Pydantic's `input` field: it may contain an API key.
    return [
        {
            "location": [str(item) for item in issue.get("loc", ())],
            "message": str(issue.get("msg", "Invalid value")),
            "type": str(issue.get("type", "value_error")),
        }
        for issue in error.errors()
    ]


def _default_services(paths: StudioPaths, session_guard: LocalSessionGuard) -> StudioServices:
    repository = StudioRepository(paths.database)
    ingestor = DocumentIngestor(paths, repository)
    indexes = StudioIndexManager(paths, repository)

    def chat_client_factory(settings: Any, api_key: str | None) -> OpenAICompatibleChatClient:
        return OpenAICompatibleChatClient(
            base_url=settings.base_url,
            model=settings.model,
            api_key=api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    return StudioServices(
        repository=repository,
        ingestor=ingestor,
        indexes=indexes,
        chat_client_factory=chat_client_factory,
        session_guard=session_guard,
        rl_workspace=RLWorkspace(paths.root / "rl").ensure(),
    )


def create_studio_app(
    data_dir: Path | str | None = None,
    *,
    services: StudioServices | None = None,
    allow_network: bool = False,
) -> FastAPI:
    """Create the local-only Studio API application.

    Passing ``services`` is primarily useful for embedding and tests. The application owns those
    resources for its lifespan and closes them on shutdown. Network access is opt-in and still
    requires same-origin requests plus a bootstrap-issued CSRF token.
    """

    if services is not None and data_dir is not None:
        raise ValueError("data_dir and services cannot be used together")
    session_guard = services.session_guard if services is not None else LocalSessionGuard()
    configured_paths = (
        StudioPaths(Path(data_dir).expanduser().resolve())
        if data_dir is not None
        else StudioPaths.default()
    )
    configured_upload_limit = (
        int(services.ingestor.max_file_bytes) if services is not None else DEFAULT_MAX_UPLOAD_BYTES
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = services
        if runtime is None:
            paths = await asyncio.to_thread(configured_paths.ensure)
            runtime = await asyncio.to_thread(_default_services, paths, session_guard)
        try:
            recovered_jobs, recovered_sources = await asyncio.to_thread(
                runtime.repository.recover_interrupted_work
            )
            if recovered_jobs or recovered_sources:
                LOGGER.info(
                    "Recovered %d interrupted Studio jobs and %d sources",
                    recovered_jobs,
                    recovered_sources,
                )
            if runtime.rl_workspace is not None:
                await asyncio.to_thread(runtime.rl_workspace.recover_interrupted)
                runtime.rl_job_runner = RLJobRunner(runtime.rl_workspace)
            runtime.job_runner = StudioJobRunner(runtime)
            app.state.studio = runtime
            yield
        finally:
            if runtime.job_runner is not None:
                await runtime.job_runner.close()
            if runtime.rl_job_runner is not None:
                await runtime.rl_job_runner.close()
            session_guard.clear()
            await close_maybe_async(runtime.indexes)
            await close_maybe_async(runtime.ingestor)
            await close_maybe_async(runtime.repository)

    app = FastAPI(
        title="AgenticRLForge RL Studio API",
        version=__version__,
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.exception_handler(StudioAPIError)
    async def studio_error(request: Request, error: StudioAPIError) -> JSONResponse:
        return _error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            code="validation_error",
            message="请求内容不正确, 请检查后重试。",
            details={"issues": _validation_details(error)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        message = error.detail if isinstance(error.detail, str) else "请求无法完成。"
        code = "not_found" if error.status_code == 404 else "http_error"
        return _error_response(
            request,
            status_code=error.status_code,
            code=code,
            message=message,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception(
            "Unhandled Studio API error (request %s)", _request_id(request), exc_info=error
        )
        return _error_response(
            request,
            status_code=500,
            code="internal_error",
            message="操作失败, 请稍后重试。",
        )

    app.include_router(router, prefix="/api/v1")
    app.include_router(rl_router, prefix="/api/v1")
    app.include_router(router, prefix="/api", include_in_schema=False)
    app.include_router(rl_router, prefix="/api", include_in_schema=False)

    static_directory = _static_directory()
    index_file = static_directory / "index.html"
    app.mount(
        "/static",
        StaticFiles(directory=static_directory, check_dir=True),
        name="studio-static",
    )

    # The short aliases let the packaged HTML use sibling-relative assets. This keeps the real
    # server page intact while allowing a directly opened file to render one clear launch notice
    # instead of an unstyled wall of HTML.
    @app.get("/styles.css", include_in_schema=False)
    async def studio_styles() -> FileResponse:
        return FileResponse(static_directory / "styles.css", media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    async def studio_script() -> FileResponse:
        return FileResponse(static_directory / "app.js", media_type="text/javascript")

    @app.get("/icon.svg", include_in_schema=False)
    async def studio_icon() -> FileResponse:
        return FileResponse(static_directory / "icon.svg", media_type="image/svg+xml")

    @app.get("/", include_in_schema=False)
    async def studio_home() -> FileResponse:
        return FileResponse(index_file, media_type="text/html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def studio_spa_fallback(full_path: str) -> FileResponse:
        if (
            full_path == "api"
            or full_path.startswith("api/")
            or full_path == "static"
            or full_path.startswith("static/")
            or Path(full_path).suffix
        ):
            raise StarletteHTTPException(status_code=404, detail="Not found")
        return FileResponse(index_file, media_type="text/html")

    # Added inside-out so request IDs and hardening headers also cover middleware rejections.
    app.add_middleware(
        UploadBodyLimitMiddleware,
        maximum_bytes=configured_upload_limit + _MULTIPART_OVERHEAD_ALLOWANCE,
        rl_dataset_maximum_bytes=(2 * MAX_DATASET_FILE_BYTES + _MULTIPART_OVERHEAD_ALLOWANCE),
    )
    app.add_middleware(
        TrustedHostMiddleware,
        # Starlette currently splits IPv6 Host headers at the first colon and sees "[".
        # The stricter middleware immediately outside this one has already validated ::1.
        allowed_hosts=["*"] if allow_network else sorted((*LOCAL_HOSTS, "[")),
    )
    app.add_middleware(
        LocalAPISecurityMiddleware,
        session_guard=session_guard,
        local_only=not allow_network,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(StudioBrowserHeadersMiddleware)
    return app

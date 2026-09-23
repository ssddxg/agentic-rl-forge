from __future__ import annotations

import asyncio
import hmac
import inspect
import secrets
import threading
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass(slots=True)
class _Session:
    csrf_token: str
    expires_at: float
    last_seen: float


class LocalSessionGuard:
    """Small in-memory session store used only to protect the local Studio API."""

    def __init__(
        self,
        *,
        cookie_name: str = "arf_studio_session",
        ttl_seconds: int = 12 * 60 * 60,
        max_sessions: int = 1_024,
    ) -> None:
        if ttl_seconds < 60:
            raise ValueError("ttl_seconds must be at least 60")
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.cookie_name = cookie_name
        self.ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def bootstrap(self, existing_session_id: str | None = None) -> tuple[str, str]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if existing_session_id:
                current = self._sessions.get(existing_session_id)
                if current is not None and current.expires_at > now:
                    current.expires_at = now + self.ttl_seconds
                    current.last_seen = now
                    return existing_session_id, current.csrf_token

            session_id = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].last_seen)
                del self._sessions[oldest]
            self._sessions[session_id] = _Session(
                csrf_token=csrf_token,
                expires_at=now + self.ttl_seconds,
                last_seen=now,
            )
            return session_id, csrf_token

    def validate(self, session_id: str | None, csrf_token: str | None) -> bool:
        if not session_id or not csrf_token:
            return False
        now = time.monotonic()
        with self._lock:
            current = self._sessions.get(session_id)
            if current is None or current.expires_at <= now:
                self._sessions.pop(session_id, None)
                return False
            if not hmac.compare_digest(current.csrf_token, csrf_token):
                return False
            current.expires_at = now + self.ttl_seconds
            current.last_seen = now
            return True

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _prune(self, now: float) -> None:
        expired = [key for key, session in self._sessions.items() if session.expires_at <= now]
        for key in expired:
            del self._sessions[key]


def is_local_origin(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
        # Origin headers never contain path/query/fragment. Reject malformed lookalikes.
        return (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in LOCAL_HOSTS
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def _origin_matches_request(origin: str, scope: Scope, headers: Headers) -> bool:
    try:
        parsed = urlsplit(origin)
        request_host = urlsplit(f"//{headers.get('host', '')}")
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        request_scheme = str(scope.get("scheme", "http"))
        request_port = request_host.port or (443 if request_scheme == "https" else 80)
        return (
            parsed.scheme == request_scheme
            and parsed.hostname is not None
            and parsed.hostname == request_host.hostname
            and origin_port == request_port
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def _cookie_value(headers: Headers, name: str) -> str | None:
    raw_cookie = headers.get("cookie")
    if not raw_cookie:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(raw_cookie)
    except ValueError:
        return None
    morsel = cookie.get(name)
    return morsel.value if morsel is not None else None


def _request_id(scope: Scope) -> str:
    state = scope.setdefault("state", {})
    current = state.get("request_id")
    if isinstance(current, str):
        return current
    generated = secrets.token_hex(12)
    state["request_id"] = generated
    return generated


def _security_error(
    scope: Scope,
    code: str,
    message: str,
    status_code: int,
    *,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    error: dict[str, object] = {
        "code": code,
        "message": message,
        "requestId": _request_id(scope),
    }
    if details is not None:
        error["details"] = details
    return JSONResponse(
        {"error": error},
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def _local_host(host_header: str | None) -> bool:
    if not host_header:
        return False
    try:
        return urlsplit(f"//{host_header}").hostname in LOCAL_HOSTS
    except ValueError:
        return False


class LocalAPISecurityMiddleware:
    """Reject cross-site writes and require a bootstrap-issued CSRF token."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        session_guard: LocalSessionGuard,
        api_prefixes: tuple[str, ...] = ("/api/v1", "/api"),
        local_only: bool = True,
    ) -> None:
        self.app = app
        self.session_guard = session_guard
        self.api_prefixes = api_prefixes
        self.local_only = local_only

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        if not any(path == prefix or path.startswith(f"{prefix}/") for prefix in self.api_prefixes):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host = headers.get("host")
        if self.local_only and not _local_host(host):
            await _security_error(
                scope,
                "host_not_allowed",
                "Studio can only be opened from this computer.",
                400,
            )(scope, receive, send)
            return
        origin = headers.get("origin")
        origin_allowed = (
            is_local_origin(origin) and _origin_matches_request(origin, scope, headers)
            if self.local_only and origin is not None
            else origin is None or _origin_matches_request(origin, scope, headers)
        )
        if not origin_allowed:
            await _security_error(
                scope,
                "origin_not_allowed",
                "Only the local Studio origin is allowed.",
                403,
            )(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        if method in UNSAFE_METHODS:
            session_id = _cookie_value(headers, self.session_guard.cookie_name)
            csrf_token = headers.get("x-csrf-token")
            if not self.session_guard.validate(session_id, csrf_token):
                await _security_error(
                    scope,
                    "csrf_failed",
                    "Refresh Studio and try the request again.",
                    403,
                )(scope, receive, send)
                return

        await self.app(scope, receive, send)


class RequestContextMiddleware:
    """Attach a non-user-controlled request identifier to responses and error envelopes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _request_id(scope)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class UploadBodyLimitMiddleware:
    """Bound multipart request bodies before Starlette parses or spools them."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        maximum_bytes: int,
        rl_dataset_maximum_bytes: int | None = None,
    ) -> None:
        if maximum_bytes < 1:
            raise ValueError("maximum_bytes must be positive")
        if rl_dataset_maximum_bytes is not None and rl_dataset_maximum_bytes < 1:
            raise ValueError("rl_dataset_maximum_bytes must be positive")
        self.app = app
        self.maximum_bytes = maximum_bytes
        self.rl_dataset_maximum_bytes = rl_dataset_maximum_bytes or maximum_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        is_source_upload = (
            scope["type"] == "http"
            and method == "POST"
            and path.endswith("/sources")
            and (path.startswith("/api/v1/") or path.startswith("/api/"))
        )
        is_rl_dataset_upload = (
            scope["type"] == "http"
            and method == "POST"
            and path
            in {
                "/api/v1/rl/datasets",
                "/api/rl/datasets",
            }
        )
        if not is_source_upload and not is_rl_dataset_upload:
            await self.app(scope, receive, send)
            return
        maximum_bytes = (
            self.rl_dataset_maximum_bytes if is_rl_dataset_upload else self.maximum_bytes
        )

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > maximum_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                await _security_error(
                    scope,
                    "invalid_content_length",
                    "The upload has an invalid Content-Length header.",
                    400,
                )(scope, receive, send)
                return

        try:
            body = await drain_request_body(receive, maximum_bytes=maximum_bytes)
        except ValueError:
            await self._reject(scope, receive, send)
            return
        except ConnectionError:
            return
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _security_error(
            scope,
            "request_too_large",
            "The upload is larger than the configured limit.",
            413,
        )(scope, receive, send)


async def close_maybe_async(resource: Any) -> None:
    close = getattr(resource, "aclose", None)
    if close is not None:
        if inspect.iscoroutinefunction(close):
            await close()
        else:
            result = await asyncio.to_thread(close)
            if inspect.isawaitable(result):
                await result
        return
    close = getattr(resource, "close", None)
    if close is not None:
        if inspect.iscoroutinefunction(close):
            await close()
        else:
            result = await asyncio.to_thread(close)
            if inspect.isawaitable(result):
                await result


async def drain_request_body(receive: Receive, *, maximum_bytes: int) -> bytes:
    """Read an ASGI body with an enforced limit, including chunked requests."""

    body = bytearray()
    while True:
        message: Message = await receive()
        if message["type"] == "http.disconnect":
            raise ConnectionError("client disconnected while sending the request")
        if message["type"] != "http.request":
            continue
        body.extend(message.get("body", b""))
        if len(body) > maximum_bytes:
            raise ValueError("request body is too large")
        if not message.get("more_body", False):
            return bytes(body)

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentic_rl_forge.contracts import Trajectory

_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class _MetricKey:
    name: str
    labels: tuple[tuple[str, str], ...]


@dataclass(slots=True)
class _Histogram:
    buckets: tuple[float, ...]
    counts: list[int]
    count: int = 0
    total: float = 0.0


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    counters: Mapping[str, float]
    gauges: Mapping[str, float]
    histograms: Mapping[str, Mapping[str, float | int]]


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[_MetricKey, float] = {}
        self._gauges: dict[_MetricKey, float] = {}
        self._histograms: dict[_MetricKey, _Histogram] = {}
        self._help: dict[str, str] = {}
        self._types: dict[str, str] = {}

    def increment(
        self,
        name: str,
        amount: float = 1.0,
        *,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        if amount < 0:
            raise ValueError("counter increments cannot be negative")
        key = self._key(name, labels)
        with self._lock:
            self._declare(name, help_text, "counter")
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._declare(name, help_text, "gauge")
            self._gauges[key] = value

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
        buckets: Sequence[float] = _DEFAULT_BUCKETS,
        help_text: str = "",
    ) -> None:
        ordered_buckets = tuple(sorted(set(float(bucket) for bucket in buckets)))
        if not ordered_buckets or any(not math.isfinite(bucket) for bucket in ordered_buckets):
            raise ValueError("histogram buckets must contain finite values")
        key = self._key(name, labels)
        with self._lock:
            self._declare(name, help_text, "histogram")
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = _Histogram(
                    buckets=ordered_buckets,
                    counts=[0 for _ in ordered_buckets],
                )
                self._histograms[key] = histogram
            elif histogram.buckets != ordered_buckets:
                raise ValueError(f"histogram {name!r} was already declared with other buckets")
            for index, bucket in enumerate(histogram.buckets):
                if value <= bucket:
                    histogram.counts[index] += 1
            histogram.count += 1
            histogram.total += value

    def record_trajectory(self, trajectory: Trajectory) -> None:
        labels = {
            "status": trajectory.status.value,
            "origin": trajectory.provenance.origin.value,
        }
        self.increment(
            "arf_trajectories_total",
            labels=labels,
            help_text="Completed trajectories by status and origin.",
        )
        self.increment(
            "arf_generated_tokens_total",
            float(trajectory.total_generated_tokens),
            help_text="Generated rollout tokens.",
        )
        self.increment(
            "arf_observation_tokens_total",
            float(trajectory.total_observation_tokens),
            help_text="Environment observation tokens.",
        )
        self.observe(
            "arf_trajectory_reward",
            trajectory.total_reward,
            buckets=(-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0),
            help_text="Distribution of total trajectory rewards.",
        )
        if trajectory.completed_at is not None:
            self.observe(
                "arf_trajectory_duration_seconds",
                (trajectory.completed_at - trajectory.started_at).total_seconds(),
                help_text="End-to-end rollout duration in seconds.",
            )

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            counters = {self._display_key(key): value for key, value in self._counters.items()}
            gauges = {self._display_key(key): value for key, value in self._gauges.items()}
            histograms = {
                self._display_key(key): {
                    "count": histogram.count,
                    "sum": histogram.total,
                    **{
                        f"le_{bucket:g}": count
                        for bucket, count in zip(histogram.buckets, histogram.counts, strict=True)
                    },
                }
                for key, histogram in self._histograms.items()
            }
        return MetricsSnapshot(counters, gauges, histograms)

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            names = sorted(
                {key.name for key in self._counters}
                | {key.name for key in self._gauges}
                | {key.name for key in self._histograms}
            )
            for name in names:
                if self._help.get(name):
                    lines.append(f"# HELP {name} {self._escape_help(self._help[name])}")
                if any(key.name == name for key in self._counters):
                    lines.append(f"# TYPE {name} counter")
                    for key, value in sorted(
                        self._counters.items(), key=lambda item: item[0].labels
                    ):
                        if key.name == name:
                            lines.append(f"{name}{self._format_labels(key.labels)} {value:g}")
                if any(key.name == name for key in self._gauges):
                    lines.append(f"# TYPE {name} gauge")
                    for key, value in sorted(self._gauges.items(), key=lambda item: item[0].labels):
                        if key.name == name:
                            lines.append(f"{name}{self._format_labels(key.labels)} {value:g}")
                if any(key.name == name for key in self._histograms):
                    lines.append(f"# TYPE {name} histogram")
                    for key, histogram in sorted(
                        self._histograms.items(), key=lambda item: item[0].labels
                    ):
                        if key.name != name:
                            continue
                        for bucket, count in zip(histogram.buckets, histogram.counts, strict=True):
                            labels = (*key.labels, ("le", f"{bucket:g}"))
                            lines.append(f"{name}_bucket{self._format_labels(labels)} {count}")
                        labels = (*key.labels, ("le", "+Inf"))
                        lines.append(
                            f"{name}_bucket{self._format_labels(labels)} {histogram.count}"
                        )
                        lines.append(
                            f"{name}_sum{self._format_labels(key.labels)} {histogram.total:g}"
                        )
                        lines.append(
                            f"{name}_count{self._format_labels(key.labels)} {histogram.count}"
                        )
        return "\n".join(lines) + "\n"

    def _declare(self, name: str, help_text: str, metric_type: str) -> None:
        previous_type = self._types.get(name)
        if previous_type is not None and previous_type != metric_type:
            raise ValueError(f"metric {name!r} was already declared as {previous_type}")
        self._types[name] = metric_type
        previous = self._help.get(name)
        if previous and help_text and previous != help_text:
            raise ValueError(f"metric {name!r} was declared with conflicting help text")
        if help_text:
            self._help[name] = help_text

    @staticmethod
    def _key(name: str, labels: Mapping[str, str] | None) -> _MetricKey:
        if _METRIC_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid metric name {name!r}")
        normalized = []
        for label, value in sorted((labels or {}).items()):
            if _LABEL_NAME.fullmatch(label) is None:
                raise ValueError(f"invalid metric label {label!r}")
            normalized.append((label, str(value)))
        return _MetricKey(name, tuple(normalized))

    @staticmethod
    def _display_key(key: _MetricKey) -> str:
        return f"{key.name}{MetricsRegistry._format_labels(key.labels)}"

    @staticmethod
    def _format_labels(labels: Sequence[tuple[str, str]]) -> str:
        if not labels:
            return ""
        values = ",".join(
            f'{name}="{MetricsRegistry._escape_label(value)}"' for name, value in labels
        )
        return "{" + values + "}"

    @staticmethod
    def _escape_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    @staticmethod
    def _escape_help(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n")


def instrument_fastapi(
    app: Any,
    registry: MetricsRegistry,
    *,
    service: str,
) -> None:
    try:
        from fastapi import Request, Response
    except ImportError as error:
        raise RuntimeError("install the server extra to enable HTTP observability") from error

    @app.middleware("http")  # type: ignore[untyped-decorator]
    async def observe_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started_at = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            labels = {
                "service": service,
                "method": request.method,
                "path": str(path),
                "status": str(status_code),
            }
            registry.increment(
                "arf_http_requests_total",
                labels=labels,
                help_text="HTTP requests handled by AgenticRLForge services.",
            )
            registry.observe(
                "arf_http_request_duration_seconds",
                time.perf_counter() - started_at,
                labels={key: value for key, value in labels.items() if key != "status"},
                help_text="HTTP request latency in seconds.",
            )

    @app.get("/metrics", include_in_schema=False)  # type: ignore[untyped-decorator]
    async def metrics() -> Response:
        return Response(
            content=registry.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

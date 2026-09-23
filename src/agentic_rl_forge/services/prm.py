from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from agentic_rl_forge.search import ProcessRewardInput, ProcessRewardModel
from agentic_rl_forge.services.observability import MetricsRegistry, instrument_fastapi


class PRMScoreRequest(BaseModel):
    inputs: list[ProcessRewardInput] = Field(min_length=1)


def create_prm_app(
    model: ProcessRewardModel,
    *,
    metrics: MetricsRegistry | None = None,
) -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as error:
        raise RuntimeError("install the server extra to run the PRM service") from error

    registry = metrics or MetricsRegistry()
    app = FastAPI(title="AgenticRLForge Process Reward Service", version="1.0")
    instrument_fastapi(app, registry, service="prm")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model_version": model.version}

    @app.post("/score")
    async def score(request: PRMScoreRequest) -> dict[str, object]:
        started_at = time.perf_counter()
        try:
            scores = await model.score_batch(request.inputs)
        except Exception:
            registry.increment(
                "arf_prm_batches_total",
                labels={"status": "error", "model_version": model.version},
                help_text="PRM scoring batches by outcome.",
            )
            raise
        registry.increment(
            "arf_prm_batches_total",
            labels={"status": "ok", "model_version": model.version},
            help_text="PRM scoring batches by outcome.",
        )
        registry.increment(
            "arf_prm_examples_total",
            float(len(request.inputs)),
            labels={"model_version": model.version},
            help_text="Examples scored by the PRM service.",
        )
        registry.observe(
            "arf_prm_batch_duration_seconds",
            time.perf_counter() - started_at,
            labels={"model_version": model.version},
            help_text="PRM batch scoring latency in seconds.",
        )
        return {
            "model_version": model.version,
            "scores": [item.model_dump(mode="json") for item in scores],
        }

    return app

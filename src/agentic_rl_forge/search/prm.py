from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

import httpx
from pydantic import Field

from agentic_rl_forge.contracts import ContractModel, JsonObject


class ProcessRewardInput(ContractModel):
    state: JsonObject
    action: JsonObject | None = None
    history: tuple[JsonObject, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)


class ProcessScore(ContractModel):
    value: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    metadata: JsonObject = Field(default_factory=dict)

    @property
    def calibrated_value(self) -> float:
        return self.value * self.confidence


class ProcessRewardModel(Protocol):
    @property
    def version(self) -> str: ...

    async def score_batch(
        self, inputs: Sequence[ProcessRewardInput]
    ) -> tuple[ProcessScore, ...]: ...


class HeuristicProcessRewardModel:
    def __init__(
        self,
        scorer: Callable[[ProcessRewardInput], ProcessScore],
        *,
        version: str = "heuristic-prm-v1",
    ) -> None:
        self._scorer = scorer
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    async def score_batch(self, inputs: Sequence[ProcessRewardInput]) -> tuple[ProcessScore, ...]:
        return tuple(self._scorer(item) for item in inputs)


class HTTPProcessRewardModel:
    def __init__(
        self,
        *,
        endpoint: str,
        version: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)
        self._owns_client = client is None
        self._endpoint = endpoint
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    async def score_batch(self, inputs: Sequence[ProcessRewardInput]) -> tuple[ProcessScore, ...]:
        response = await self._client.post(
            self._endpoint,
            json={"inputs": [item.model_dump(mode="json") for item in inputs]},
        )
        response.raise_for_status()
        payload = response.json()
        raw_scores = payload.get("scores") if isinstance(payload, dict) else None
        if not isinstance(raw_scores, list) or len(raw_scores) != len(inputs):
            raise ValueError("PRM response must contain one score for every input")
        return tuple(ProcessScore.model_validate(score) for score in raw_scores)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class TemperatureCalibratedPRM:
    def __init__(
        self,
        inner: ProcessRewardModel,
        *,
        temperature: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._inner = inner
        self._temperature = temperature
        self._bias = bias

    @property
    def version(self) -> str:
        return f"{self._inner.version}:temp={self._temperature}:bias={self._bias}"

    async def score_batch(self, inputs: Sequence[ProcessRewardInput]) -> tuple[ProcessScore, ...]:
        scores = await self._inner.score_batch(inputs)
        calibrated = []
        for score in scores:
            probability = (score.value + 1.0) / 2.0
            probability = min(max(probability, 1e-6), 1.0 - 1e-6)
            logit = math.log(probability / (1.0 - probability))
            adjusted = 1.0 / (1.0 + math.exp(-((logit + self._bias) / self._temperature)))
            calibrated.append(
                score.model_copy(
                    update={
                        "value": 2.0 * adjusted - 1.0,
                        "metadata": {
                            **score.metadata,
                            "uncalibrated_value": score.value,
                        },
                    }
                )
            )
        return tuple(calibrated)

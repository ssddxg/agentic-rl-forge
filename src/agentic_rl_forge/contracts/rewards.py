from __future__ import annotations

from enum import Enum

from pydantic import Field

from agentic_rl_forge.contracts.base import ContractModel, JsonObject


class RewardSource(str, Enum):
    OUTCOME = "outcome"
    PROCESS = "process"
    FORMAT = "format"
    TOOL = "tool"
    COST = "cost"
    SAFETY = "safety"
    ANTI_HACKING = "anti_hacking"
    PREFERENCE = "preference"


class RewardSignal(ContractModel):
    name: str = Field(min_length=1)
    source: RewardSource
    value: float
    weight: float = 1.0
    terminal: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: JsonObject = Field(default_factory=dict)

    @property
    def weighted_value(self) -> float:
        return self.value * self.weight * self.confidence


class RewardSummary(ContractModel):
    signals: tuple[RewardSignal, ...] = ()

    @property
    def total(self) -> float:
        return sum(signal.weighted_value for signal in self.signals)

    def by_source(self) -> dict[RewardSource, float]:
        totals: dict[RewardSource, float] = {}
        for signal in self.signals:
            totals[signal.source] = totals.get(signal.source, 0.0) + signal.weighted_value
        return totals

from __future__ import annotations

from pydantic import Field, PositiveInt

from agentic_rl_forge.contracts import ContractModel, JsonObject


class BudgetSignals(ContractModel):
    policy_entropy: float = Field(default=0.0, ge=0.0)
    prm_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    task_complexity: float = Field(default=0.0, ge=0.0, le=1.0)
    recent_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: JsonObject = Field(default_factory=dict)


class SearchBudget(ContractModel):
    simulations: PositiveInt
    branch_factor: PositiveInt
    max_depth: PositiveInt
    reason: str


class AdaptiveComputeBudget:
    def __init__(
        self,
        *,
        min_simulations: int = 4,
        max_simulations: int = 64,
        min_branch_factor: int = 2,
        max_branch_factor: int = 8,
        max_depth: int = 8,
        entropy_scale: float = 4.0,
    ) -> None:
        if not 1 <= min_simulations <= max_simulations:
            raise ValueError("simulation limits are invalid")
        if not 1 <= min_branch_factor <= max_branch_factor:
            raise ValueError("branch factor limits are invalid")
        if max_depth < 1 or entropy_scale <= 0:
            raise ValueError("max_depth and entropy_scale must be positive")
        self._min_simulations = min_simulations
        self._max_simulations = max_simulations
        self._min_branch_factor = min_branch_factor
        self._max_branch_factor = max_branch_factor
        self._max_depth = max_depth
        self._entropy_scale = entropy_scale

    def allocate(self, signals: BudgetSignals) -> SearchBudget:
        normalized_entropy = min(signals.policy_entropy / self._entropy_scale, 1.0)
        difficulty = (
            0.25 * normalized_entropy
            + 0.35 * signals.prm_uncertainty
            + 0.25 * signals.task_complexity
            + 0.15 * signals.recent_failure_rate
        )
        simulations = self._interpolate(self._min_simulations, self._max_simulations, difficulty)
        branch_factor = self._interpolate(
            self._min_branch_factor, self._max_branch_factor, difficulty
        )
        return SearchBudget(
            simulations=simulations,
            branch_factor=branch_factor,
            max_depth=self._max_depth,
            reason=f"difficulty={difficulty:.4f}",
        )

    @staticmethod
    def _interpolate(lower: int, upper: int, ratio: float) -> int:
        return max(lower, min(upper, round(lower + ratio * (upper - lower))))

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    DataOrigin,
    Message,
    MessageRole,
    Provenance,
    RewardSignal,
    RewardSource,
    RewardSummary,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
)
from agentic_rl_forge.data import VerifiedRejectionSampler
from agentic_rl_forge.search import (
    ActionCandidate,
    AdaptiveComputeBudget,
    AsyncMCTS,
    BudgetSignals,
    HeuristicProcessRewardModel,
    MCTSConfig,
    ProcessRewardInput,
    ProcessScore,
    SearchTransition,
    TemperatureCalibratedPRM,
)


class NumberDomain:
    async def propose(
        self,
        state: int,
        *,
        branch_factor: int,
        seed: int,
    ) -> Sequence[ActionCandidate[int]]:
        del state, branch_factor, seed
        return (ActionCandidate(1, 0.5), ActionCandidate(2, 0.5))

    async def transition(self, state: int, action: int) -> SearchTransition[int]:
        next_state = state + action
        terminal = next_state >= 4
        return SearchTransition(
            state=next_state,
            reward=1.0 if terminal else 0.0,
            terminal=terminal,
        )

    async def evaluate(self, states: Sequence[int]) -> tuple[ProcessScore, ...]:
        return tuple(ProcessScore(value=min(state / 4.0, 1.0), confidence=1.0) for state in states)


@pytest.mark.asyncio
async def test_mcts_uses_process_values_and_terminal_rewards() -> None:
    result = await AsyncMCTS(
        NumberDomain(),
        MCTSConfig(simulations=24, branch_factor=2, max_depth=3, exploration=1.0),
    ).search(0, seed=4)

    assert result.action == 2
    assert result.simulations == 24
    assert result.expanded_nodes >= 2
    assert sum(item.visits for item in result.root_actions) == 24


def test_adaptive_budget_spends_more_compute_on_uncertain_tasks() -> None:
    allocator = AdaptiveComputeBudget(
        min_simulations=4,
        max_simulations=40,
        min_branch_factor=2,
        max_branch_factor=6,
    )
    easy = allocator.allocate(BudgetSignals())
    hard = allocator.allocate(
        BudgetSignals(
            policy_entropy=4.0,
            prm_uncertainty=1.0,
            task_complexity=1.0,
            recent_failure_rate=1.0,
        )
    )

    assert easy.simulations == 4
    assert hard.simulations == 40
    assert hard.branch_factor == 6


@pytest.mark.asyncio
async def test_temperature_calibration_preserves_metadata() -> None:
    base = HeuristicProcessRewardModel(
        lambda item: ProcessScore(value=float(item.state["value"]), metadata={"source": "raw"})
    )
    calibrated = TemperatureCalibratedPRM(base, temperature=1.0, bias=1.0)

    score = (await calibrated.score_batch((ProcessRewardInput(state={"value": 0.0}),)))[0]

    assert score.value > 0
    assert score.metadata["source"] == "raw"
    assert score.metadata["uncalibrated_value"] == 0.0


def make_trajectory(
    trajectory_id: str,
    *,
    task_id: str = "task-1",
    answer: str = "yes",
    succeeded: bool = True,
) -> Trajectory:
    now = datetime.now(timezone.utc)
    step = TrajectoryStep(
        index=0,
        input_messages=(Message(role=MessageRole.USER, content="Return yes."),),
        action=AgentAction(kind=ActionKind.FINAL, final_answer=answer),
        generated_token_count=1,
        generated_token_mask=(1,),
    )
    outcome = RewardSummary(
        signals=(
            RewardSignal(
                name="outcome",
                source=RewardSource.OUTCOME,
                value=1.0 if succeeded else 0.0,
                terminal=True,
            ),
        )
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        group_id="group-1",
        policy_version="policy-1",
        environment_version="env-1",
        provenance=Provenance(
            origin=DataOrigin.ON_POLICY,
            producer="test",
            producer_version="1",
        ),
        status=TrajectoryStatus.SUCCEEDED if succeeded else TrajectoryStatus.FAILED,
        steps=(step,),
        final_reward=outcome,
        started_at=now,
        completed_at=now,
    )


def test_rejection_sampling_verifies_deduplicates_and_tracks_lineage() -> None:
    source = make_trajectory("source-1")
    duplicate = make_trajectory("source-2")
    failure = make_trajectory("source-3", succeeded=False)
    other_task = make_trajectory("source-4", task_id="task-2")

    result = VerifiedRejectionSampler(min_reward=0.5, top_k_per_task=1).select(
        (source, duplicate, failure, other_task)
    )

    assert len(result.accepted) == 2
    accepted_by_task = {item.task_id: item for item in result.accepted}
    assert accepted_by_task["task-1"].provenance.origin is DataOrigin.REJECTION_SAMPLING
    assert accepted_by_task["task-1"].provenance.parent_ids == ("source-1",)
    reasons = {item.trajectory_id: item.reason for item in result.rejected}
    assert reasons["source-2"] == "duplicate"
    assert reasons["source-3"] == "not_successful"

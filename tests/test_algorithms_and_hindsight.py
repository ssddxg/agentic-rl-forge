from datetime import datetime, timezone

import pytest

from agentic_rl_forge.algorithms import (
    GroupReward,
    NashMDBatchBuilder,
    PolicySample,
    RulePreferenceModel,
    geometric_mixture_weights,
    group_relative_advantages,
)
from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    DataOrigin,
    Message,
    MessageRole,
    Provenance,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
)
from agentic_rl_forge.data import (
    HindsightTrajectoryRelabeling,
    ToolAchievementRelabeler,
    ToolEvidenceVerifier,
)


def test_grpo_advantages_are_centered_and_handle_zero_variance() -> None:
    advantages = group_relative_advantages(
        (GroupReward("a", 0.0), GroupReward("b", 1.0), GroupReward("c", 2.0))
    )

    assert sum(item.advantage for item in advantages) == pytest.approx(0.0)
    assert advantages[0].advantage < 0 < advantages[-1].advantage
    tied = group_relative_advantages((GroupReward("a", 1.0), GroupReward("b", 1.0)))
    assert {item.advantage for item in tied} == {0.0}


def test_nash_md_geometric_mixture_requires_reference_only_when_used() -> None:
    samples = (
        PolicySample("a", "prompt", "short", current_logprob=-0.2),
        PolicySample("b", "prompt", "longer", current_logprob=-1.2),
    )

    weights = geometric_mixture_weights(samples, beta=0.0)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1]
    with pytest.raises(ValueError, match="reference_logprob"):
        geometric_mixture_weights(samples, beta=0.25)


@pytest.mark.asyncio
async def test_reference_free_nash_md_builds_self_play_preferences() -> None:
    preference = RulePreferenceModel(
        lambda learner, opponent: 0.8 if len(learner.response) > len(opponent.response) else 0.2
    )
    builder = NashMDBatchBuilder(preference, beta=0.0, seed=3)
    samples = (
        PolicySample("a", "prompt", "one", current_logprob=-0.2),
        PolicySample("b", "prompt", "a longer response", current_logprob=-0.5),
        PolicySample("c", "prompt", "middle", current_logprob=-0.4),
    )

    pairs = await builder.build(samples, comparisons_per_prompt=4)

    assert builder.reference_free
    assert len(pairs) == 4
    assert all(pair.learner.sample_id != pair.opponent.sample_id for pair in pairs)
    assert all(abs(pair.advantage) == pytest.approx(0.6) for pair in pairs)


def failed_tool_trajectory() -> Trajectory:
    now = datetime.now(timezone.utc)
    call = ToolCall(call_id="call-1", name="lookup_order", arguments={"order_id": "7"})
    tool_step = TrajectoryStep(
        index=0,
        input_messages=(Message(role=MessageRole.USER, content="Cancel order 7."),),
        action=AgentAction(kind=ActionKind.TOOL, tool_calls=(call,)),
        tool_results=(
            ToolResult(
                call_id="call-1",
                name="lookup_order",
                content='{"status":"paid"}',
                ok=True,
                metadata={
                    "achieved_goal": "Retrieve the current status of order 7.",
                    "goal_confidence": 0.95,
                },
            ),
        ),
    )
    final_step = TrajectoryStep(
        index=1,
        input_messages=(Message(role=MessageRole.USER, content="Cancel order 7."),),
        action=AgentAction(kind=ActionKind.FINAL, final_answer="Unable to cancel."),
    )
    return Trajectory(
        trajectory_id="failed-1",
        task_id="cancel-order",
        group_id="group-1",
        policy_version="policy-1",
        environment_version="orders-v1",
        provenance=Provenance(
            origin=DataOrigin.ON_POLICY,
            producer="test",
            producer_version="1",
        ),
        status=TrajectoryStatus.FAILED,
        steps=(tool_step, final_step),
        started_at=now,
        completed_at=now,
    )


@pytest.mark.asyncio
async def test_hindsight_relabeling_requires_executable_evidence_and_masks_actions() -> None:
    pipeline = HindsightTrajectoryRelabeling(
        ToolAchievementRelabeler(),
        ToolEvidenceVerifier(),
    )

    result = await pipeline.relabel(failed_tool_trajectory())

    assert len(result.accepted) == 1
    example = result.accepted[0]
    assert example.goal.description == "Retrieve the current status of order 7."
    assert example.action_loss_mask == (True, False)
    assert example.sample_weight == pytest.approx(0.95)
    assert example.provenance.origin is DataOrigin.HINDSIGHT_RELABELED
    assert example.provenance.parent_ids == ("failed-1",)
    assert example.verification.evidence["call_id"] == "call-1"

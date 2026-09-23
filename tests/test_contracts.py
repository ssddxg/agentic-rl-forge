from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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
    SideEffect,
    TaskSpec,
    ToolCall,
    ToolResult,
    ToolSpec,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStep,
    VerifierSpec,
)


def make_task() -> TaskSpec:
    return TaskSpec(
        task_id="qa-1",
        messages=(Message(role=MessageRole.USER, content="Find the answer."),),
        tools=(
            ToolSpec(
                name="search",
                description="Search a private corpus",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                side_effect=SideEffect.READ,
            ),
        ),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "42"}),
    )


def test_task_contract_is_stable_and_hashable() -> None:
    task = make_task()
    same_task = make_task()

    assert task.digest() == same_task.digest()
    assert len(task.digest()) == 64


def test_write_tool_must_not_claim_idempotency() -> None:
    with pytest.raises(ValidationError, match="non-idempotent"):
        ToolSpec(
            name="send_email",
            description="Send a message",
            input_schema={"type": "object"},
            side_effect=SideEffect.EXTERNAL,
        )


def test_tool_action_requires_matching_results() -> None:
    call = ToolCall(call_id="call-1", name="search", arguments={"query": "answer"})
    action = AgentAction(kind=ActionKind.TOOL, tool_calls=(call,))

    with pytest.raises(ValidationError, match="exactly match"):
        TrajectoryStep(
            index=0,
            input_messages=make_task().messages,
            action=action,
            tool_results=(
                ToolResult(
                    call_id="different-call",
                    name="search",
                    content="result",
                    ok=True,
                ),
            ),
        )


def test_reward_summary_uses_weights_and_confidence() -> None:
    summary = RewardSummary(
        signals=(
            RewardSignal(name="success", source=RewardSource.OUTCOME, value=1.0),
            RewardSignal(
                name="cost",
                source=RewardSource.COST,
                value=-0.2,
                weight=0.5,
                confidence=0.5,
            ),
        )
    )

    assert summary.total == pytest.approx(0.95)
    assert summary.by_source()[RewardSource.COST] == pytest.approx(-0.05)


def test_derived_provenance_requires_lineage() -> None:
    with pytest.raises(ValidationError, match="parent_id"):
        Provenance(
            origin=DataOrigin.MCTS,
            producer="tree-search",
            producer_version="1",
            transform="mcts-select",
        )


def test_successful_trajectory_enforces_contiguous_steps_and_final_action() -> None:
    now = datetime.now(timezone.utc)
    provenance = Provenance(
        origin=DataOrigin.ON_POLICY,
        producer="rollout-worker",
        producer_version="policy-7",
    )
    final_step = TrajectoryStep(
        index=1,
        input_messages=make_task().messages,
        action=AgentAction(kind=ActionKind.FINAL, final_answer="42"),
    )

    with pytest.raises(ValidationError, match="contiguous"):
        Trajectory(
            task_id="qa-1",
            group_id="group-1",
            policy_version="policy-7",
            environment_version="search-v1",
            provenance=provenance,
            status=TrajectoryStatus.SUCCEEDED,
            steps=(final_step,),
            completed_at=now,
        )


def test_on_policy_guard_rejects_stale_or_derived_data() -> None:
    now = datetime.now(timezone.utc)
    final_step = TrajectoryStep(
        index=0,
        input_messages=make_task().messages,
        action=AgentAction(kind=ActionKind.FINAL, final_answer="42"),
    )
    trajectory = Trajectory(
        task_id="qa-1",
        group_id="group-1",
        policy_version="policy-7",
        environment_version="search-v1",
        provenance=Provenance(
            origin=DataOrigin.ON_POLICY,
            producer="rollout-worker",
            producer_version="policy-7",
        ),
        status=TrajectoryStatus.SUCCEEDED,
        steps=(final_step,),
        completed_at=now,
    )

    trajectory.require_on_policy("policy-7")
    with pytest.raises(ValueError, match="does not match"):
        trajectory.require_on_policy("policy-8")

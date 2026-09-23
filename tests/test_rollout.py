import asyncio

import pytest

from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    Message,
    MessageRole,
    TaskSpec,
    ToolCall,
    ToolSpec,
    Trajectory,
    TrajectoryStatus,
    VerifierSpec,
)
from agentic_rl_forge.environments import InMemorySearchTool, LocalToolEnvironment
from agentic_rl_forge.rewards import (
    CompletionGuardReward,
    CostReward,
    ExactMatchOutcome,
    InvalidActionReward,
    RepeatedActionPenalty,
    RewardEngine,
    ToolExecutionReward,
)
from agentic_rl_forge.rollout import (
    AgentLoop,
    GenerationRequest,
    PolicyOutput,
    RolloutBatch,
    RolloutConfig,
    RolloutScheduler,
    ScriptedPolicy,
)


def search_task(search: InMemorySearchTool) -> TaskSpec:
    return TaskSpec(
        task_id="capital-france",
        messages=(Message(role=MessageRole.USER, content="What is the capital of France?"),),
        tools=(search.spec,),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "Paris"}),
        max_steps=4,
    )


def reward_engine() -> RewardEngine:
    return RewardEngine(
        (
            ExactMatchOutcome(),
            ToolExecutionReward(),
            CostReward(),
            InvalidActionReward(),
            RepeatedActionPenalty(repeat_threshold=2),
            CompletionGuardReward(),
        )
    )


@pytest.mark.asyncio
async def test_search_rollout_records_masks_provenance_and_reward() -> None:
    search = InMemorySearchTool(
        {
            "france": "Paris is the capital and largest city of France.",
            "germany": "Berlin is the capital of Germany.",
        }
    )
    call = ToolCall(call_id="search-1", name="search", arguments={"query": "France capital"})
    policy = ScriptedPolicy(
        (
            PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.TOOL,
                    reasoning="I should verify the answer.",
                    tool_calls=(call,),
                    raw_text="<search>France capital</search>",
                ),
                generated_token_count=5,
                policy_logprobs=(-0.1,) * 5,
                model="test-model",
            ),
            PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.FINAL,
                    reasoning="The retrieved passage directly states the answer.",
                    final_answer="Paris",
                    raw_text="<answer>Paris</answer>",
                ),
                generated_token_count=4,
                policy_logprobs=(-0.05,) * 4,
                model="test-model",
            ),
        ),
        version="policy-1",
    )
    environment = LocalToolEnvironment((search,))
    loop = AgentLoop(policy=policy, environment=environment, rewards=reward_engine())

    trajectory = await loop.run(search_task(search), group_id="group-1", seed=7)

    assert trajectory.status is TrajectoryStatus.SUCCEEDED
    assert trajectory.is_on_policy
    assert trajectory.policy_version == "policy-1"
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].generated_token_mask[:5] == (1,) * 5
    assert set(trajectory.steps[0].generated_token_mask[5:]) == {0}
    assert trajectory.final_reward.signals[0].metadata["matched"] is True
    assert trajectory.total_reward < 1.0
    trajectory.require_on_policy("policy-1")


@pytest.mark.asyncio
async def test_repeated_actions_are_truncated() -> None:
    search = InMemorySearchTool({"france": "Paris is the capital of France."})
    repeated_outputs = []
    for index in range(3):
        repeated_outputs.append(
            PolicyOutput(
                action=AgentAction(
                    kind=ActionKind.TOOL,
                    tool_calls=(
                        ToolCall(
                            call_id=f"search-{index}",
                            name="search",
                            arguments={"query": "France"},
                        ),
                    ),
                ),
                generated_token_count=2,
            )
        )
    loop = AgentLoop(
        policy=ScriptedPolicy(repeated_outputs),
        environment=LocalToolEnvironment((search,)),
        rewards=reward_engine(),
        config=RolloutConfig(repeated_action_limit=3),
    )

    trajectory = await loop.run(search_task(search))

    assert trajectory.status is TrajectoryStatus.TRUNCATED
    assert len(trajectory.steps) == 3
    assert any(signal.name == "repeated_action" for signal in trajectory.steps[-1].rewards.signals)
    assert any(signal.name == "incomplete_trajectory" for signal in trajectory.final_reward.signals)


@pytest.mark.asyncio
async def test_scheduler_cancels_and_joins_sibling_rollouts_after_failure() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class CoordinatedPolicy:
        version = "coordinated-policy"

        async def generate(
            self,
            messages: tuple[Message, ...],
            tools: tuple[ToolSpec, ...],
            request: GenerationRequest,
        ) -> PolicyOutput:
            del tools, request
            question = messages[-1].content
            if question == "fail":
                await started.wait()
                return PolicyOutput(
                    action=AgentAction(kind=ActionKind.FINAL, final_answer="done"),
                    generated_token_count=1,
                )
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("unreachable")

    class FailingCallback:
        async def on_trajectory(self, trajectory: Trajectory) -> None:
            if trajectory.task_id == "fail":
                raise RuntimeError("expected failure")

        async def on_batch(self, batch: RolloutBatch) -> None:
            del batch

    def task(task_id: str) -> TaskSpec:
        return TaskSpec(
            task_id=task_id,
            messages=(Message(role=MessageRole.USER, content=task_id),),
            verifier=VerifierSpec(kind="exact_match", config={"answer": "done"}),
        )

    def loop_factory() -> AgentLoop:
        return AgentLoop(
            policy=CoordinatedPolicy(),
            environment=LocalToolEnvironment(()),
            rewards=RewardEngine((ExactMatchOutcome(),)),
        )

    with pytest.raises(RuntimeError, match="expected failure"):
        await RolloutScheduler(
            loop_factory,
            max_concurrency=2,
            callbacks=(FailingCallback(),),
        ).collect(
            (task("fail"), task("wait")),
            rollouts_per_task=1,
        )

    assert cancelled.is_set()

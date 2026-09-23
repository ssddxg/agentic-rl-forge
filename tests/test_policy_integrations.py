import json

import httpx
import pytest

from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    Message,
    MessageRole,
    TaskSpec,
    ToolSpec,
    VerifierSpec,
)
from agentic_rl_forge.environments import LocalToolEnvironment
from agentic_rl_forge.integrations import (
    search_r1_exact_match_score,
    task_to_verl_record,
    trajectory_to_sft_messages,
    trajectory_to_verl_record,
    validate_grpo_batch,
)
from agentic_rl_forge.rewards import ExactMatchOutcome, RewardEngine
from agentic_rl_forge.rollout import (
    AgentLoop,
    GenerationRequest,
    OpenAICompatiblePolicy,
    PolicyOutput,
    RolloutScheduler,
    ScriptedPolicy,
    SearchR1Parser,
)


def final_task(task_id: str = "simple-1") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        messages=(Message(role=MessageRole.USER, content="Return yes."),),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "yes"}),
    )


def test_search_r1_parser_prefers_first_protocol_event() -> None:
    parser = SearchR1Parser()
    action = parser.parse(
        "<think>I need evidence.</think><search>capital of France</search><answer>not yet</answer>"
    )

    assert action.kind is ActionKind.TOOL
    assert action.tool_calls[0].arguments == {"query": "capital of France"}
    assert action.reasoning == "I need evidence."


@pytest.mark.asyncio
async def test_openai_compatible_policy_parses_native_tool_calls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "I will search.",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query":"France capital"}',
                                    },
                                }
                            ],
                        },
                        "logprobs": {"content": [{"logprob": -0.2}, {"logprob": -0.1}]},
                    }
                ],
                "usage": {"completion_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    policy = OpenAICompatiblePolicy(
        base_url="http://model.local",
        model="model",
        version="policy-2",
        client=client,
    )
    tool = ToolSpec(
        name="search",
        description="Search documents.",
        input_schema={"type": "object"},
    )

    output = await policy.generate(
        (Message(role=MessageRole.USER, content="Question"),),
        (tool,),
        GenerationRequest(seed=1),
    )

    assert output.action.kind is ActionKind.TOOL
    assert output.action.tool_calls[0].name == "search"
    assert output.policy_logprobs == (-0.2, -0.1)
    assert output.model == "served-model"
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_policy_serializes_native_tool_history_and_handles_sparse_logprobs() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][1]["role"] == "assistant"
        assert json.loads(payload["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {
            "query": "France capital"
        }
        assert payload["messages"][2]["role"] == "tool"
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "yes"},
                        "logprobs": {"content": [{"logprob": -0.2}, {"logprob": -0.1}]},
                    }
                ],
                "usage": {"completion_tokens": 3},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    policy = OpenAICompatiblePolicy(
        base_url="http://model.local",
        model="model",
        version="policy-2",
        client=client,
    )
    output = await policy.generate(
        (
            Message(role=MessageRole.USER, content="Question"),
            Message(
                role=MessageRole.ASSISTANT,
                content="I will search.",
                metadata={
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "search",
                            "arguments": {"query": "France capital"},
                        }
                    ]
                },
            ),
            Message(
                role=MessageRole.TOOL,
                content="Paris is the capital of France.",
                tool_call_id="call-1",
                name="search",
            ),
        ),
        (),
        GenerationRequest(),
    )

    assert output.action.kind is ActionKind.FINAL
    assert output.generated_token_count == 3
    assert output.policy_logprobs == ()
    await client.aclose()


@pytest.mark.asyncio
async def test_grouped_scheduler_and_verl_export_preserve_policy_version() -> None:
    environment = LocalToolEnvironment(())
    rewards = RewardEngine((ExactMatchOutcome(),))

    def loop_factory() -> AgentLoop:
        policy = ScriptedPolicy(
            (
                PolicyOutput(
                    action=AgentAction(
                        kind=ActionKind.FINAL,
                        final_answer="yes",
                        raw_text="<answer>yes</answer>",
                    ),
                    generated_token_count=3,
                    policy_logprobs=(-0.1, -0.1, -0.1),
                ),
            ),
            version="policy-3",
        )
        return AgentLoop(policy=policy, environment=environment, rewards=rewards)

    batch = await RolloutScheduler(loop_factory, max_concurrency=4).collect(
        (final_task("task-1"), final_task("task-2")),
        rollouts_per_task=2,
        seed=10,
    )

    assert len(batch.trajectories) == 4
    assert len(batch.grouped()) == 2
    validate_grpo_batch(
        batch.trajectories,
        expected_policy_version="policy-3",
        expected_group_size=2,
    )
    record = trajectory_to_verl_record(batch.trajectories[0])
    assert record.response_mask == (1, 1, 1)
    assert task_to_verl_record(final_task()).reward_model["ground_truth"] == "yes"
    messages = trajectory_to_sft_messages(batch.trajectories[0])
    assert messages[-1]["content"] == "<answer>yes</answer>"


def test_search_r1_reward_extracts_last_answer() -> None:
    assert search_r1_exact_match_score(
        "<answer>wrong</answer><answer>The Paris.</answer>", "Paris"
    ) == pytest.approx(1.0)
    assert search_r1_exact_match_score("No answer tag", "Paris") == 0.0

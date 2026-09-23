import httpx
import pytest

from agentic_rl_forge.contracts import (
    ActionKind,
    AgentAction,
    Message,
    MessageRole,
    TaskSpec,
    VerifierSpec,
)
from agentic_rl_forge.environments import (
    HTTPRetrievalTool,
    LocalToolEnvironment,
    ToolContext,
)
from agentic_rl_forge.rollout import PolicyOutput, ScriptedPolicy
from agentic_rl_forge.search import (
    AdaptiveComputeBudget,
    AgentSearchDomain,
    AgentTreeSearchController,
    BudgetSignals,
    HeuristicProcessRewardModel,
    PolicyCandidateGenerator,
    ProcessScore,
)


@pytest.mark.asyncio
async def test_agent_tree_search_prunes_incorrect_terminal_action() -> None:
    task = TaskSpec(
        task_id="answer",
        messages=(Message(role=MessageRole.USER, content="Return Paris."),),
        verifier=VerifierSpec(kind="exact_match", config={"answer": "Paris"}),
    )
    policy = ScriptedPolicy(
        (
            PolicyOutput(
                action=AgentAction(kind=ActionKind.FINAL, final_answer="London"),
                generated_token_count=1,
            ),
            PolicyOutput(
                action=AgentAction(kind=ActionKind.FINAL, final_answer="Paris"),
                generated_token_count=1,
            ),
        ),
        version="tree-policy",
    )
    prm = HeuristicProcessRewardModel(lambda item: ProcessScore(value=0.0))
    environment = LocalToolEnvironment(())
    domain = AgentSearchDomain(
        environment=environment,
        candidate_generator=PolicyCandidateGenerator(policy),
        process_reward_model=prm,
    )
    root = await domain.start(task)
    controller = AgentTreeSearchController(
        domain,
        AdaptiveComputeBudget(
            min_simulations=4,
            max_simulations=4,
            min_branch_factor=2,
            max_branch_factor=2,
            max_depth=2,
        ),
    )

    result = await controller.choose(root, BudgetSignals(), seed=1)

    assert result.action.kind is ActionKind.FINAL
    assert result.action.final_answer == "Paris"
    assert len(result.root_actions) == 1
    assert result.root_actions[0].process_value == 1.0
    await domain.close(root)


@pytest.mark.asyncio
async def test_http_retrieval_tool_uses_search_r1_protocol() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload == {"queries": ["France"], "topk": 2, "return_scores": True}
        return httpx.Response(
            200,
            json={
                "result": [[{"document": {"id": "1", "contents": "Paris, France"}, "score": 0.9}]]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = HTTPRetrievalTool("http://retriever.local/retrieve", top_k=2, client=client)

    result = await tool.execute(
        {"query": "France"},
        ToolContext(session_id="s", task_id="t", state={}),
    )

    assert "Paris" in result.content
    assert result.metadata["result_count"] == 1
    await client.aclose()

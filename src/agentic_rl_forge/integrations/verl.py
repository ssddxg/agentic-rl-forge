from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from agentic_rl_forge.contracts import (
    ActionKind,
    ContractModel,
    DataOrigin,
    TaskSpec,
    Trajectory,
)


class VerlTaskRecord(ContractModel):
    data_source: str
    prompt: tuple[dict[str, Any], ...]
    ability: str
    reward_model: dict[str, Any]
    extra_info: dict[str, Any] = Field(default_factory=dict)


class VerlTrajectoryRecord(ContractModel):
    trajectory_id: str
    task_id: str
    group_id: str
    policy_version: str
    environment_version: str
    origin: DataOrigin
    status: str
    response_mask: tuple[int, ...]
    policy_logprobs: tuple[float, ...]
    reward: float
    trajectory: dict[str, Any]


def task_to_verl_record(
    task: TaskSpec,
    *,
    data_source: str = "agentic_rl_forge",
    ability: str = "tool_reasoning",
) -> VerlTaskRecord:
    ground_truth = task.verifier.config.get("answer")
    return VerlTaskRecord(
        data_source=data_source,
        prompt=tuple(
            message.model_dump(mode="json", exclude_none=True) for message in task.messages
        ),
        ability=ability,
        reward_model={
            "style": "rule",
            "ground_truth": ground_truth,
            "verifier": task.verifier.kind,
        },
        extra_info={
            "task_id": task.task_id,
            "max_steps": task.max_steps,
            "max_generated_tokens": task.max_generated_tokens,
            "tools": [tool.model_dump(mode="json", exclude_none=True) for tool in task.tools],
            **task.metadata,
        },
    )


def trajectory_to_verl_record(trajectory: Trajectory) -> VerlTrajectoryRecord:
    response_mask = tuple(token for step in trajectory.steps for token in step.generated_token_mask)
    policy_logprobs = tuple(
        logprob for step in trajectory.steps for logprob in step.policy_logprobs
    )
    return VerlTrajectoryRecord(
        trajectory_id=trajectory.trajectory_id,
        task_id=trajectory.task_id,
        group_id=trajectory.group_id,
        policy_version=trajectory.policy_version,
        environment_version=trajectory.environment_version,
        origin=trajectory.provenance.origin,
        status=trajectory.status.value,
        response_mask=response_mask,
        policy_logprobs=policy_logprobs,
        reward=trajectory.total_reward,
        trajectory=trajectory.model_dump(mode="json", exclude_none=True),
    )


def validate_grpo_batch(
    trajectories: Sequence[Trajectory],
    *,
    expected_policy_version: str,
    expected_group_size: int,
) -> None:
    if expected_group_size < 2:
        raise ValueError("GRPO groups require at least two trajectories")
    groups: dict[str, list[Trajectory]] = {}
    for trajectory in trajectories:
        trajectory.require_on_policy(expected_policy_version)
        groups.setdefault(trajectory.group_id, []).append(trajectory)
    if not groups:
        raise ValueError("GRPO batch cannot be empty")
    for group_id, group in groups.items():
        if len(group) != expected_group_size:
            raise ValueError(
                f"group {group_id!r} has {len(group)} trajectories; expected {expected_group_size}"
            )
        if len({trajectory.task_id for trajectory in group}) != 1:
            raise ValueError(f"group {group_id!r} contains multiple tasks")
        masks = [record.response_mask for record in map(trajectory_to_verl_record, group)]
        if any(not mask for mask in masks):
            raise ValueError(f"group {group_id!r} contains an empty response mask")


def export_jsonl(records: Iterable[ContractModel], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as output:
        for record in records:
            output.write(record.canonical_bytes())
            output.write(b"\n")
            count += 1
    return count


def search_r1_exact_match_score(solution: str, ground_truth: str | list[str]) -> float:
    answer_matches = re.findall(r"<answer>(.*?)</answer>", solution, flags=re.DOTALL | re.I)
    if not answer_matches:
        return 0.0
    predicted = _normalize_answer(answer_matches[-1])
    expected = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    return float(predicted in {_normalize_answer(answer) for answer in expected})


def trajectory_to_sft_messages(trajectory: Trajectory) -> list[dict[str, Any]]:
    if not trajectory.steps:
        raise ValueError("cannot export an empty trajectory for SFT")
    messages = [
        message.model_dump(mode="json", exclude_none=True)
        for message in trajectory.steps[0].input_messages
    ]
    for step in trajectory.steps:
        action = step.action
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": action.raw_text or action.reasoning or action.final_answer or "",
        }
        if action.kind is ActionKind.TOOL:
            assistant["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in action.tool_calls
            ]
        messages.append(assistant)
        messages.extend(
            {
                "role": "tool",
                "name": result.name,
                "tool_call_id": result.call_id,
                "content": result.content,
            }
            for result in step.tool_results
        )
    return messages


def _normalize_answer(value: str) -> str:
    lowered = value.casefold().strip()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return " ".join(lowered.split())

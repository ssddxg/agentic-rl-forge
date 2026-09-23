from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import orjson

from agentic_rl_forge.contracts import (
    DataOrigin,
    JsonObject,
    Provenance,
    Trajectory,
    TrajectoryStatus,
    new_id,
)


@dataclass(frozen=True, slots=True)
class AchievedGoal:
    description: str
    verifier_kind: str
    verifier_config: JsonObject
    confidence: float
    supporting_step_indices: tuple[int, ...]
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("achieved goals require a description")
        if not 0 <= self.confidence <= 1:
            raise ValueError("goal confidence must be between zero and one")
        if not self.supporting_step_indices:
            raise ValueError("achieved goals require supporting steps")


@dataclass(frozen=True, slots=True)
class GoalVerification:
    accepted: bool
    confidence: float
    evidence: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("verification confidence must be between zero and one")


class GoalRelabeler(Protocol):
    @property
    def version(self) -> str: ...

    async def propose(self, trajectory: Trajectory) -> Sequence[AchievedGoal]: ...


class GoalVerifier(Protocol):
    @property
    def version(self) -> str: ...

    async def verify(self, trajectory: Trajectory, goal: AchievedGoal) -> GoalVerification: ...


@dataclass(frozen=True, slots=True)
class HindsightExample:
    example_id: str
    source_trajectory_id: str
    synthetic_task_id: str
    goal: AchievedGoal
    action_loss_mask: tuple[bool, ...]
    sample_weight: float
    verification: GoalVerification
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class HindsightResult:
    accepted: tuple[HindsightExample, ...]
    rejected: tuple[tuple[str, str], ...]


class ToolAchievementRelabeler:
    def __init__(self, *, version: str = "tool-achievement-v1") -> None:
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    async def propose(self, trajectory: Trajectory) -> Sequence[AchievedGoal]:
        goals = []
        for step in trajectory.steps:
            for result in step.tool_results:
                if not result.ok:
                    continue
                description = str(
                    result.metadata.get("achieved_goal")
                    or f"Successfully execute the {result.name} tool and obtain a valid result."
                )
                goals.append(
                    AchievedGoal(
                        description=description,
                        verifier_kind="tool_evidence",
                        verifier_config={
                            "tool_name": result.name,
                            "call_id": result.call_id,
                            "result_digest": hashlib.sha256(
                                result.content.encode("utf-8")
                            ).hexdigest(),
                        },
                        confidence=float(result.metadata.get("goal_confidence", 0.8)),
                        supporting_step_indices=(step.index,),
                    )
                )
        return goals


class ToolEvidenceVerifier:
    @property
    def version(self) -> str:
        return "tool-evidence-v1"

    async def verify(self, trajectory: Trajectory, goal: AchievedGoal) -> GoalVerification:
        if goal.verifier_kind != "tool_evidence":
            return GoalVerification(False, 0.0, {"reason": "unsupported_verifier"})
        expected_name = goal.verifier_config.get("tool_name")
        expected_call = goal.verifier_config.get("call_id")
        expected_digest = goal.verifier_config.get("result_digest")
        content_pattern = goal.verifier_config.get("content_regex")
        for step_index in goal.supporting_step_indices:
            if step_index < 0 or step_index >= len(trajectory.steps):
                continue
            for result in trajectory.steps[step_index].tool_results:
                if not result.ok or result.name != expected_name:
                    continue
                if expected_call is not None and result.call_id != expected_call:
                    continue
                digest = hashlib.sha256(result.content.encode("utf-8")).hexdigest()
                if expected_digest is not None and digest != expected_digest:
                    continue
                if (
                    content_pattern is not None
                    and re.search(str(content_pattern), result.content) is None
                ):
                    continue
                return GoalVerification(
                    True,
                    1.0,
                    {
                        "step_index": step_index,
                        "call_id": result.call_id,
                        "result_digest": digest,
                    },
                )
        return GoalVerification(False, 1.0, {"reason": "evidence_not_found"})


class HindsightTrajectoryRelabeling:
    def __init__(
        self,
        relabeler: GoalRelabeler,
        verifier: GoalVerifier,
        *,
        min_goal_confidence: float = 0.7,
        min_verifier_confidence: float = 0.9,
        max_goals_per_trajectory: int = 4,
    ) -> None:
        if not 0 <= min_goal_confidence <= 1:
            raise ValueError("min_goal_confidence must be between zero and one")
        if not 0 <= min_verifier_confidence <= 1:
            raise ValueError("min_verifier_confidence must be between zero and one")
        if max_goals_per_trajectory < 1:
            raise ValueError("max_goals_per_trajectory must be positive")
        self._relabeler = relabeler
        self._verifier = verifier
        self._min_goal_confidence = min_goal_confidence
        self._min_verifier_confidence = min_verifier_confidence
        self._max_goals = max_goals_per_trajectory

    async def relabel(self, trajectory: Trajectory) -> HindsightResult:
        if trajectory.status is TrajectoryStatus.SUCCEEDED:
            return HindsightResult((), ((trajectory.trajectory_id, "already_successful"),))
        proposed = await self._relabeler.propose(trajectory)
        accepted: list[HindsightExample] = []
        rejected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for goal in proposed:
            if len(accepted) >= self._max_goals:
                rejected.append((goal.description, "goal_limit"))
                continue
            goal_digest = hashlib.sha256(
                orjson.dumps(
                    {"description": goal.description, "config": goal.verifier_config},
                    option=orjson.OPT_SORT_KEYS,
                )
            ).hexdigest()
            if goal_digest in seen:
                rejected.append((goal.description, "duplicate_goal"))
                continue
            seen.add(goal_digest)
            if goal.confidence < self._min_goal_confidence:
                rejected.append((goal.description, "low_goal_confidence"))
                continue
            verification = await self._verifier.verify(trajectory, goal)
            if not verification.accepted:
                rejected.append((goal.description, "verification_failed"))
                continue
            if verification.confidence < self._min_verifier_confidence:
                rejected.append((goal.description, "low_verifier_confidence"))
                continue
            valid_indices = {
                index
                for index in goal.supporting_step_indices
                if 0 <= index < len(trajectory.steps)
            }
            if not valid_indices:
                rejected.append((goal.description, "invalid_support"))
                continue
            action_loss_mask = tuple(
                index in valid_indices for index in range(len(trajectory.steps))
            )
            synthetic_task_id = f"htr_{trajectory.task_id}_{goal_digest[:12]}"
            accepted.append(
                HindsightExample(
                    example_id=new_id("htr"),
                    source_trajectory_id=trajectory.trajectory_id,
                    synthetic_task_id=synthetic_task_id,
                    goal=goal,
                    action_loss_mask=action_loss_mask,
                    sample_weight=goal.confidence * verification.confidence,
                    verification=verification,
                    provenance=Provenance(
                        origin=DataOrigin.HINDSIGHT_RELABELED,
                        producer="hindsight-trajectory-relabeling",
                        producer_version="1",
                        parent_ids=(trajectory.trajectory_id,),
                        transform="verified_achieved_goal_relabeling",
                        metadata={
                            "relabeler_version": self._relabeler.version,
                            "verifier_version": self._verifier.version,
                            "goal_digest": goal_digest,
                        },
                    ),
                )
            )
        return HindsightResult(tuple(accepted), tuple(rejected))

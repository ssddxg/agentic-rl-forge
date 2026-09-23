from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from agentic_rl_forge.contracts import JsonObject
from agentic_rl_forge.search.prm import ProcessScore

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class ActionCandidate(Generic[ActionT]):
    action: ActionT
    policy_prior: float = 1.0
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchTransition(Generic[StateT]):
    state: StateT
    reward: float = 0.0
    terminal: bool = False
    metadata: JsonObject = field(default_factory=dict)


class SearchDomain(Protocol[StateT, ActionT]):
    async def propose(
        self,
        state: StateT,
        *,
        branch_factor: int,
        seed: int,
    ) -> Sequence[ActionCandidate[ActionT]]: ...

    async def transition(self, state: StateT, action: ActionT) -> SearchTransition[StateT]: ...

    async def evaluate(self, states: Sequence[StateT]) -> tuple[ProcessScore, ...]: ...


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 32
    branch_factor: int = 4
    max_depth: int = 8
    exploration: float = 1.5
    discount: float = 1.0
    prm_prior_weight: float = 1.0
    prune_below: float = -1.0

    def __post_init__(self) -> None:
        if self.simulations < 1 or self.branch_factor < 1 or self.max_depth < 1:
            raise ValueError("simulations, branch_factor, and max_depth must be positive")
        if self.exploration < 0 or not 0 < self.discount <= 1:
            raise ValueError("exploration or discount is outside the supported range")
        if not -1 <= self.prune_below <= 1:
            raise ValueError("prune_below must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class RootActionStats(Generic[ActionT]):
    action: ActionT
    visits: int
    mean_value: float
    prior: float
    process_value: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class MCTSResult(Generic[ActionT]):
    action: ActionT
    root_actions: tuple[RootActionStats[ActionT], ...]
    simulations: int
    expanded_nodes: int


@dataclass(slots=True)
class _Node(Generic[StateT, ActionT]):
    state: StateT
    parent: _Node[StateT, ActionT] | None = None
    action: ActionT | None = None
    prior: float = 1.0
    edge_reward: float = 0.0
    terminal: bool = False
    depth: int = 0
    process_score: ProcessScore = field(
        default_factory=lambda: ProcessScore(value=0.0, confidence=0.0, uncertainty=1.0)
    )
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: list[_Node[StateT, ActionT]] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class AsyncMCTS(Generic[StateT, ActionT]):
    def __init__(self, domain: SearchDomain[StateT, ActionT], config: MCTSConfig) -> None:
        self._domain = domain
        self._config = config

    async def search(self, root_state: StateT, *, seed: int = 0) -> MCTSResult[ActionT]:
        root_score = (await self._domain.evaluate((root_state,)))[0]
        root = _Node[StateT, ActionT](state=root_state, process_score=root_score)
        expanded_nodes = 0
        random_source = random.Random(seed)
        for simulation in range(self._config.simulations):
            node = root
            path = [root]
            while node.expanded and node.children and not node.terminal:
                node = self._select(node, random_source)
                path.append(node)
            if not node.terminal and node.depth < self._config.max_depth and not node.expanded:
                children = await self._expand(node, seed=seed + simulation)
                expanded_nodes += len(children)
                if children:
                    node = max(children, key=lambda child: child.prior)
                    path.append(node)
            leaf_value = 0.0 if node.terminal else node.process_score.calibrated_value
            self._backup(path, leaf_value)
        if not root.children:
            raise RuntimeError("MCTS root has no viable actions")
        best = max(root.children, key=lambda child: (child.visits, child.mean_value))
        if best.action is None:
            raise RuntimeError("selected MCTS node has no action")
        stats = tuple(
            RootActionStats(
                action=child.action,
                visits=child.visits,
                mean_value=child.mean_value,
                prior=child.prior,
                process_value=child.process_score.value,
                uncertainty=child.process_score.uncertainty,
            )
            for child in sorted(
                root.children,
                key=lambda item: (item.visits, item.mean_value),
                reverse=True,
            )
            if child.action is not None
        )
        return MCTSResult(
            action=best.action,
            root_actions=stats,
            simulations=self._config.simulations,
            expanded_nodes=expanded_nodes,
        )

    def _select(
        self,
        node: _Node[StateT, ActionT],
        random_source: random.Random,
    ) -> _Node[StateT, ActionT]:
        parent_scale = math.sqrt(node.visits + 1)
        scored = []
        for child in node.children:
            exploration = self._config.exploration * child.prior * parent_scale / (1 + child.visits)
            scored.append((child.mean_value + exploration, random_source.random(), child))
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    async def _expand(
        self,
        node: _Node[StateT, ActionT],
        *,
        seed: int,
    ) -> list[_Node[StateT, ActionT]]:
        candidates = list(
            await self._domain.propose(
                node.state,
                branch_factor=self._config.branch_factor,
                seed=seed,
            )
        )
        candidates = candidates[: self._config.branch_factor]
        if not candidates:
            node.expanded = True
            return []
        transitions = []
        for candidate in candidates:
            transitions.append(await self._domain.transition(node.state, candidate.action))
        scores = await self._domain.evaluate(tuple(item.state for item in transitions))
        if len(scores) != len(transitions):
            raise ValueError("search domain returned an invalid number of process scores")
        viable = [
            (candidate, transition, score)
            for candidate, transition, score in zip(candidates, transitions, scores, strict=True)
            if score.value >= self._config.prune_below
        ]
        if not viable:
            viable = [
                max(
                    zip(candidates, transitions, scores, strict=True),
                    key=lambda item: item[2].value,
                )
            ]
        raw_priors = [
            max(candidate.policy_prior, 1e-8)
            * math.exp(self._config.prm_prior_weight * score.calibrated_value)
            for candidate, _, score in viable
        ]
        normalizer = sum(raw_priors)
        children = []
        for (candidate, transition, score), raw_prior in zip(viable, raw_priors, strict=True):
            children.append(
                _Node(
                    state=transition.state,
                    parent=node,
                    action=candidate.action,
                    prior=raw_prior / normalizer,
                    edge_reward=transition.reward,
                    terminal=transition.terminal,
                    depth=node.depth + 1,
                    process_score=score,
                )
            )
        node.children.extend(children)
        node.expanded = True
        return children

    def _backup(self, path: list[_Node[StateT, ActionT]], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            if node.parent is not None:
                value = node.edge_reward + self._config.discount * value
            node.visits += 1
            node.value_sum += value

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from agentic_rl_forge.contracts import JsonObject


@dataclass(frozen=True, slots=True)
class PolicySample:
    sample_id: str
    prompt_id: str
    response: str
    current_logprob: float
    reference_logprob: float | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NashMDPair:
    prompt_id: str
    learner: PolicySample
    opponent: PolicySample
    preference_probability: float
    advantage: float
    beta: float


class PairwisePreferenceModel(Protocol):
    @property
    def version(self) -> str: ...

    async def compare(self, learner: PolicySample, opponent: PolicySample) -> float: ...


class RulePreferenceModel:
    def __init__(
        self,
        comparator: Callable[[PolicySample, PolicySample], float],
        *,
        version: str = "rule-preference-v1",
    ) -> None:
        self._comparator = comparator
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    async def compare(self, learner: PolicySample, opponent: PolicySample) -> float:
        probability = self._comparator(learner, opponent)
        if not 0 <= probability <= 1:
            raise ValueError("preference probability must be between zero and one")
        return probability


def geometric_mixture_weights(
    samples: Sequence[PolicySample],
    *,
    beta: float,
) -> tuple[float, ...]:
    if not samples:
        raise ValueError("mixture requires at least one sample")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between zero and one")
    logits = []
    for sample in samples:
        if beta > 0 and sample.reference_logprob is None:
            raise ValueError("reference_logprob is required when beta is greater than zero")
        reference_logprob = sample.reference_logprob or 0.0
        logits.append((1.0 - beta) * sample.current_logprob + beta * reference_logprob)
    maximum = max(logits)
    unnormalized = [math.exp(logit - maximum) for logit in logits]
    normalizer = sum(unnormalized)
    return tuple(weight / normalizer for weight in unnormalized)


class NashMDBatchBuilder:
    def __init__(
        self,
        preference_model: PairwisePreferenceModel,
        *,
        beta: float = 0.25,
        seed: int = 0,
    ) -> None:
        if not 0 <= beta <= 1:
            raise ValueError("beta must be between zero and one")
        self._preference_model = preference_model
        self._beta = beta
        self._random = random.Random(seed)

    @property
    def reference_free(self) -> bool:
        return self._beta == 0

    async def build(
        self,
        samples: Sequence[PolicySample],
        *,
        comparisons_per_prompt: int = 1,
    ) -> tuple[NashMDPair, ...]:
        if comparisons_per_prompt < 1:
            raise ValueError("comparisons_per_prompt must be positive")
        groups: dict[str, list[PolicySample]] = {}
        for sample in samples:
            groups.setdefault(sample.prompt_id, []).append(sample)
        pairs: list[NashMDPair] = []
        for prompt_id, group in groups.items():
            if len(group) < 2:
                raise ValueError(f"prompt {prompt_id!r} requires at least two policy samples")
            mixture_weights = geometric_mixture_weights(group, beta=self._beta)
            for _ in range(comparisons_per_prompt):
                learner_index = self._random.randrange(len(group))
                opponent_index = self._sample_opponent(mixture_weights, learner_index)
                learner = group[learner_index]
                opponent = group[opponent_index]
                preference = await self._preference_model.compare(learner, opponent)
                if not 0 <= preference <= 1:
                    raise ValueError("preference model returned a probability outside [0, 1]")
                pairs.append(
                    NashMDPair(
                        prompt_id=prompt_id,
                        learner=learner,
                        opponent=opponent,
                        preference_probability=preference,
                        advantage=2.0 * preference - 1.0,
                        beta=self._beta,
                    )
                )
        return tuple(pairs)

    def _sample_opponent(self, weights: tuple[float, ...], learner_index: int) -> int:
        available = [index for index in range(len(weights)) if index != learner_index]
        available_weights = [weights[index] for index in available]
        normalizer = sum(available_weights)
        normalized = [weight / normalizer for weight in available_weights]
        return self._random.choices(available, weights=normalized, k=1)[0]

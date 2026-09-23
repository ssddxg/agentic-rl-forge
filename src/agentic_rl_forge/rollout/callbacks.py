from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from agentic_rl_forge.contracts import RunLeaseToken, Trajectory, TrajectoryShardManifest
from agentic_rl_forge.storage.shards import ShardedTrajectoryStore
from agentic_rl_forge.storage.sqlite import SQLiteTrajectoryStore

if TYPE_CHECKING:
    from agentic_rl_forge.rollout.scheduler import RolloutBatch


class RolloutCallback(Protocol):
    async def on_trajectory(self, trajectory: Trajectory) -> None: ...

    async def on_batch(self, batch: RolloutBatch) -> None: ...


class RolloutMetricsSink(Protocol):
    def record_trajectory(self, trajectory: Trajectory) -> None: ...

    def increment(
        self,
        name: str,
        amount: float = 1.0,
        *,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None: ...

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None: ...


class CompositeRolloutCallback:
    def __init__(self, callbacks: Sequence[RolloutCallback]) -> None:
        self._callbacks = tuple(callbacks)

    async def on_trajectory(self, trajectory: Trajectory) -> None:
        for callback in self._callbacks:
            await callback.on_trajectory(trajectory)

    async def on_batch(self, batch: RolloutBatch) -> None:
        for callback in self._callbacks:
            await callback.on_batch(batch)


class SQLiteRolloutCallback:
    def __init__(
        self,
        store: SQLiteTrajectoryStore,
        *,
        run_id: str | None = None,
        lease: RunLeaseToken | None = None,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._lease = lease

    async def on_trajectory(self, trajectory: Trajectory) -> None:
        await asyncio.to_thread(
            self._store.put,
            trajectory,
            run_id=self._run_id,
            lease=self._lease,
        )

    async def on_batch(self, batch: RolloutBatch) -> None:
        del batch


class ShardedRolloutCallback:
    def __init__(
        self,
        store: ShardedTrajectoryStore,
        *,
        expected_policy_version: str | None = None,
        finalize_batch: bool = True,
    ) -> None:
        self._store = store
        self._expected_policy_version = expected_policy_version
        self._finalize_batch = finalize_batch
        self._manifest: TrajectoryShardManifest | None = None

    @property
    def manifest(self) -> TrajectoryShardManifest | None:
        return self._manifest

    async def on_trajectory(self, trajectory: Trajectory) -> None:
        await asyncio.to_thread(self._store.put, trajectory)

    async def on_batch(self, batch: RolloutBatch) -> None:
        if not self._finalize_batch:
            return
        expected = self._expected_policy_version or batch.policy_version
        self._manifest = await asyncio.to_thread(
            self._store.finalize,
            expected_policy_version=expected,
            metadata={"callback": "rollout-scheduler"},
        )


class MetricsRolloutCallback:
    def __init__(self, registry: RolloutMetricsSink) -> None:
        self._registry = registry

    async def on_trajectory(self, trajectory: Trajectory) -> None:
        self._registry.record_trajectory(trajectory)

    async def on_batch(self, batch: RolloutBatch) -> None:
        self._registry.increment(
            "arf_rollout_batches_total",
            labels={"policy_version": batch.policy_version},
            help_text="Completed rollout batches.",
        )
        self._registry.set_gauge(
            "arf_rollout_batch_trajectories",
            float(len(batch.trajectories)),
            labels={"policy_version": batch.policy_version},
            help_text="Trajectories in the latest completed rollout batch.",
        )

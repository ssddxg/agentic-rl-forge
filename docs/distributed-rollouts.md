# Distributed rollout capture

Long-running rollout jobs should persist each completed trajectory before the whole batch finishes.
A single append-only JSONL file is a poor coordination point for multiple workers: writes can
interleave, one partial record can invalidate the tail, and a terminated batch loses in-memory
results.

`ShardedTrajectoryStore` uses one immutable file per trajectory and a separate immutable manifest
per finalized view of a run.

## Layout

```text
ROOT/
└── runs/
    └── RUN_ID/
        ├── shards/
        │   ├── SHA256_TRAJECTORY_ID.json
        │   └── ...
        └── manifests/
            ├── manifest_CONTENT_DIGEST.json
            └── ...
```

Filenames are derived from hashes rather than raw trajectory IDs, so external IDs cannot create
paths outside the run directory. Every shard still contains and validates the original trajectory
ID.

## Atomicity and conflicts

A writer serializes the complete trajectory into a temporary file, flushes it, and creates the
final path with an exclusive hard link. Two workers writing identical content under the same ID
converge idempotently. Different content under the same ID is rejected.

This implementation targets a POSIX-compatible local or shared filesystem that supports hard
links. It does not claim atomic behavior on an object-store mount. A native S3, GCS, or OSS adapter
must use the provider's conditional-create primitive and preserve the same conflict semantics.

For native S3-compatible conditional writes and trainer batch manifests, see
[`object-storage.md`](object-storage.md).

Atomic trajectory files reject conflicting commits, but they do not prevent two hosts from paying
for the same model call before either file exists. `SlotClaimCoordinator` adds an optional
append-only claim before planned inference and immutable renewal heartbeats during long attempts.
See [`slot-claims.md`](slot-claims.md) for its storage consistency contract and failure model.

## Scheduler callbacks

Callbacks run after each trajectory completes and after the validated batch is assembled:

```python
from agentic_rl_forge.contracts import RunKind, RunManifest
from agentic_rl_forge.rollout import (
    MetricsRolloutCallback,
    RolloutScheduler,
    ShardedRolloutCallback,
    SQLiteRolloutCallback,
)
from agentic_rl_forge.services import MetricsRegistry
from agentic_rl_forge.storage import SQLiteTrajectoryStore, ShardedTrajectoryStore

sqlite_store = SQLiteTrajectoryStore("artifacts/trajectories.db")
run = RunManifest(name="distributed-rollout", kind=RunKind.ROLLOUT)
sqlite_store.create_run(run)
shard_store = ShardedTrajectoryStore("artifacts/sharded-runs", run_id=run.run_id)
metrics = MetricsRegistry()
lease = sqlite_store.acquire_run_lease(run.run_id, owner_id="worker-42", ttl_s=60)

scheduler = RolloutScheduler(
    loop_factory,
    max_concurrency=64,
    callbacks=(
        SQLiteRolloutCallback(sqlite_store, run_id=run.run_id, lease=lease.token),
        ShardedRolloutCallback(shard_store, expected_policy_version="policy-42"),
        MetricsRolloutCallback(metrics),
    ),
)
```

Applications that acquire a lease must renew it before expiry and use the same token for the
terminal run transition. The built-in Search-R1 collection pipeline manages this heartbeat loop.
The epoch is a fencing token. Ordering the SQLite callback first, as above and in the built-in
pipeline, ensures callbacks from a previous owner fail before later callbacks commit shards or
metrics for that trajectory.

An expired lease does not automatically fail a run. Operators first inspect `run-liveness`, then use
the preview-first `run-reconcile` workflow when the worker is confirmed dead. A recovered worker or
new lease epoch invalidates the preview before any terminal transition.

Run leases and slot claims solve different scopes. A run lease fences writes to one run manifest;
a slot claim suppresses simultaneous work across separate runs that share one rollout plan.

Per-trajectory callback failures fail collection instead of silently dropping data. Batch callbacks
run only after on-policy and group-size validation succeeds. The sharded callback then finalizes a
manifest using the batch policy version.

## Recovery workflow

Import existing JSONL into shards:

```bash
arf shard-import artifacts/trajectories.jsonl artifacts/sharded-runs --run-id run-42
```

After an interrupted collection, scan the surviving shards and finalize them:

```bash
arf shard-finalize artifacts/sharded-runs --run-id run-42 \
  --expected-policy-version policy-42
```

Verify or compact a specific manifest:

```bash
arf shard-verify artifacts/sharded-runs --run-id run-42 \
  --manifest-id MANIFEST_ID
arf shard-export artifacts/sharded-runs artifacts/run-42.jsonl \
  --run-id run-42 --manifest-id MANIFEST_ID
```

Verification reports two related states:

- `valid`: every referenced shard exists and matches its recorded byte size, SHA-256, and
  trajectory ID;
- `complete`: the manifest is valid and the run has no extra shard files that are absent from that
  manifest.

An older manifest can therefore remain valid while becoming incomplete after more workers finish.
Finalizing again produces another content-addressed manifest without mutating the earlier view.

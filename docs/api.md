# Python API guide

AgenticRLForge keeps its supported Python surface in package-level exports. Import from the
modules below instead of reaching into implementation files. Contracts are immutable Pydantic
models unless a type is documented as a service, store, or runtime object.

## Build tasks and tools

Use `agentic_rl_forge.contracts` for the shared data model:

- `TaskSpec`, `Message`, `ToolSpec`, and `VerifierSpec` define an executable task.
- `Trajectory`, `TrajectoryStep`, `AgentAction`, and `EnvironmentSnapshot` record an attempt.
- `Provenance` and `DataOrigin` distinguish on-policy samples from derived data.
- `RunManifest`, `DatasetCollectionManifest`, `CheckpointManifest`,
  `TrajectoryShardManifest`, and `TrainerBatchManifest` freeze handoff boundaries.

Search-R1 tasks can be created directly:

```python
from agentic_rl_forge.integrations import build_search_r1_task

task = build_search_r1_task(
    task_id="capital-france",
    question="What is the capital of France?",
    answers=["Paris"],
)
```

`agentic_rl_forge.environments` exports local and remote execution boundaries. For local tools,
wrap a Python handler in `FunctionTool`, or use `InMemorySearchTool`, `HTTPRetrievalTool`, and
`CalculatorTool`. `LocalToolEnvironment` validates task tool declarations before opening a
session and validates every call against its JSON Schema.

## Collect rollouts

`agentic_rl_forge.rollout` contains the runtime path:

- `AgentLoop` executes a multi-turn task and records masks, snapshots, rewards, and provenance.
- `OpenAICompatiblePolicy` connects to OpenAI-compatible vLLM or SGLang endpoints.
- `SearchR1Parser` and `NativeToolParser` convert model output into typed actions.
- `RolloutScheduler` collects task-grouped attempts with bounded concurrency.
- `SQLiteRolloutCallback`, `ShardedRolloutCallback`, and `MetricsRolloutCallback` persist and
  observe trajectories as they complete.
- `SignalAwareRolloutFilter` rejects stale, incomplete, low-variance, or low-diversity groups
  before trainer handoff.

Minimal grouped collection:

```python
from agentic_rl_forge.rollout import RolloutScheduler

scheduler = RolloutScheduler(loop_factory, max_concurrency=32, callbacks=callbacks)
batch = await scheduler.collect(tasks, rollouts_per_task=5, seed=0)
batch.validate_on_policy(rollouts_per_task=5)
```

The loop factory must return a fresh `AgentLoop`. Environments may share immutable tool
definitions, but each rollout creates its own environment session.

For a complete Search-R1 collection job, use `agentic_rl_forge.pipelines`:

```python
from pathlib import Path

from agentic_rl_forge.pipelines import (
    collect_search_r1,
    load_search_r1_collection_config,
)

config = load_search_r1_collection_config(Path("configs/search_r1_collection.yaml"))
result = await collect_search_r1(
    Path("data/qa.jsonl"),
    Path("artifacts/collections"),
    config,
)
```

`SearchR1CollectionConfig` validates bounded selection, concurrency, decoding, reward settings,
and secret-safe endpoint URLs. `SearchR1CollectionResult` points to the immutable run artifacts.
Pass `api_key` at runtime or use the CLI environment-variable option; do not place credentials in
the configuration file.

`RolloutPlanBuilder` from `agentic_rl_forge.rollout` can also be used independently. A
`RolloutPlan` assigns deterministic group, slot, trajectory, and seed identities. Pass the plan and
any validated `existing_trajectories` to `RolloutScheduler.collect`; callbacks receive reused
trajectories for attachment to the new run, while the policy is invoked only for missing slots.
`inspect_search_r1_plan` from `agentic_rl_forge.pipelines` performs the same cache and provenance
check without creating an output directory or contacting model and retrieval endpoints.

## Expand and run experiment matrices

Use `agentic_rl_forge.experiments` to compose existing configs and artifact contracts into a
content-addressed local experiment:

```python
from pathlib import Path

from agentic_rl_forge.experiments import ExperimentRunner, build_experiment_plan

plan = build_experiment_plan(Path("configs/experiments/search_r1_matrix.yaml"))
runner = ExperimentRunner(Path("."), Path("artifacts/experiment-state"))

preview = runner.report(plan)
report = runner.run(plan, confirm_plan_id=plan.plan_id, max_workers=2)
assert report.plan_id == plan.plan_id
```

`build_experiment_plan()` performs deterministic Cartesian expansion, applies type-aware exclusions,
overrides existing base-config paths, renders trial/stage templates, and enforces the matrix trial
cap. `ExperimentRunner` executes argument arrays without a shell, preserves append-only failed
attempts, conditionally publishes immutable success records, verifies artifact drift on every
retry, and evaluates benchmark/comparison gates into a deterministic `ExperimentReport`.

Typed artifact declarations summarize dataset, rollout, checkpoint, trainer-batch, benchmark, and
comparison identities without replacing their native deep verifiers.

Build a read-only cross-plan index and a deterministic multi-objective analysis:

```python
from agentic_rl_forge.experiments import (
    ExperimentAnalyzer,
    ExperimentIndexBuilder,
    ExperimentObjective,
    ExperimentObjectiveDirection,
    ExperimentRankingSpec,
    render_experiment_html,
)

index = ExperimentIndexBuilder(Path("."), Path("artifacts/experiment-state")).build()
score = next(metric for metric in index.metric_definitions if metric.metric == "group_pass_rate")
spec = ExperimentRankingSpec(
    objectives=(
        ExperimentObjective(
            metric_id=score.metric_id,
            direction=ExperimentObjectiveDirection.MAXIMIZE,
        ),
    )
)
analysis = ExperimentAnalyzer().analyze(index, spec)
dashboard = render_experiment_html(index, analysis)
```

Objective entries must be sorted by metric ID. The analyzer filters to completed trials by default,
requires every eligible trial to contain every objective, min-max normalizes each objective after
applying its direction, computes the weighted mean score, and marks the exact non-dominated set.
Optional baseline IDs add raw deltas and direction-aware improvements without changing source
reports. `render_experiment_csv`, `render_experiment_markdown`, and `render_experiment_html` are
byte deterministic for an unchanged index and analysis.

Preview and confirm an immutable promotion:

```python
from agentic_rl_forge.experiments import (
    ExperimentPromoter,
    ExperimentPromotionPolicy,
)

promoter = ExperimentPromoter(
    Path("."),
    Path("artifacts/experiment-state"),
    Path("artifacts/experiment-promotions"),
)
preview = promoter.preview(
    index,
    analysis,
    promotion_name="search-r1-candidate",
    plan_id=selected.plan_id,
    trial_id=selected.trial_id,
    policy=ExperimentPromotionPolicy(
        maximum_rank=1,
        require_pareto_front=True,
        minimum_approvals=2,
    ),
)
assert preview.eligible
record = promoter.promote(
    index,
    analysis,
    preview,
    confirm_preview_id=preview.preview_id,
    operator="release-operator",
    reason="approved after reproducibility and quality review",
    approvers=("evaluation-owner", "model-owner"),
)
```

`preview()` is read-only. It rebuilds the index, current report, artifact graph, checkpoint payload
verification, and dataset-lineage check, so a changed artifact or newly stale analysis produces a
different preview or an ineligible check. `promote()` repeats that preview before attempting one
conditional-create `decision.json`. Exact retries restore missing sidecars; a different decision
under the same promotion name is rejected.

Package, attest, inspect, and safely receive the metadata through the same Python API:

```python
from agentic_rl_forge.data import Ed25519ManifestSigner
from agentic_rl_forge.experiments import (
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestor,
)

archive_path = Path("artifacts/releases/search-r1-candidate.promotion.tar.gz")
archive = ExperimentPromotionArchive("artifacts/experiment-promotions/search-r1-candidate")
receipt = archive.pack(archive_path)

signer = Ed25519ManifestSigner.from_private_key_base64(private_key_text)
signed = ExperimentPromotionArchiveAttestor(signer).sign(receipt)

inspected = ExperimentPromotionArchive.inspect(
    archive_path,
    expected_sha256=receipt.content_digest,
)
verification = ExperimentPromotionArchiveAttestor.verify(
    signed,
    inspected,
    trusted_public_keys=(trusted_public_key_text,),
)
assert verification.valid

ExperimentPromotionArchive.unpack(
    archive_path,
    "received/search-r1-candidate",
    expected_sha256=receipt.content_digest,
)
```

`pack()` accepts only the eight promotion files and verifies their complete semantic relationship.
`inspect()` requires the expected digest and also rejects byte-valid tar streams that are not in the
project's canonical gzip/tar representation. `unpack()` writes into a temporary sibling, repeats all
checks, and atomically publishes a flat promotion directory. The attestor validates receipt equality,
embedded key identity, payload digest, Ed25519 signature, and membership in the caller's trust set.

Register the authenticated archive in an append-only lifecycle and publish an environment alias:

```python
from agentic_rl_forge.experiments import (
    ExperimentPromotionAliasAction,
    ExperimentPromotionRegistry,
    ExperimentPromotionStage,
)
from agentic_rl_forge.storage import LocalBlobStore

registry = ExperimentPromotionRegistry(LocalBlobStore("artifacts/promotion-registry"))
trusted_keys = (trusted_public_key_text,)

for stage in (
    ExperimentPromotionStage.CANDIDATE,
    ExperimentPromotionStage.STAGING,
    ExperimentPromotionStage.PRODUCTION,
):
    preview = registry.preview_lifecycle(
        archive_path,
        signed,
        trusted_public_keys=trusted_keys,
        target_stage=stage,
        operator="release-operator",
        reason=f"authorize transition to {stage.value}",
        authorizers=("release-owner",),
    )
    assert preview.eligible
    registry.execute_lifecycle(
        preview,
        archive_path,
        signed,
        trusted_public_keys=trusted_keys,
        confirm_preview_id=preview.preview_id,
    )

alias = registry.preview_alias(
    archive_path,
    signed,
    trusted_public_keys=trusted_keys,
    environment="production",
    action=ExperimentPromotionAliasAction.ASSIGN,
    operator="deployment-operator",
    reason="deploy the authorized production candidate",
    authorizers=("deployment-owner",),
)
assert alias.eligible
registry.execute_alias(
    alias,
    archive_path,
    signed,
    trusted_public_keys=trusted_keys,
    confirm_preview_id=alias.preview_id,
)
assert registry.status().valid
```

The registry uses conditional-create sequence slots for lifecycle events and compare-and-swap
generations for environment aliases. Every preview revalidates the exact signed archive, while the
operator and authorizer quorum remain an independent deployment-governance control. Rollback is
accepted only when an earlier event proves that the target promotion previously occupied the same
environment.

Fetch explicitly authorized remote references before acquisition:

```python
from agentic_rl_forge.experiments import (
    ExperimentPromotionRemoteFetcher,
    ExperimentPromotionRemotePolicy,
    ExperimentPromotionRemoteReaderRouter,
)

policy = ExperimentPromotionRemotePolicy(
    allowed_https_authorities=("models.example.com",),
    allowed_s3_buckets=("reviewed-models",),
)
with ExperimentPromotionRemoteReaderRouter(
    allowed_https_authorities=policy.allowed_https_authorities,
    allowed_s3_buckets=policy.allowed_s3_buckets,
) as reader:
    fetcher = ExperimentPromotionRemoteFetcher(reader)
    fetch_plan = fetcher.preview(
        archive_path,
        signed,
        "artifacts/promotion-cache",
        trusted_public_keys=trusted_keys,
        policy=policy,
    )
    fetch_record = fetcher.execute(
        fetch_plan,
        archive_path,
        signed,
        trusted_public_keys=trusted_keys,
        confirm_plan_id=fetch_plan.plan_id,
    )
```

The reader router implements HTTPS and S3. It denies sources outside the exact authority/bucket
allowlists, refuses HTTPS redirects, binds remote validators into the plan, and retrieves bounded
ranges into immutable resumable chunks. The final record exists only after complete SHA-256
verification against the signed promotion manifest.

Resolve and materialize the receiver's complete artifact graph through the same signed archive:

```python
from agentic_rl_forge.experiments import ExperimentPromotionAcquirer

acquirer = ExperimentPromotionAcquirer(
    project_root="/srv/agentic-rl-forge",
    state_root="/srv/agentic-rl-forge/artifacts/experiment-state",
    remote_records=(Path(fetch_record.plan.cache_root) / fetch_record.record_key,),
)
plan = acquirer.preview(
    archive_path,
    signed,
    "artifacts/received-promotions",
    trusted_public_keys=trusted_keys,
)
assert plan.eligible

record = acquirer.execute(
    plan,
    archive_path,
    signed,
    trusted_public_keys=trusted_keys,
    confirm_plan_id=plan.plan_id,
)
```

`preview()` is read-only and deterministically reports verified, missing, mismatched, unsafe, and
remote references. It also reparses native artifact contracts and validates checkpoint payload,
parent, and dataset ancestry. `execute()` recomputes the plan, streams exact local bytes into a
content-addressed receiver prefix, resumes matching partial progress, and publishes the canonical
acquisition record last. It never fetches a remote URI.

For planned work distributed across hosts, `SlotClaimCoordinator` provides an optional guard around
each missing slot:

```python
from agentic_rl_forge.storage import (
    ClaimStoreConsistency,
    LocalBlobStore,
    SlotClaimCoordinator,
)

claims = SlotClaimCoordinator(
    LocalBlobStore("artifacts/shared-coordination"),
    consistency=ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST,
)
scheduler = RolloutScheduler(
    loop_factory,
    callbacks=callbacks,
    slot_claims=claims,
    claim_owner_id="worker-42",
    claim_ttl_s=900,
    claim_renewal_interval_s=300,
)
```

Claims require a `RolloutPlan`; unplanned collection is unchanged. The owner ID must identify one
worker attempt, and the TTL must cover inference plus trajectory callbacks. The Search-R1 pipeline
creates a local coordinator automatically when `enable_slot_claims` is true.
`claims.inspect_plan(plan)` returns a read-only `SlotClaimPlanStatus` with one entry per plan slot
and aggregate counts for unclaimed, active, expired, completed, and abandoned states.

When `claim_renewal_interval_s` is configured, the scheduler appends `SlotClaimRenewal` records while
the rollout is active. The interval must not exceed half the TTL. A renewal failure cancels the
rollout before any later persistence callback can begin.

## Persist and hand off data

`agentic_rl_forge.storage` exposes three complementary persistence layers:

- `SQLiteTrajectoryStore` stores local run metadata and queryable immutable trajectories.
- `ShardedTrajectoryStore` commits one file per trajectory for recoverable multi-worker capture.
- `ConditionalBlobStore` is the provider-neutral object boundary implemented by
  `LocalBlobStore` and `S3ConditionalBlobStore`.
- `SlotClaimCoordinator` uses immutable claim epochs and release records to suppress simultaneous
  execution of one planned slot across workers.
- `RunArtifactBundle` builds and verifies one content-addressed manifest over immutable collection
  outputs and recursively verifies every trajectory shard.
- `RunArtifactTransport` publishes and fetches committed archives as bounded immutable chunks;
  `RunArtifactMirror` previews and executes trusted store-to-store replication of those graphs.

`TrainerBatchExporter` from `agentic_rl_forge.integrations` validates GRPO group shape,
on-policy provenance, policy version, and trajectory content before writing a content-verified
trainer payload and manifest.

For long-running jobs, `SQLiteTrajectoryStore.acquire_run_lease`, `renew_run_lease`, and
`finish_run(..., lease=token)` provide epoch-fenced writes. `list_run_liveness` is read-only and
classifies active, stale, and terminal runs. A `SQLiteRolloutCallback` must receive the lease token
when its run has an active or historical lease.

Stale-run mutation is a separate preview/execute API:

```python
preview = store.preview_run_reconciliation(run_id, stale_after_s=300)
if preview.eligible:
    record = store.reconcile_stale_run(
        preview,
        operator_id="operator@example.com",
        reason="worker host was terminated",
    )
```

`reconcile_stale_run` atomically revalidates the preview state digest, takes a higher lease epoch,
marks the run failed, and stores an immutable `RunReconciliationRecord`. It fails closed if the run
recovers or changes after preview. `get_run_reconciliation` and `list_run_reconciliations` expose the
audit history without mutating it.

```python
from agentic_rl_forge.integrations import TrainerBatchExporter
from agentic_rl_forge.storage import LocalBlobStore

exporter = TrainerBatchExporter(LocalBlobStore("artifacts/trainer-store"))
manifest = exporter.export(
    batch.trajectories,
    expected_policy_version=batch.policy_version,
    expected_group_size=5,
)
assert exporter.verify(manifest).valid
```

Verify a completed collection after local copying or publication:

```python
from agentic_rl_forge.storage import RunArtifactBundle

bundle = RunArtifactBundle("artifacts/collections")
manifest = bundle.load("artifacts/collections/runs/RUN_ID/artifact-manifest.json")
verification = bundle.verify(manifest)
assert verification.valid
assert verification.shard_verification is not None
assert verification.shard_verification.complete
```

Create and receive a deterministic single-run archive:

```python
from agentic_rl_forge.storage import RunArtifactArchive

archive = RunArtifactArchive("artifacts/collections")
receipt = archive.pack(manifest, "artifacts/releases/run.tar.gz")

received = RunArtifactArchive.unpack(
    "artifacts/releases/run.tar.gz",
    "artifacts/received/run",
    expected_sha256=receipt.content_digest,
)
assert received == receipt
```

`pack` verifies the source bundle and self-inspects the emitted archive. `unpack` requires an
expected SHA-256, safely extracts into a temporary directory, and returns only after full bundle
verification succeeds. The CLI automatically reads the adjacent `.sha256` sidecar.

Sign the exact receipt and verify it against an explicit trust set:

```python
from agentic_rl_forge.data import Ed25519ManifestSigner, RunArtifactArchiveAttestor

signer = Ed25519ManifestSigner.generate()
signed = RunArtifactArchiveAttestor(signer).sign(receipt)
verification = RunArtifactArchiveAttestor.verify(
    signed,
    receipt,
    trusted_public_keys=(signer.public_key_base64,),
)
assert verification.valid
```

The signed statement includes a key ID derived from the raw Ed25519 public key. Verification does
not trust that embedded key automatically; callers must provide the allowed public keys from their
own trust configuration. Multiple keys support rotation without weakening exact receipt binding.

Publish and fetch without buffering the full archive:

```python
from agentic_rl_forge.storage import LocalBlobStore, RunArtifactTransport

transport = RunArtifactTransport(
    LocalBlobStore("artifacts/release-store"),
    chunk_size_bytes=8 * 1024 * 1024,
    max_workers=4,
)
published = transport.publish(
    "artifacts/releases/run.tar.gz",
    expected_sha256=receipt.content_digest,
    attestation=signed,
)
fetched = transport.fetch(
    receipt.archive_id,
    "artifacts/downloads/run.tar.gz",
    trusted_public_keys=(signer.public_key_base64,),
    require_attestation=True,
)
assert fetched.commit == published.commit

status = transport.status(receipt.archive_id)
assert status.state.value == "committed"
```

The provider-neutral protocol stores bounded content-addressed chunks and writes a commit marker
last. Upload retries reuse verified remote chunks. Download retries rehash the local partial prefix
and continue only from a complete chunk boundary. `S3ConditionalBlobStore` can replace
`LocalBlobStore` without changing transport manifests or receipts.

For abandoned staging prefixes, call `preview_garbage_collection()` with a retention threshold.
Only an eligible preview's exact `state_digest` can be passed to `garbage_collect()` together with
an operator and reason. The execution path conditionally installs a tombstone on the release
decision key, deletes only identity-matched objects, resumes under the same tombstone after an
interruption, and returns an immutable `RunArtifactGcRecord`. A committed release is never
eligible.

For a shared store, `inventory()` returns one stable `RunArtifactStoreInventory` across every
release prefix. `plan_garbage_collection()` converts that reviewed snapshot into a deterministic
`RunArtifactGcPlan` containing only eligible staged prefixes and aggregate reclaimable bytes.
`execute_garbage_collection_plan()` requires the exact plan ID plus operator context, writes a
durable batch intent, runs a bounded number of the same per-prefix collectors, resumes completed or
interrupted members, and commits one `RunArtifactGcBatchRecord` only after every candidate has
member evidence.

Mirror one committed graph without reconstructing the archive locally:

```python
from agentic_rl_forge.storage import LocalBlobStore, RunArtifactMirror

mirror = RunArtifactMirror(
    LocalBlobStore("artifacts/primary-store"),
    LocalBlobStore("artifacts/replica-store"),
)
plan = mirror.plan(
    receipt.archive_id,
    trusted_public_keys=(signer.public_key_base64,),
    require_attestation=True,
)
record = mirror.execute(
    plan,
    confirm_plan_id=plan.plan_id,
    operator="release@example.com",
    reason="approved disaster-recovery replica",
    trusted_public_keys=(signer.public_key_base64,),
    max_workers=4,
)
assert record.plan.plan_id == plan.plan_id
```

`plan()` is read-only and binds the authenticated source graph, every source identity, the exact
destination snapshot, copy/reuse decisions, and byte totals. `execute()` revalidates the same trust
result and storage identities, accepts only exact partial-copy progress, transfers non-commit
objects with bounded concurrency, and publishes the commit last. It returns an immutable
`RunArtifactMirrorRecord` only after the destination graph passes complete release and signature
verification.

For an exact repository-level selection, use the same service's batch API:

```python
batch_plan = mirror.plan_batch(
    include_all_committed=True,
    trusted_public_keys=(signer.public_key_base64,),
    require_attestation=True,
    max_release_count=256,
    max_copy_bytes=1_099_511_627_776,
    max_workers=4,
)
batch_record = mirror.execute_batch(
    batch_plan,
    confirm_plan_id=batch_plan.plan_id,
    operator="release@example.com",
    reason="approved disaster-recovery synchronization",
    trusted_public_keys=(signer.public_key_base64,),
    release_workers=4,
    object_workers=2,
)
status = mirror.batch_status(
    batch_plan,
    operator=batch_record.intent.operator,
    reason=batch_record.intent.reason,
    trusted_public_keys=(signer.public_key_base64,),
)
assert status.record == batch_record

ledger = mirror.operations_ledger()
inspection = mirror.inspect_batch(
    batch_record.intent.batch_id,
    trusted_public_keys=(signer.public_key_base64,),
)
assert ledger.batch_count >= 1
assert inspection.health.value == "complete"
```

`plan_batch()` accepts either explicit archive IDs or `include_all_committed=True`, snapshots both
store inventories, aggregates the exact member copy/reuse budget, and binds hard release-count and
copy-byte caps into the plan ID. `execute_batch()` persists a canonical intent, bounds the product
of release and object workers to 64, and resumes through each member's normal mirror evidence.
`batch_status()` is read-only and distinguishes pending, partial, destination-complete, completed,
and invalid members while revalidating source trust and selected destination prefixes.

`operations_ledger()` discovers canonical destination evidence without a local plan.
`inspect_batch()` performs source-aware lookup by batch ID and derives stable health and resolution
eligibility. `resolve_batch()` immutably cancels an entirely unstarted intent or supersedes an
eligible invalid intent after exact status-digest confirmation. Completion and resolution share one
conditional-create decision key, and either missing query sidecar can be recovered from that
authoritative decision.

Use `DatasetManifestBuilder` and `Ed25519ManifestSigner` from `agentic_rl_forge.data` to freeze,
check split overlap, and optionally sign dataset inputs. Use `CheckpointRegistry` to register and
verify local checkpoint artifacts and lineage.

## Evaluate and derive data

`BenchmarkAggregator` in `agentic_rl_forge.evaluation` produces metrics with explicit
numerators and denominators. `BenchmarkComparator` performs task-paired comparisons and
deterministic bootstrap intervals.

Derived-data components never convert their output into on-policy trajectories:

- `PRMDatasetBuilder` creates step-level process-reward examples.
- `VerifiedRejectionSampler` keeps verified, deduplicated attempts.
- `HindsightTrajectoryRelabeling` requires executable evidence and verifier confidence.
- `NashMDBatchBuilder` builds preference self-play pairs.
- `AsyncMCTS` and `AgentTreeSearchController` perform PUCT search over environment branches.

## Embed services

`agentic_rl_forge.services` provides FastAPI app factories and dependency-free metrics:

- `create_retriever_app` serves the Search-R1 batch retrieval protocol.
- `create_prm_app` serves a `ProcessRewardModel`.
- `MetricsRegistry` records rollout and request telemetry and renders Prometheus text.

The `server` optional dependency is required to instantiate the HTTP applications. Core
contracts, storage, evaluation, data preparation, and local rollouts remain CPU-installable.

## Compatibility policy

Names exported from package `__init__.py` files are the intended public surface. Alpha releases
may evolve these APIs, but changes should preserve serialized contract readability or include an
explicit migration path. Implementation modules, underscored attributes, and optional native
verl adapter internals are not compatibility commitments.

See the runnable [offline pipeline](../examples/README.md) for a complete CPU-only path from
task construction through a verified trainer batch.

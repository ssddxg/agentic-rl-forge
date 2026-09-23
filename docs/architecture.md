# Architecture

AgenticRLForge separates online policy optimization from derived-data improvement while using
the same environment and verification stack for both paths.

```text
                              +---------------------------+
                              | Task and tool registry    |
                              +-------------+-------------+
                                            |
                                            v
+----------------+   actions   +------------+-------------+   observations
| Policy server  +-----------> | Stateful environment     +------------------+
| vLLM / SGLang  |             | search / API / browser   |                  |
+-------+--------+             +------------+-------------+                  |
        ^                                   |                                |
        |                                   v                                v
        |                      +------------+-------------+       +----------+---------+
        |                      | Trajectory recorder      |       | Outcome verifier   |
        |                      +------------+-------------+       | and reward stack   |
        |                                   |                     +----------+---------+
        |                                   +---------------+----------------+
        |                                                   |
        |                              +--------------------+--------------------+
        |                              |                                         |
        |                              v                                         v
        |                  +-----------+------------+              +-------------+-----------+
        |                  | On-policy GRPO path    |              | Derived-data path       |
        |                  | exact policy version  |              | MCTS / RS / HTR / replay|
        |                  +-----------+------------+              +-------------+-----------+
        |                              |                                         |
        +------------------------------+                        +----------------+-------------+
                                                               | SFT / preference / PRM data|
                                                               +------------------------------+

All completed trajectories flow through a content-verified persistence boundary. Evaluation,
derived-data builders, and observability consume the same immutable records instead of relying
on framework-specific in-memory objects.
```

## Package boundaries

- `contracts`: versioned task, tool, environment, trajectory, reward, and provenance models.
- `environments`: lifecycle and state management for search, APIs, browsers, and sandboxes.
- `rollout`: policy clients, agent loops, batching, budgeting, masks, and trajectory recording.
- `rewards`: outcome, process, cost, format, safety, and anti-hacking reward composition.
- `search`: PRM interfaces, MCTS, uncertainty estimation, and adaptive compute allocation.
- `data`: rejection sampling, replay, hindsight relabeling, manifests, and dataset export.
- `storage`: run manifests, SQLite persistence, atomic trajectory shards, deterministic archives,
  conditional-store release transport, trusted provider-to-provider mirroring, and state-bound
  staging cleanup.
- `evaluation`: benchmark aggregation and rollout-signal diagnostics with explicit denominators.
- `experiments`: deterministic config matrices, local stage orchestration, lineage evidence,
  regression reports, cross-plan discovery, ranking, Pareto analysis, immutable promotion,
  portable promotion attestations, and static dashboards over existing repository contracts.
- `algorithms`: GRPO utilities, Nash-MD preference mixtures, and self-play orchestration.
- `integrations`: verl, model servers, retrievers, MCP, WebArena, and observability adapters.
- `services`: retrieval and PRM endpoints, health checks, and Prometheus-compatible metrics.
- `pipelines`: configuration-driven end-to-end collection jobs over the stable runtime APIs.
- `cli`: local workflows, validation, evaluation, conversion, and diagnostics.

## Dependency direction

Core contracts have no dependency on training frameworks. Environments depend on contracts;
rollout depends on contracts and environments; algorithms consume recorded trajectories;
integrations adapt these stable boundaries to external systems. External framework objects must
not leak into the core models.

## Persistence invariants

- A trajectory ID is immutable. Re-inserting identical content is idempotent; different content
  under the same ID is rejected.
- Run manifests begin in `running` state and transition once to `completed` or `failed`.
- Renewable heartbeats are operational state stored separately from manifests. Lease epochs fence
  stale workers from attaching trajectories or performing terminal transitions.
- Stale-run reconciliation requires a state-bound preview, explicit operator reason, and a new lease
  epoch. The failed terminal transition and append-only reconciliation evidence commit atomically.
- Rollout plans are content-addressed by source, config, policy, task order, group size, and seed;
  their deterministic slots allow exact reuse without changing terminal run history.
- Experiment plans bind matrix/base-config bytes, type-preserving parameters, resolved configs,
  rendered commands, stage dependencies, artifacts, and metric gates. Successful stage evidence is
  immutable and retries revalidate it; failed attempts remain append-only; deterministic reports
  separate incomplete execution, invalid evidence, command failure, and measured regression.
- Experiment operations indexes discover only canonical persisted plans, regenerate reports from
  validated stage evidence, verify benchmark artifact digests, and assign semantic metric IDs from
  experiment, stage, output position, statistic, and unit. Ranking snapshots bind the exact index,
  objective directions and weights, eligibility policy, baseline, normalized scores, baseline
  improvements, and Pareto membership into a second content-derived identity.
- Experiment promotion rechecks the saved index against current state, reconstructs the selected
  report, verifies declared artifacts and local checkpoint payloads, and binds dataset lineage,
  rank/Pareto policy, technical checks, operator rationale, and approvers into one conditional-create
  decision. The decision embeds the preview and reproducibility manifest; model-card and canonical
  metadata files are recoverable sidecars rather than competing sources of truth.
- A promotion metadata archive accepts exactly the authoritative decision and seven deterministic
  sidecars. Canonical gzip/tar bytes make its digest reproducible; inspection reparses every contract,
  cross-checks identities, and regenerates the model card. Safe extraction completes in isolation.
  An optional Ed25519 attestation binds the exact archive receipt to a trusted publisher key while
  remaining deliberately separate from the decision's human approval assertions.
- Promotion lifecycle state is an append-only stream per promotion ID. Candidate, staging,
  production, and retired events occupy contiguous sequence keys and commit their predecessor.
  Environment aliases use independent generation keys as compare-and-swap slots; rollback must
  reference both the current event and a genuine earlier event for the target promotion. Retirement
  is blocked until all active aliases are detached.
- Promotion acquisition treats the reproducibility manifest as a closed receiver-side artifact
  graph. Project and experiment-state locators are resolved below explicit roots, streamed through
  size/digest checks, and reparsed through native contracts. Checkpoint payload identity, parent
  ancestry, and dataset lineage are graph invariants. Remote locators require a separate fetch plan
  that binds the publisher trust set, exact HTTPS authority or S3 bucket allowlists, provider
  validators, byte limits, chunking, and cache root. Execution uses conditional range reads and
  immutable chunks, then commits one receipt per artifact and a final fetch record. Acquisition
  accepts only a completely verified fetch record; otherwise the URI remains unresolved.
  Materialized files occupy a content-addressed prefix, use conditional file publication, and
  become complete only when the final acquisition record is present.
- Planned slots can use append-only claim epochs and renewal records. Acquisition happens before
  inference, heartbeats extend effective expiry, current ownership is checked around persistence,
  and release records preserve completed or abandoned outcomes without rewriting history.
- Run membership is recorded separately from trajectory content, so one verified trajectory can
  be referenced by multiple downstream runs without duplication.
- Distributed rollout capture writes independent atomic shards; finalized manifests provide
  immutable, content-addressed views that can coexist as a run grows.
- A completed collection emits a top-level content-addressed artifact manifest. It binds all
  immutable run files and recursively verifies the finalized shard manifest while excluding shared
  mutable coordination databases.
- A deterministic single-run archive contains exactly the files reachable from that artifact
  manifest. Canonical headers and order make identical inputs byte-reproducible; extraction accepts
  only regular files and completes in isolation before the verified tree is published.
- A signed archive attestation binds that archive's complete receipt to a derived Ed25519 key ID and
  signing time. Authenticity requires the embedded key to match an independently configured trust
  set; signature validity alone never establishes publisher identity.
- Conditional-store publication writes bounded content-addressed chunks before one immutable
  decision object. Upload and download workers are bounded; chunk ordering, digest verification,
  and commit-last visibility do not depend on completion order.
- Release mirroring binds a validated source graph, external publisher trust result, complete source
  and destination object identities, copy/reuse decisions, and unrelated destination keys into a
  content-derived plan. Execution accepts only exact partial progress, copies metadata and chunks
  with bounded concurrency, publishes the destination decision object last, and records immutable
  evidence only after complete destination validation.
- Store-wide mirror plans freeze both lifecycle inventories, an explicit sorted release selection,
  every member trust/diff plan, aggregate bytes, and declarative release/byte caps. Batch intent
  precedes member work; concurrency is bounded across releases and within each release; completed
  members survive partial failure; and a shared immutable decision key makes completed and resolved
  outcomes mutually exclusive. Sidecar loss is recoverable from the embedded terminal evidence.
  Destination-side ledger discovery and by-ID inspection do not require an operator's saved plan.
  Unselected prefixes are outside the authorization set and remain untouched.
- A release commit and garbage-collection tombstone compete for the same conditional-create key.
  Cleanup requires an aged staged inventory, an exact state-digest confirmation, unchanged object
  identities, and immutable operator evidence. Committed release graphs are never cleanup targets.
- Store-wide retention plans bind a complete lifecycle inventory, candidate previews, and aggregate
  bytes into one plan ID. Batch execution persists intent first, preserves every member's existing
  decision fence, and publishes completion evidence only after all reviewed candidates finish.
- Archive transport stores bounded content-addressed chunks and immutable metadata before a final
  commit marker. Remote and local consumers recognize only committed graphs; retries verify and
  reuse chunks, while visible destination archives are published only after complete validation.
- Checkpoint manifests bind steps to immutable config and artifact digests and can preserve parent
  checkpoint, dataset, and RNG lineage.
- Derived examples retain source IDs and transforms through the provenance contract.
- PRM train, validation, and test assignment is deterministic at task level to prevent leakage.

## Trainer-facing signal boundary

Rollout filtering operates only on complete, same-version on-policy groups. Reward variance and
semantic diversity are separate gates: different strings with identical rewards do not provide a
GRPO advantage signal, while reward variation from duplicated behavior may indicate unstable
verification. Accepted trajectories remain byte-for-byte equivalent to stored records.

Policy comparison is paired by task rather than computed from unrelated aggregate reports. This
keeps benchmark composition fixed and allows deterministic bootstrap intervals over task-level
deltas.

`RolloutScheduler` exposes per-trajectory and post-validation batch callbacks. Persistence and
telemetry therefore remain outside the agent loop and policy implementation. SQLite, sharded
storage, and metrics callbacks can be composed without changing rollout semantics.

The provider-neutral blob boundary requires conditional creation. Local files use exclusive hard
links; S3-compatible stores use `If-None-Match: *`. Trainer batch manifests are built only after
GRPO group validation and bind the exported payload to exact trajectory content digests.

Slot coordination additionally requires strong read-after-write behavior for exact reads and
prefix listings. This is an explicit deployment precondition because a conditional write cannot
compensate for a listing that hides a newer claim epoch.

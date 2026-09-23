# Operations and data lifecycle

This guide covers the CPU workflows that should be validated before using a GPU training stack.

## Trajectory lifecycle

Generate a local trajectory and write it as validated JSONL:

```bash
arf demo --output artifacts/demo-trajectories.jsonl
```

Import it into a run-scoped SQLite store:

```bash
arf trajectory-import artifacts/demo-trajectories.jsonl artifacts/trajectories.db \
  --run-name local-demo
```

The importer creates a run manifest, inserts the batch atomically, attaches records to the run,
and marks the run completed. If import fails, the manifest is marked failed. Identical records
can be imported again without duplicating trajectory content. A reused trajectory ID with a
different content digest is rejected.

Useful inspection commands:

```bash
arf trajectory-summary artifacts/trajectories.db
arf trajectory-export artifacts/trajectories.db artifacts/successes.jsonl \
  --status succeeded
arf inspect-trajectory artifacts/trajectories.db TRAJECTORY_ID
```

Filters are available for run ID, policy version, trajectory status, and provenance origin. The
Python `TrajectoryQuery` API additionally supports task, group, and environment-version filters.

For concurrent workers or interruption recovery, use the sharded workflow described in
[`distributed-rollouts.md`](distributed-rollouts.md). SQLite remains the indexed local query store;
shards are the streaming durability boundary and can be compacted back into validated JSONL.

## Run heartbeats and stale detection

Live Search-R1 collection acquires a fenced SQLite run lease before starting model requests. The
lease has an owner ID and monotonically increasing epoch. Background heartbeats renew its expiry;
every SQLite trajectory callback and the terminal run transition must present the matching token.
An expired or superseded worker therefore cannot continue attaching trajectories or mark a run
complete after another owner takes over the lease.

Inspect liveness without changing any run status:

```bash
arf run-liveness artifacts/collections/trajectories.db --stale-after 300
arf run-liveness artifacts/collections/trajectories.db \
  --stale-after 300 --only-stale --fail-on-stale
```

The report classifies runs as `active`, `stale`, or `terminal`. A running run is stale when its
lease expired, its lease was released without a terminal transition, or it has exceeded the startup
grace period without ever publishing a heartbeat. `--fail-on-stale` is suitable for monitoring and
CI because it exits nonzero but deliberately does not repair or fail the run.

Run manifests remain the immutable experiment record. Heartbeats live in a separate table and the
SQLite schema migrates existing v1 and v2 stores to v3 automatically. Operator-directed
reconciliation is kept separate from detection so a delayed worker, a dead host, and an
intentionally paused external service are not conflated.

### Explicit stale-run reconciliation

`run-reconcile` is preview-only unless `--execute` is present:

```bash
arf run-reconcile artifacts/collections/trajectories.db RUN_ID \
  --stale-after 300
```

The preview includes `eligible`, the exact liveness reason, current heartbeat, run digest, and a
`state_digest`. The state digest excludes observation time but binds the run content, heartbeat,
stale threshold, state, and detail. Repeating a preview while nothing changes produces the same
digest.

After checking the preview, execute with all confirmation fields:

```bash
arf run-reconcile artifacts/collections/trajectories.db RUN_ID \
  --stale-after 300 \
  --execute \
  --confirm-state-digest STATE_DIGEST \
  --operator operator@example.com \
  --reason "worker host was terminated" \
  --output artifacts/reconciliations/RUN_ID.json
```

Execution rechecks the state inside the same SQLite transaction. If a worker renewed, released,
took over the lease, or finished the run after preview, the digest or eligibility changes and no
mutation occurs. An eligible reconciliation takes a new lease epoch, marks the run failed, releases
the takeover lease, and inserts an immutable evidence record atomically. SQLite triggers reject
updates and deletes of reconciliation evidence.

The operator and reason are preserved in both the terminal run metadata and evidence record. Do not
put credentials or secrets in the reason. Repeating the exact execute command is idempotent; a
different operator, reason, or state digest is rejected.

## Planned-slot coordination

Run leases fence one run's SQLite writes. Planned-slot claims separately suppress simultaneous
model calls across continuation runs sharing the same rollout plan. Inspect their current state
without writing to the coordination directory:

```bash
arf slot-claim-status \
  artifacts/collections/plans/PLAN_ID.json \
  artifacts/collections \
  --fail-on-expired
```

Use the report to distinguish live owners from expired attempts before scheduling a retry. Storage
consistency, renewal-window, and clock requirements are documented in
[`slot-claims.md`](slot-claims.md).

## Release staging lifecycle

Chunked archive publication can leave immutable staged objects after a host or network failure.
They are invisible without a valid commit and reusable by a retry. Inspect the prefix before
deciding whether the publication should resume or be retired:

```bash
arf run-artifacts-status RUN_ARCHIVE_ID --store-root artifacts/release-store
arf run-artifacts-gc RUN_ARCHIVE_ID --store-root artifacts/release-store \
  --min-age-seconds 86400 --output artifacts/gc/RUN_ARCHIVE_ID.preview.json
```

Preview does not mutate storage. It reports eligibility and a stable digest over the exact object
identities and retention threshold. Execute only after confirming the publication is abandoned:

```bash
arf run-artifacts-gc RUN_ARCHIVE_ID --store-root artifacts/release-store \
  --min-age-seconds 86400 --execute \
  --confirm-state-digest STATE_DIGEST \
  --operator operator@example.com \
  --reason "publisher host was decommissioned" \
  --output artifacts/gc/RUN_ARCHIVE_ID.record.json
```

The decision tombstone fences late commits before deletion starts. New or mutated objects abort
cleanup, each deletion checks the observed identity, and exact retries resume after interruption.
Keep the completion record with release operations evidence. For S3-compatible services, validate
both conditional create and conditional delete behavior against the exact endpoint in use.

For periodic store-wide review, snapshot the inventory and build a deterministic plan:

```bash
arf run-artifacts-inventory --store-root artifacts/release-store \
  --output artifacts/gc/inventory.json
arf run-artifacts-gc-plan --store-root artifacts/release-store \
  --min-age-seconds 86400 --output artifacts/gc/retention-plan.json
```

Review the plan's full inventory, candidate IDs, per-prefix state digests, and reclaimable bytes.
Then execute with an exact plan-ID confirmation:

```bash
arf run-artifacts-gc-batch artifacts/gc/retention-plan.json \
  --store-root artifacts/release-store --execute \
  --confirm-plan-id RUN_GC_PLAN_ID \
  --operator operator@example.com \
  --reason "scheduled cleanup of expired staging prefixes" \
  --workers 4 --output artifacts/gc/batch-record.json
```

The saved batch intent makes an interrupted run resumable. Already completed members are validated
and reused, in-progress tombstones continue under their original previews, and the batch record is
created only after every candidate has a durable completion record. A newly created release prefix
is outside the reviewed candidate set and remains untouched.

## Release replication

Use a mirror plan for controlled disaster-recovery, regional, or provider migration of one
committed release. Planning performs all source trust and destination comparison work without
writing to either store:

```bash
arf run-artifacts-mirror-plan RUN_ARCHIVE_ID \
  --source-s3-bucket primary-releases \
  --source-s3-prefix search-r1 \
  --destination-s3-bucket replica-releases \
  --destination-s3-prefix search-r1 \
  --public-key trust/release.pub \
  --require-attestation \
  --output artifacts/mirrors/RUN_ARCHIVE_ID.plan.json
```

Review the source commit and signer, destination copy/reuse actions, unexpected destination keys,
object counts, byte totals, and plan ID. The saved contract can be parsed without provider access:

```bash
arf run-artifacts-mirror artifacts/mirrors/RUN_ARCHIVE_ID.plan.json
```

Execute the exact reviewed plan:

```bash
arf run-artifacts-mirror artifacts/mirrors/RUN_ARCHIVE_ID.plan.json \
  --source-s3-bucket primary-releases \
  --source-s3-prefix search-r1 \
  --destination-s3-bucket replica-releases \
  --destination-s3-prefix search-r1 \
  --public-key trust/release.pub \
  --execute --confirm-plan-id RUN_MIRROR_PLAN_ID \
  --operator operator@example.com \
  --reason "approved cross-region release replication" \
  --workers 4 \
  --output artifacts/mirrors/RUN_ARCHIVE_ID.record.json
```

Use the same trusted key set at planning and execution. Never substitute a newly generated plan
after review: source or destination drift should trigger a fresh operational decision. If execution
is interrupted, rerun the original command and plan. Exact already-created objects are treated as
progress, while any mutation, deletion, conflicting bytes, or unplanned key aborts. Consumers see
the replica only after the commit object is copied last and the destination passes full validation.

Retain the plan and mirror record together. The destination also stores immutable evidence at
`run-release-mirrors/records/<mirror_id>.json`. Store credentials remain outside those contracts.
For S3-compatible services, validate conditional creation and strong exact-read/list visibility for
the endpoint and replication topology in use.

### Repository-wide replication

For disaster-recovery synchronization or a provider migration, freeze either a repeated
`--archive-id` allowlist or all currently committed source releases:

```bash
arf run-artifacts-mirror-batch-plan \
  --all-committed \
  --source-s3-bucket primary-releases \
  --destination-s3-bucket replica-releases \
  --public-key trust/release.pub --require-attestation \
  --max-releases 256 --max-copy-bytes 1099511627776 \
  --workers 4 \
  --output artifacts/mirrors/store.plan.json
```

Review both inventory digests, the sorted release set, every member signer and action summary, and
aggregate copy/reuse bytes. Validate the saved plan offline with
`arf run-artifacts-mirror-batch artifacts/mirrors/store.plan.json`, then execute it:

```bash
arf run-artifacts-mirror-batch artifacts/mirrors/store.plan.json \
  --source-s3-bucket primary-releases \
  --destination-s3-bucket replica-releases \
  --public-key trust/release.pub \
  --execute --confirm-plan-id RUN_MIRROR_BATCH_PLAN_ID \
  --operator operator@example.com \
  --reason "approved disaster-recovery synchronization" \
  --release-workers 4 --object-workers 2 \
  --output artifacts/mirrors/store.record.json
```

The product of both worker settings cannot exceed 64. The destination stores batch intent before
members start, individual evidence as releases complete, and the final batch record only after all
members finish. On interruption, rerun the exact command; do not regenerate the plan to hide a
changed selected prefix.

Inspect persisted and in-flight progress:

```bash
arf run-artifacts-mirror-batch-status artifacts/mirrors/store.plan.json \
  --source-s3-bucket primary-releases \
  --destination-s3-bucket replica-releases \
  --public-key trust/release.pub \
  --operator operator@example.com \
  --reason "approved disaster-recovery synchronization" \
  --fail-on-invalid --fail-on-incomplete \
  --output artifacts/mirrors/store.status.json
```

An `invalid` member indicates trust or selected-prefix drift and suppresses aggregate progress
claims. `partial` is exact reusable object progress without a commit; `destination_complete` means
the graph committed but its member evidence still needs the original execute command. Newly created
unselected prefixes are not implicit batch targets.

Inventory destination-side operations independently of local plan files:

```bash
arf run-artifacts-mirror-batch-list \
  --destination-s3-bucket replica-releases \
  --fail-on-unclassified \
  --output artifacts/mirrors/ledger.json

arf run-artifacts-mirror-batch-inspect RUN_MIRROR_BATCH_ID \
  --source-s3-bucket primary-releases \
  --destination-s3-bucket replica-releases \
  --public-key trust/release.pub \
  --fail-on-unhealthy \
  --output artifacts/mirrors/inspection.json
```

To retire an unstarted or permanently invalid intent, first stop its executors and run
`run-artifacts-mirror-batch-resolve` without `--execute`. Review the status digest and eligibility,
then repeat with `--execute`, `--kind cancelled` or `--kind superseded`, the exact digest, resolver,
and reason. Supersession requires a different replacement plan ID. Never resolve active partial or
destination-complete work; continue the original command instead.

The batch's `decision.json` is the terminal fence. Completion and resolution compete to create it,
and `record.json` or `resolution.json` can be regenerated after a crash. Resolution does not delete
objects or member evidence already created by an in-flight worker, so quiescing the operation is an
operational prerequisite for clean cancellation even though the completed batch outcome remains
atomically fenced.

## Experiment matrix operations

Expand the matrix before allocating model or retrieval capacity:

```bash
arf experiment-plan configs/experiments/search_r1_matrix.yaml \
  --output artifacts/experiments/search-r1.plan.json
```

Review the trial count, resolved config digests, rendered commands, artifact paths, and gates. A
status-only run does not start stages:

```bash
arf experiment-run artifacts/experiments/search-r1.plan.json \
  --root . \
  --state-dir artifacts/experiment-state \
  --fail-on-incomplete
```

Execute with the exact plan ID and bounded trial concurrency. Use separate exit gates so operators
can distinguish unfinished work, command/evidence failures, and valid experiments that regressed:

```bash
arf experiment-run artifacts/experiments/search-r1.plan.json \
  --root . \
  --state-dir artifacts/experiment-state \
  --execute --confirm-plan-id EXPERIMENT_PLAN_ID \
  --workers 2 \
  --fail-on-incomplete --fail-on-failed --fail-on-regression \
  --output artifacts/experiments/search-r1.report.json
```

Rerun the same command after correcting an external service or missing input. Successful stages are
revalidated and reused; failed attempts remain available under the state directory. Do not delete a
success record to force a changed output through the old plan. Preserve the plan, state directory,
and final report with the run's release evidence.

Discover all canonical plans and reconstruct their current reports without executing a stage:

```bash
arf experiment-index \
  --root . \
  --state-dir artifacts/experiment-state \
  --output artifacts/experiments/index.json \
  --csv-output artifacts/experiments/index.csv \
  --fail-on-issues
```

The command validates each persisted `plan.json`, report and artifact evidence, then emits one row
per trial. The CSV begins with plan/report/trial identity and state, followed by stable parameter
columns and content-derived metric IDs. Use the JSON metric catalog to map each ID to experiment,
stage, output position, metric, statistic, and unit. Unknown top-level state entries and invalid plan
directories remain visible as discovery issues; `--fail-on-issues` makes them fail automation after
the outputs are written.

Select objective IDs from that catalog and rank completed trials:

```bash
arf experiment-analyze artifacts/experiments/index.json \
  --objective EXPERIMENT_METRIC_ID:maximize \
  --objective COST_METRIC_ID:minimize:0.5 \
  --baseline-plan-id EXPERIMENT_PLAN_ID \
  --baseline-trial-id EXPERIMENT_TRIAL_ID \
  --output artifacts/experiments/analysis.json \
  --markdown-output artifacts/experiments/dashboard.md \
  --html-output artifacts/experiments/dashboard.html
```

Weights are positive and objectives are min-max normalized after applying maximize/minimize
direction. The score is a presentation ranking, while Pareto membership is calculated directly from
the unnormalized objective values. Exact plan and trial IDs select the baseline because one trial ID
can legitimately appear in more than one plan. `--include-regressions` admits completed trials that
failed a declared gate; failed and incomplete trials remain visible but ineligible. Use
`--experiment-name`, `--plan-id`, `--max-candidates`, and `--fail-on-ineligible` to bound operational
scope and enforce CI policy.

The analysis JSON, Markdown, CSV, and standalone HTML contain no observation timestamp, remote
asset, or script. Identical inputs produce identical bytes. Weighted rank and Pareto membership do
not establish statistical significance; use paired comparison reports and confidence-bound gates
for release decisions.

### Promote one analyzed trial

Promotion is preview-only unless `--execute` is present:

```bash
arf experiment-promote \
  artifacts/experiments/index.json \
  artifacts/experiments/analysis.json \
  --name search-r1-candidate \
  --plan-id EXPERIMENT_PLAN_ID \
  --trial-id EXPERIMENT_TRIAL_ID \
  --root . \
  --state-dir artifacts/experiment-state \
  --promotion-dir artifacts/experiment-promotions \
  --maximum-rank 1 \
  --require-pareto-front \
  --minimum-approvals 2 \
  --preview-output artifacts/experiments/promotion-preview.json \
  --fail-on-ineligible
```

The preview rebuilds the cross-plan index and selected report, then checks current report identity,
analysis eligibility, trial state, optional rank/Pareto limits, required artifact kinds, checkpoint
count, local checkpoint payload size/SHA-256, and checkpoint-to-dataset lineage. The default required
kinds are `benchmark_report` and `checkpoint_manifest`; repeat `--required-artifact-kind` to replace
that set. Defaults require a clean index, one fully local and verified checkpoint, dataset lineage,
no regression state, and an approver distinct from the operator.

After reviewing every check and the manifest, confirm its exact ID:

```bash
arf experiment-promote \
  artifacts/experiments/index.json \
  artifacts/experiments/analysis.json \
  --name search-r1-candidate \
  --plan-id EXPERIMENT_PLAN_ID \
  --trial-id EXPERIMENT_TRIAL_ID \
  --root . \
  --state-dir artifacts/experiment-state \
  --promotion-dir artifacts/experiment-promotions \
  --maximum-rank 1 --require-pareto-front --minimum-approvals 2 \
  --execute --confirm-preview-id EXPERIMENT_PROMOTION_PREVIEW_ID \
  --operator release-operator \
  --reason "approved after reproducibility and quality review" \
  --approver evaluation-owner --approver model-owner \
  --record-output artifacts/experiments/promotion-record.json
```

Execution recomputes the preview to close the review-to-write race. It creates
`PROMOTION_DIR/NAME/decision.json` before publishing recoverable sidecars: `record.json`,
`manifest.json`, `model-card.md`, and canonical snapshots of the plan, report, index, and analysis.
The reproducibility manifest references digest-bound project, state, and remote artifacts but does
not copy model weights or other mutable payloads. Exact retries recover missing sidecars; another
operator, reason, approval set, candidate, or policy cannot replace an existing name.

Relaxations such as `--allow-regression`, `--allow-index-issues`,
`--allow-unverified-checkpoint`, `--allow-multiple-checkpoints`,
`--skip-dataset-lineage`, or `--operator-may-approve` are explicit policy changes and therefore
change the preview ID. Use them only when an external release process supplies the missing control.

Package the finalized metadata and verify it in CI:

```bash
arf experiment-promotion-pack \
  artifacts/experiment-promotions/search-r1-candidate \
  artifacts/releases/search-r1-candidate.promotion.tar.gz

arf experiment-promotion-inspect \
  artifacts/releases/search-r1-candidate.promotion.tar.gz
```

`experiment-promotion-inspect` exits nonzero for a digest, archive-layout, contract, identity, or
model-card mismatch. The adjacent checksum is used by default; use `--expected-sha256` when the
trusted digest arrives through another channel, or `--checksum` to name an explicit checksum file.

For an authenticated release, sign the canonical receipt and make the trusted public key available
to the receiver through an independent channel:

```bash
arf experiment-promotion-sign \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --private-key secrets/promotion-release.key

arf experiment-promotion-signature-verify \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --public-key trust/promotion-release.pub

arf experiment-promotion-unpack \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  received/search-r1-candidate \
  --attestation artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --public-key trust/promotion-release.pub
```

Supplying `--attestation` makes both inspection and unpack require at least one trusted
`--public-key`. Verification accepts a repeated key option for planned rotation and exits `1` for a
well-formed but untrusted or invalid signature. Archive or argument errors exit `2`. Safe unpack
publishes the destination only after canonical-byte and semantic verification succeeds.

### Operate the promotion lifecycle registry

Every lifecycle and environment change is preview-only unless `--execute` is present. A lifecycle
preview always re-inspects the archive, verifies its attestation against the supplied trust set,
loads the complete append-only stream, checks the legal source/target stage, and evaluates deployment
authorization separately from publisher trust.

```bash
arf experiment-promotion-lifecycle RELEASE.promotion.tar.gz RELEASE.attestation.json \
  --registry-dir artifacts/promotion-registry \
  --target-stage staging \
  --operator release-operator \
  --reason "advance after staging acceptance" \
  --authorizer evaluation-owner \
  --public-key trust/promotion-release.pub \
  --preview-output artifacts/releases/staging-preview.json \
  --fail-on-ineligible
```

Use `--minimum-authorizers` to raise the quorum. The operator cannot satisfy it unless
`--operator-may-authorize` is explicit. These identities are audit assertions rather than
cryptographic user authentication. Confirm the exact preview with the same arguments:

```bash
arf experiment-promotion-lifecycle RELEASE.promotion.tar.gz RELEASE.attestation.json \
  --registry-dir artifacts/promotion-registry \
  --target-stage staging \
  --operator release-operator \
  --reason "advance after staging acceptance" \
  --authorizer evaluation-owner \
  --public-key trust/promotion-release.pub \
  --execute --confirm-preview-id PROMOTION_LIFECYCLE_PREVIEW_ID \
  --event-output artifacts/releases/staging-event.json
```

Environment alias assignment and rollback use the same two-step workflow:

```bash
arf experiment-promotion-alias RELEASE.promotion.tar.gz RELEASE.attestation.json \
  --registry-dir artifacts/promotion-registry \
  --environment production --action assign \
  --operator deployment-operator \
  --reason "deploy after production authorization" \
  --authorizer deployment-owner \
  --public-key trust/promotion-release.pub
```

The default alias policy accepts only the `production` lifecycle stage. Repeat `--allowed-stage`
for an explicitly broader non-production environment. `--action rollback` is eligible only when the
target archive's promotion ID appears in an earlier generation of the same environment. A normal
assignment to the already-current promotion is rejected as a no-op.

Sequence and generation paths are conditional-create CAS slots. Exact retries are idempotent; a
different decision at the same slot fails closed, and later state makes an old preview stale. Before
and after operational changes, preserve deterministic status evidence:

```bash
arf experiment-promotion-registry-status artifacts/promotion-registry \
  --output artifacts/releases/promotion-registry-status.json \
  --fail-on-issues
```

Do not add notes or mutable pointers inside the registry root. Unknown files are integrity issues.
Keep human runbooks and deployment logs outside this append-only namespace.

### Acquire promotion artifacts on a receiver

If the promotion manifest contains HTTPS or S3 artifacts, authorize and fetch them first:

```bash
arf experiment-promotion-fetch-remote \
  RELEASE.promotion.tar.gz RELEASE.attestation.json artifacts/promotion-cache \
  --public-key trust/promotion-release.pub \
  --https-allow-authority models.example.com \
  --s3-allow-bucket reviewed-models \
  --plan-output artifacts/releases/remote-fetch-plan.json \
  --fail-on-ineligible

arf experiment-promotion-fetch-remote \
  RELEASE.promotion.tar.gz RELEASE.attestation.json artifacts/promotion-cache \
  --public-key trust/promotion-release.pub \
  --https-allow-authority models.example.com \
  --s3-allow-bucket reviewed-models \
  --execute --confirm-plan-id REMOTE_FETCH_PLAN_ID \
  --record-output artifacts/releases/remote-fetch-record.json
```

The preview performs only metadata requests to explicitly allowlisted sources. HTTPS redirects are
rejected. Execution binds `ETag`, version ID, or `Last-Modified` when available, uses conditional
bounded range reads, and resumes exact cached chunks after interruption. Unknown cache files,
symbolic links, changed source metadata, range inconsistencies, and final digest mismatches fail
closed. `--allow-missing-source-validator` is an explicit relaxation; SHA-256 verification still
applies, but the operator accepts weaker protection against a source changing between ranges.

Use a destination that is separate from both receiver source roots. Preview writes no acquisition
files:

```bash
arf experiment-promotion-acquire RELEASE.promotion.tar.gz RELEASE.attestation.json \
  artifacts/received-promotions \
  --root /srv/project --state-dir /srv/experiment-state \
  --public-key trust/promotion-release.pub \
  --remote-record artifacts/releases/remote-fetch-record.json \
  --plan-output artifacts/releases/acquisition-plan.json \
  --fail-on-ineligible
```

Review the signer trust result, every artifact availability and native-contract result, checkpoint
payload/ancestry/lineage issues, unique materialized byte total, and destination root. Confirm the
same plan with `--execute --confirm-plan-id`. Use `--max-materialized-bytes` as a receiver capacity
and authorization limit, not merely a warning threshold.

Execution uses `acquisitions/ACQUISITION_ID/` below the destination. Local source files are streamed
to temporary siblings and published with exclusive links. Exact preexisting files count as progress;
conflicting files and unknown entries fail. The canonical `record.json` is written last and is the
completion boundary. Re-run the original command after interruption instead of deleting partial
matching files or generating a replacement plan.

The acquisition preview never dereferences remote URIs. It verifies only the supplied remote-fetch
record and its immutable cache. `--allow-unresolved-remote` records a deliberate partial handoff,
and `--allow-missing-checkpoint-payloads` is separately required when such a URI is a checkpoint
payload. The native-contract, checkpoint-ancestry, and dataset-lineage relaxations are also
independent and change the plan identity. Preserve both plans and final records with the signed
promotion archive and deployment evidence.

## Benchmark aggregation

```bash
arf evaluate artifacts/trajectories.db --benchmark local-demo \
  --output artifacts/local-demo-report.json
```

Every `MetricValue` contains `value`, `numerator`, `denominator`, and `unit`. Reports include:

- attempt success and group pass rates;
- mean reward, steps, generated tokens, observation tokens, tool calls, and duration;
- tool success, error, and truncation rates;
- zero-variance and learning-signal group rates;
- semantic trajectory diversity within each rollout group.

A group has a usable learning signal when it contains at least two attempts and its reward
standard deviation exceeds the configured threshold. This does not replace trainer-side checks,
but it identifies GRPO batches whose normalized advantages would carry no useful signal.

Filter groups before producing a trainer input:

```bash
arf filter-rollouts artifacts/trajectories.db artifacts/filtered.jsonl \
  --policy-version policy-v1 --expected-group-size 5 \
  --min-reward-stddev 0.01 --min-unique-ratio 0.4 \
  --report artifacts/filter-report.json
```

Eligible groups are ranked by reward standard deviation, semantic uniqueness, and group size.
`--keep-top-fraction` can cap an update to the strongest groups. Filtering validates the on-policy
provenance and policy version but does not modify accepted trajectories.

## Paired policy comparison

```bash
arf compare artifacts/trajectories.db \
  --baseline-policy policy-v1 --candidate-policy policy-v2 \
  --benchmark search-r1 --bootstrap-samples 5000 \
  --output artifacts/comparison.json
```

The comparator intersects task IDs, requires equal attempt counts by default, computes per-task
values, and bootstraps paired deltas. Reports cover task pass rate, reward, generated tokens,
steps, and tool calls. Each metric includes the baseline and candidate means, absolute and relative
deltas, confidence bounds, sample count, and unit.

## Process-reward datasets

```bash
arf build-prm-dataset artifacts/trajectories.db artifacts/prm.jsonl \
  --gamma 1.0 --min-confidence 0.8 --skip-zero-variance-groups
```

Each example contains the pre-action messages and snapshot, the selected action, prior step
history, a clipped discounted-return target, confidence and sample weight, policy/environment
versions, and immutable lineage. Raw unclipped return, group reward statistics, and standardized
group advantage are retained as metadata.

Dataset split assignment hashes `task_id`, not individual steps or trajectories. This keeps all
attempts and steps for a task in exactly one of `train`, `validation`, or `test`.

## Metrics endpoints

HTTP services expose Prometheus text format at `/metrics`. The standard metric families are:

| Metric | Meaning |
| --- | --- |
| `arf_http_requests_total` | Requests by service, method, route, and status |
| `arf_http_request_duration_seconds` | Request latency histogram |
| `arf_retrieval_queries_total` | Search queries processed |
| `arf_prm_batches_total` | PRM batches by model version and outcome |
| `arf_prm_examples_total` | Process-reward examples scored |
| `arf_prm_batch_duration_seconds` | PRM scoring latency |
| `arf_trajectories_total` | Completed trajectories by status and origin |
| `arf_generated_tokens_total` | Generated rollout tokens |
| `arf_observation_tokens_total` | Observation tokens excluded from policy loss |
| `arf_trajectory_reward` | Total reward distribution |
| `arf_trajectory_duration_seconds` | End-to-end rollout duration |

The in-process `MetricsRegistry` has no external monitoring dependency. A deployment can scrape
the endpoint directly or bridge registry snapshots into its existing telemetry stack.

## Checkpoint manifests

Register a checkpoint only after all local artifacts have been flushed:

```bash
arf checkpoint-register artifacts/checkpoints \
  --run-id RUN_ID --step 100 --policy-version policy-v2 \
  --config configs/search_r1.yaml \
  --artifact actor=/checkpoints/step-100/actor.safetensors \
  --artifact optimizer=/checkpoints/step-100/optimizer.pt
```

List and verify manifests:

```bash
arf checkpoint-list artifacts/checkpoints --run-id RUN_ID
arf checkpoint-verify artifacts/checkpoints CHECKPOINT_ID
```

Local files are checked by size and SHA-256. Remote artifact references remain explicitly
unverified rather than being treated as valid local files. Manifest IDs are immutable, writes are
atomic, and `latest()` resolves by training step, creation time, and checkpoint ID.

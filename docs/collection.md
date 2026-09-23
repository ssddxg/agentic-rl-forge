# Search-R1 rollout collection

The `collect-search-r1` command runs grouped, on-policy rollouts against an OpenAI-compatible
vLLM or SGLang endpoint and a Search-R1-compatible retrieval service. It is the production-facing
counterpart to the deterministic offline example.

## Input and configuration

Input is JSONL with one task per line:

```json
{"id":"nq-1","question":"What is the capital of France?","answer":"Paris"}
```

`answer` may be a string or a list of accepted strings. Start from
[`configs/search_r1_collection.yaml`](../configs/search_r1_collection.yaml) and set the model,
policy version, endpoints, selection bounds, group size, concurrency, decoding, and reward values.
The checked-in configuration limits collection to 64 tasks so a first run is bounded.

`run_lease_ttl_s` and `heartbeat_interval_s` control liveness fencing. The heartbeat interval must
not exceed half the lease TTL. These operational settings, concurrency, retries, and timeouts do not
change rollout-plan identity; decoding, reward, task, retrieval, model, policy, and `plan_salt`
changes do.

`enable_slot_claims` enables cross-process claims for missing plan slots. The built-in local path
stores them under the collection root. `slot_claim_ttl_s` is the ownership window granted by each
claim or renewal, and `slot_claim_renewal_interval_s` controls append-only renewal heartbeats. The
interval must not exceed half the TTL. All three settings are operational and do not change
rollout-plan identity. See [`slot-claims.md`](slot-claims.md) before using a shared or
object-store-backed coordinator.

The model must emit the Search-R1 protocol:

```text
<think>reasoning</think><search>query</search>
<think>reasoning over evidence</think><answer>final answer</answer>
```

Retrieval observations are returned to the model as `<information>...</information>` user
messages. Observation tokens are recorded with a zero response mask and are not treated as model
generated tokens.

## Run a collection

Start the model and retrieval services, then run:

```bash
export OPENAI_API_KEY=your-model-endpoint-key
arf collect-search-r1 data/qa.jsonl artifacts/collections \
  --config configs/search_r1_collection.yaml
```

Local vLLM and SGLang deployments that do not require authentication can run without the
environment variable. To use a different variable name, pass `--api-key-env MODEL_API_KEY`. The
key value is never written to the run configuration, logs, summary, or artifacts.

Endpoint URLs containing embedded credentials, query parameters, or fragments are rejected to
prevent accidental secret persistence in run manifests. Put model credentials in the environment
or a preconfigured HTTP client when using the Python API.

Inspect the exact work remaining without contacting either service:

```bash
arf search-r1-plan-status data/qa.jsonl artifacts/collections \
  --config configs/search_r1_collection.yaml
```

The JSON output includes every reusable and missing slot with its stable task, group, trajectory,
rollout index, and seed. It also reports conflicting trajectory IDs and exits nonzero when a stored
trajectory uses a planned ID but fails policy or slot-provenance validation. If the output root has
not been created, the command is read-only and reports every slot as missing.

## Artifacts

Multiple collections can share one output root:

```text
artifacts/collections/
├── trajectories.db
├── plans/<plan_id>.json
├── coordination/slot-claims/plans/<plan_id>/slots/<slot_id>/
├── shards/runs/<run_id>/
└── runs/<run_id>/
    ├── artifact-manifest.json
    ├── trajectories.jsonl
    ├── benchmark-report.json
    ├── metrics.prom
    ├── run-manifest.json
    └── summary.json
```

Each trajectory is persisted to SQLite and an atomic shard before it can enter the completed
batch. The shard manifest is finalized only after policy-version and group-shape validation. The
run manifest fingerprints the source bytes, records the exact task window and collection config,
and records only whether an API key was configured.

Before collection, the pipeline builds a content-addressed rollout plan. Its identity includes the
source digest, full collection config, policy version, ordered task IDs, group size, and base seed.
Every task group and rollout slot receives a deterministic group ID, trajectory ID, slot ID, and
seed. Re-running the same input and config reads matching trajectories from SQLite, validates their
policy and slot provenance, attaches them to a new continuation run, and calls the model only for
missing slots.

Completed and failed run manifests remain terminal; resume never rewrites their history. A fully
cached rerun makes no model or retrieval requests but still creates an independently inspectable
run with a new shard manifest. Change `plan_salt`, the seed, policy version, or another sampling
setting when a fresh set of attempts is required.

`artifact-manifest.json` is the portable verification root for a completed run. It binds exact byte
sizes and SHA-256 digests for the rollout plan, terminal run manifest, trajectory JSONL, benchmark
report, Prometheus metrics, summary, and finalized shard manifest. Verification then follows the
shard manifest and checks every referenced trajectory shard, trajectory identity, plan membership,
run/report/summary linkage, and unexpected shard files.

```bash
arf run-artifacts-verify \
  artifacts/collections/runs/RUN_ID/artifact-manifest.json
```

Paths inside the manifest are relative to the collection root. Copying the complete collection root
to another directory preserves verification. Use `--root` only when the manifest is not at the
standard `runs/<run_id>/artifact-manifest.json` location. The shared `trajectories.db`, slot claims,
and heartbeats are operational state and are deliberately excluded from the immutable run bundle.

## Pack one run for publication

Create a self-contained archive from a verified run:

```bash
arf run-artifacts-pack \
  artifacts/collections/runs/RUN_ID/artifact-manifest.json \
  artifacts/releases/RUN_ID.tar.gz
```

The command verifies the source bundle before packing, writes a deterministic gzip-compressed
USTAR archive, reads it back through the strict archive validator, and publishes a neighboring
`RUN_ID.tar.gz.sha256` file. Repeating the command with unchanged inputs produces identical archive
bytes and is idempotent when the existing output matches. It refuses to replace an archive or
checksum with different content.

The archive contains only the canonical artifact manifest, its seven immutable top-level files,
and the exact trajectory shards referenced by the shard manifest. Member timestamps, ownership,
permissions, order, and header layout are fixed. Directories and mutable databases, claims,
renewals, releases, and heartbeats are excluded.

On the receiving host, keep the archive and checksum sidecar together:

```bash
arf run-artifacts-unpack \
  artifacts/releases/RUN_ID.tar.gz \
  artifacts/received/RUN_ID
```

The destination must not already exist. Unpacking validates the sidecar before decompression,
requires the artifact manifest to be the first member, and rejects absolute or escaping paths,
links, special files, duplicate or unexpected members, reordered members, altered metadata,
truncation, and content-digest mismatches. Extraction happens in a sibling temporary directory;
the destination appears only after the extracted bundle passes full semantic and recursive shard
verification. Use `--expected-sha256 DIGEST` when the expected digest arrives through a separate
trusted channel, or `--checksum PATH` for a sidecar stored elsewhere.

## Authenticate a published archive

The checksum detects accidental corruption but cannot authenticate a publisher if an attacker can
replace both the archive and sidecar. Install the optional signing dependency and create a release
key outside the repository:

```bash
pip install -e ".[signing]"
arf manifest-keygen secrets/release.key artifacts/release.pub

arf run-artifacts-sign \
  artifacts/releases/RUN_ID.tar.gz \
  artifacts/releases/RUN_ID.attestation.json \
  --private-key secrets/release.key
```

Signing first performs strict archive inspection, then signs a canonical statement containing the
complete archive receipt, run ID, artifact-manifest ID, signer key ID, and timezone-aware signing
time. The output is immutable and an exact retry returns the existing valid attestation.

Verify against one or more independently distributed trusted public keys:

```bash
arf run-artifacts-signature-verify \
  artifacts/releases/RUN_ID.tar.gz \
  artifacts/releases/RUN_ID.attestation.json \
  --public-key trust/release-2026.pub \
  --public-key trust/release-2027.pub
```

Supplying multiple keys supports a controlled rotation window. Verification requires an exact
archive receipt match, a valid statement digest and Ed25519 signature, a signer key ID derived from
the embedded public key, and membership in the supplied trust set. A cryptographically valid
signature from an unknown key fails trust validation.

Trusted-signature verification can authorize unpacking even when the checksum sidecar was not
transported:

```bash
arf run-artifacts-unpack \
  artifacts/releases/RUN_ID.tar.gz \
  artifacts/received/RUN_ID \
  --attestation artifacts/releases/RUN_ID.attestation.json \
  --public-key trust/release-2026.pub
```

If a sidecar or explicit digest is also supplied, it must agree with the archive. Keep private keys
outside source control and distribute trusted public-key fingerprints through a separate channel.

## Move archives through local or S3-compatible storage

Use the chunked transport when archives are too large for one in-memory object or transfers may be
interrupted:

```bash
arf run-artifacts-publish artifacts/releases/RUN_ID.tar.gz \
  --store-root artifacts/release-store \
  --attestation artifacts/releases/RUN_ID.attestation.json \
  --workers 4

arf run-artifacts-fetch RUN_ARCHIVE_ID artifacts/received/RUN_ID.tar.gz \
  --store-root artifacts/release-store \
  --public-key trust/release.pub \
  --require-attestation \
  --workers 4
```

Use the equivalent `--s3-bucket`, `--s3-prefix`, `--s3-endpoint-url`, and `--s3-region` options for
remote storage. Uploads and downloads resume at verified chunk boundaries. A release becomes
discoverable only when its immutable commit object exists, and a downloaded archive becomes visible
only after full transport, signature, archive-structure, artifact, and shard checks pass. See
[`object-storage.md`](object-storage.md) for the object protocol and operational requirements.

Use `run-artifacts-status` to distinguish staged data from a committed release. Aged staged
prefixes can be reclaimed only through the preview-first `run-artifacts-gc` workflow: review the
exact inventory digest, then execute with that digest plus an operator and reason. The same
decision key fences a concurrent publisher, while committed graphs remain permanently ineligible.
Shared stores can use `run-artifacts-inventory`, `run-artifacts-gc-plan`, and confirmed
`run-artifacts-gc-batch` execution to review aggregate reclaimable bytes and resume a partially
completed multi-prefix cleanup without expanding the approved target set.

To replicate a committed collection archive between stores, use `run-artifacts-mirror-plan` first.
Supply trusted public keys and `--require-attestation` when publisher identity is required. The
read-only plan inventories the exact source graph and destination state, then labels every canonical
object for copy or reuse. Execute the saved plan through `run-artifacts-mirror --execute` only with
its exact plan ID, operator, reason, and the same trust roots. Non-commit objects copy with bounded
parallelism; the destination commit is written last. An interrupted attempt resumes from verified
objects under the original plan, while any source or destination drift fails closed. Preserve the
canonical mirror record alongside the archive's other publication evidence.

For several collection archives, `run-artifacts-mirror-batch-plan` freezes an explicit archive
allowlist or every source release committed at that observation. The plan includes both store
inventories, each authenticated member plan, and aggregate transfer forecasts. Confirm the exact
batch ID through `run-artifacts-mirror-batch --execute`; release and object concurrency are
separately bounded with a maximum product of 64. A durable intent plus per-release records makes a
partial batch resumable, while `run-artifacts-mirror-batch-status` reports pending, partial,
destination-complete, completed, and invalid members without mutation. New unselected releases are
not added during continuation.

Bind `--max-releases` and `--max-copy-bytes` to every batch plan. Destination operators can recover
the complete operation set with `run-artifacts-mirror-batch-list` and inspect one intent by ID
without locating the saved plan. Preview-confirmed resolution is limited to wholly unstarted or
invalid intents; completed and resolved outcomes share one immutable decision fence, while query
sidecars can be reconstructed after interruption. Resolution does not retract collection archive
objects already copied by an in-flight worker.

If a persistence callback fails, the scheduler cancels and joins sibling rollouts before marking
the run failed. Already committed SQLite trajectories are eligible for exact slot reuse on the next
invocation. Shards remain independently recoverable with `arf shard-finalize` and
`arf shard-export`; no background rollout continues writing after the run transition.

Inspect live claim state for a stored plan without contacting services or changing claim files:

```bash
arf slot-claim-status \
  artifacts/collections/plans/PLAN_ID.json \
  artifacts/collections \
  --fail-on-expired
```

The report includes every planned slot and counts for `unclaimed`, `active`, `expired`, `completed`,
and `abandoned`. Active and expired entries include the latest renewal when present. Add
`--fail-on-active` when a monitoring job should return nonzero while any worker still owns a live
slot.

Collection holds a renewable fenced run lease throughout model calls and artifact construction.
The lease is released by the terminal SQLite transition. A hard-killed process stops renewing, so
`arf run-liveness` can identify the stale run without mutating its manifest.

For every missing planned slot, collection acquires an append-only slot claim before inference,
checks that the same claim is still current before callbacks persist the trajectory, and records a
completed or abandoned release. A competing active claim fails the run instead of silently doing
duplicate work. Re-running after the other worker commits reuses its verified trajectory; re-running
after claim expiry takes over with the next epoch. A background renewal loop stays active during
model, environment, and trajectory callback work. Renewal failure cancels that rollout and prevents
later callbacks from starting.

## Trainer handoff

Inspect reward variance and diversity before exporting:

```bash
arf filter-rollouts artifacts/collections/trajectories.db artifacts/filtered.jsonl \
  --run-id RUN_ID --policy-version POLICY_VERSION --expected-group-size 5 \
  --report artifacts/filter-report.json

arf trainer-batch-export artifacts/filtered.jsonl artifacts/trainer-store \
  --policy-version POLICY_VERSION --group-size 5 --source-run-id RUN_ID
```

The filter preserves accepted trajectory bytes. The trainer exporter independently revalidates
on-policy provenance, group membership, policy version, masks, and content digests.

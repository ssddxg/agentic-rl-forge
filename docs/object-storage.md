# Conditional object storage and trainer handoff

Distributed workers need create-if-absent semantics. A read followed by an unconditional write is
not sufficient: two writers can both observe a missing key and silently overwrite one another.

The `ConditionalBlobStore` boundary exposes five operations:

- `put_if_absent` creates a key or verifies that the existing bytes are identical;
- `get` returns exact bytes;
- `head` returns size, ETag, SHA-256 metadata, last-modified time, and user metadata;
- `list` returns deterministic keys under a prefix;
- `delete_if_match` deletes only when the complete observed object identity still matches.

## Local backend

`LocalBlobStore` uses flushed temporary files and exclusive hard links. It is suitable for local
development and POSIX shared filesystems.

```python
from agentic_rl_forge.storage import LocalBlobStore

store = LocalBlobStore("artifacts/blobs")
result = store.put_if_absent("runs/run-1/result.json", b"{}\n")
```

Keys must be relative POSIX paths and cannot contain `..` path components.

## S3-compatible backend

Install the optional client dependency:

```bash
pip install -e ".[object-store]"
```

Create a store with the default AWS client or an S3-compatible endpoint:

```python
from agentic_rl_forge.storage import create_s3_blob_store

store = create_s3_blob_store(
    bucket="agent-rl-artifacts",
    prefix="experiments/search-r1",
    endpoint_url=None,
    region_name="us-west-2",
)
```

The adapter sends `If-None-Match: *` with `PutObject`. S3 returns HTTP 412 when the key already
exists and HTTP 409 for a conflicting concurrent operation. The adapter compares existing bytes on
412, retries bounded 409 conflicts, and raises `BlobConflictError` if the existing object differs.
Identity-checked deletion sends the observed ETag through `DeleteObject` `IfMatch`, treats a missing
object as an idempotent result, and fails closed on a precondition conflict. Confirm that a custom
S3-compatible endpoint implements this request field before enabling built-in garbage collection.

Credentials, encryption, bucket policies, retention, replication, and lifecycle rules remain the
deployment's responsibility. No credentials are stored in project configuration or manifests.

## Slot-claim consistency

`SlotClaimCoordinator` can use any `ConditionalBlobStore`, but its constructor requires the caller
to acknowledge strong read-after-write consistency for both exact reads and prefix listings. The
acknowledgement is a deployment assertion, not a runtime probe. Verify the guarantees of the exact
service, region, gateway, replication mode, and S3-compatible implementation in use.

Conditional create alone is insufficient: claim acquisition lists prior epochs before creating the
next immutable epoch. An eventually consistent listing could select an already-used epoch or miss a
newer owner. Do not enable object-store-backed claims when those semantics are unavailable. Full
operational guidance is in [`slot-claims.md`](slot-claims.md).

## Trainer batch manifests

Only complete same-policy on-policy GRPO groups should cross the trainer boundary. Export a filtered
trajectory file into a local conditional store:

```bash
arf trainer-batch-export artifacts/filtered.jsonl artifacts/trainer-store \
  --policy-version policy-v1 --group-size 5 --source-run-id run-1
```

The generated `TrainerBatchManifest` records:

- exact policy and environment versions;
- group size, group count, and trajectory count;
- source run ID;
- payload key, byte size, and SHA-256;
- every trajectory ID, task ID, group ID, origin, reward, and canonical content digest.

Verify before consumption:

```bash
arf trainer-batch-verify artifacts/trainer-store \
  --manifest-key trainer/batches/BATCH_ID/manifest.json
```

Verification parses every verl trajectory record, reconstructs the embedded trajectory contract,
and checks record order, membership, policy version, provenance, task/group IDs, reward, and
canonical trajectory digest. A trainer should reject any batch whose verification is not valid.

`TrainerBatchExporter` accepts any `ConditionalBlobStore`, so the same manifest and validation
logic can target `S3ConditionalBlobStore` without changing batch semantics.

## Resumable run archive transport

`RunArtifactTransport` moves deterministic run archives through any `ConditionalBlobStore` without
loading the full archive into memory. The default chunk size is 8 MiB. Each chunk is bounded,
content-addressed, and created conditionally, so a retry verifies and reuses already transferred
chunks.

The canonical object layout is:

```text
run-releases/<archive_id>/
├── chunks/<index>-<sha256>.part
├── archive.sha256
├── attestation.json                  # optional
├── manifests/<transport_id>.json
└── commit.json                       # written last
```

The transport manifest binds the complete archive receipt, configured chunk size, ordered chunk
keys, exact chunk sizes and SHA-256 digests, checksum object, and optional signed attestation. The
commit is a small content-addressed record that binds the exact transport manifest.

Only `commit.json` makes a release visible. Chunks, checksum, attestation, and manifest are uploaded
and read back for byte verification first. If any write, read, or digest check fails, no commit is
created. `run-artifacts-list` reports only prefixes with a commit marker, so fully staged but
uncommitted data never appears as a release.

Publish and fetch through a local conditional store:

```bash
arf run-artifacts-publish artifacts/releases/RUN_ID.tar.gz \
  --store-root artifacts/release-store \
  --attestation artifacts/releases/RUN_ID.attestation.json \
  --workers 4

arf run-artifacts-list --store-root artifacts/release-store

arf run-artifacts-fetch RUN_ARCHIVE_ID artifacts/downloads/RUN_ID.tar.gz \
  --store-root artifacts/release-store \
  --public-key trust/release.pub \
  --require-attestation \
  --workers 4
```

For S3 or a compatible endpoint, replace `--store-root` with provider options:

```bash
arf run-artifacts-publish artifacts/releases/RUN_ID.tar.gz \
  --s3-bucket agent-rl-artifacts \
  --s3-prefix experiments/search-r1 \
  --s3-region us-west-2 \
  --attestation artifacts/releases/RUN_ID.attestation.json
```

Credentials come only from the normal boto3 credential chain. They are never written into release
metadata or CLI output. `--s3-endpoint-url` supports services that implement the required
conditional `PutObject` semantics; built-in GC additionally requires conditional `DeleteObject`.

`--workers` accepts 1 through 64. Only that many chunk operations can be in flight, archive bytes
are still read incrementally, and downloaded chunks are written in manifest order. Metadata and the
final decision object remain serial. A worker failure never creates a commit, and a failed download
keeps only its hidden verified prefix.

### Resume and visibility guarantees

An interrupted upload can leave immutable chunks or metadata under the release prefix, but without
`commit.json` consumers ignore them. Retrying recomputes the same chunk keys, verifies existing
bytes, completes missing objects, and creates the commit last. A failure after the commit write can
be retried safely because the complete committed graph is reloaded and verified.

Downloads use a hidden file named from the destination and archive ID. Before resuming, every
complete local chunk is rehashed against the transport manifest. A partial or corrupt next chunk is
truncated and downloaded again. The checksum and attestation sidecars are published only after the
complete archive passes strict archive inspection; the visible archive path is linked last and acts
as the local commit marker.

Remote post-write verification reads each object back. This deliberately adds transfer cost in
exchange for end-to-end evidence that the configured service returned the exact bytes.

### Dry-run-first release mirroring

`RunArtifactMirror` copies one already committed release directly between any two
`ConditionalBlobStore` implementations. Source and destination can independently be local or
S3-compatible stores. The source is never inferred from object names alone: planning first loads
and validates the complete committed graph and, when public keys are supplied, authenticates the
archive attestation against those external trust anchors.

Create a read-only plan:

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

Local endpoints use `--source-store-root` and `--destination-store-root`. Each side has independent
`--*-s3-bucket`, `--*-s3-prefix`, `--*-s3-endpoint-url`, and `--*-s3-region` options, so local-to-S3,
S3-to-local, and S3-to-S3 transfers use the same protocol. Credentials continue to come from the
normal boto3 credential chain.

The canonical plan records:

- the exact source commit, transport manifest, ordered object graph, and trust result;
- complete source identities including size, SHA-256, ETag, and last-modified time;
- whether each destination object must be copied or can be reused;
- the complete identity of every unrelated object already present in that release prefix;
- exact copy/reuse counts and bytes; and
- a content-derived plan ID that excludes only the observation timestamp.

Validate a saved plan locally without opening either provider or changing storage:

```bash
arf run-artifacts-mirror artifacts/mirrors/RUN_ARCHIVE_ID.plan.json
```

After review, execute only with the exact plan ID and durable operator context:

```bash
arf run-artifacts-mirror artifacts/mirrors/RUN_ARCHIVE_ID.plan.json \
  --source-s3-bucket primary-releases \
  --source-s3-prefix search-r1 \
  --destination-s3-bucket replica-releases \
  --destination-s3-prefix search-r1 \
  --public-key trust/release.pub \
  --execute \
  --confirm-plan-id RUN_MIRROR_PLAN_ID \
  --operator release@example.com \
  --reason "approved cross-region release replication" \
  --workers 4 \
  --output artifacts/mirrors/RUN_ARCHIVE_ID.record.json
```

Execution repeats source trust verification and rechecks both inventories before copying. At most
`--workers` non-commit objects are transferred concurrently. Every source read is checked against
the planned storage identity and content digest; every conditional destination write is read back
and verified. The destination commit is copied serially only after all chunks and metadata pass a
second snapshot check, then the complete destination release is independently loaded and
authenticated.

An interruption before the commit leaves no visible destination release. Repeating the original
plan recognizes exact objects created by the earlier attempt as progress, reuses them, and still
writes the commit last. A completed operation writes canonical immutable evidence at
`run-release-mirrors/records/<mirror_id>.json`; repeating the same plan, operator, and reason returns
that record after revalidating both stores. An already mirrored release produces a reuse-only plan.

Mirroring fails closed if a source object changes, a planned destination object changes or
disappears, an unrelated destination object changes, a new unplanned key appears, a destination
key contains different bytes, or the destination commit conflicts. A plan intentionally binds
unrelated keys rather than deleting or ignoring them. Mirroring never performs garbage collection
and does not weaken the destination's normal commit visibility rules.

### Store-wide mirror batches

Use a batch plan when the reviewed unit is a set of releases rather than one archive. Selection is
always explicit: repeat `--archive-id` for an allowlist, or use `--all-committed` to freeze the set
of releases committed at planning time. The two modes cannot be combined, and an empty selection
is rejected.

```bash
arf run-artifacts-mirror-batch-plan \
  --all-committed \
  --source-s3-bucket primary-releases \
  --source-s3-prefix search-r1 \
  --destination-s3-bucket replica-releases \
  --destination-s3-prefix search-r1 \
  --public-key trust/release.pub \
  --require-attestation \
  --max-releases 256 \
  --max-copy-bytes 1099511627776 \
  --workers 4 \
  --output artifacts/mirrors/store.plan.json
```

Planning captures complete source and destination lifecycle inventories at one observation time,
then creates a normal trusted mirror plan for every selected archive. The batch plan binds both
inventory state digests, the selection mode and sorted archive IDs, all member plan IDs, aggregate
object counts, exact copy/reuse byte forecasts, and the reviewed release-count and copy-byte safety
caps. Planning fails before execution if either cap is exceeded. A final inventory pass rejects
concurrent changes while the plan is being built. Observation time and planning worker count do not
affect the content-derived plan ID.

Validate the saved contract without provider access:

```bash
arf run-artifacts-mirror-batch artifacts/mirrors/store.plan.json
```

Execute the frozen release set only after review:

```bash
arf run-artifacts-mirror-batch artifacts/mirrors/store.plan.json \
  --source-s3-bucket primary-releases \
  --source-s3-prefix search-r1 \
  --destination-s3-bucket replica-releases \
  --destination-s3-prefix search-r1 \
  --public-key trust/release.pub \
  --execute \
  --confirm-plan-id RUN_MIRROR_BATCH_PLAN_ID \
  --operator release@example.com \
  --reason "approved disaster-recovery synchronization" \
  --release-workers 4 \
  --object-workers 2 \
  --output artifacts/mirrors/store.record.json
```

An immutable intent is written at
`run-release-mirrors/batches/<batch_id>/intent.json` before any member starts. Releases run with
bounded cross-release concurrency while each release retains its own object-level bound. The
product of `--release-workers` and `--object-workers` cannot exceed 64. Every member still uses its
own trust result, exact identities, commit-last rule, and immutable evidence record.

If one member fails, successful members remain committed and evidenced; incomplete members retain
only exact resumable object progress. Repeating the original plan, operator, reason, and trust set
reuses completed records and continues unfinished members. Once every selected release has valid
member evidence, execution conditionally creates
`run-release-mirrors/batches/<batch_id>/decision.json` with a completed outcome, then writes the
recoverable `record.json` sidecar. A retry reconstructs a missing sidecar from the authoritative
decision.

Inspect a started batch without mutation:

```bash
arf run-artifacts-mirror-batch-status artifacts/mirrors/store.plan.json \
  --source-s3-bucket primary-releases \
  --source-s3-prefix search-r1 \
  --destination-s3-bucket replica-releases \
  --destination-s3-prefix search-r1 \
  --public-key trust/release.pub \
  --operator release@example.com \
  --reason "approved disaster-recovery synchronization" \
  --fail-on-invalid
```

Member states are `pending`, `partial`, `destination_complete`, `completed`, and `invalid`.
`destination_complete` identifies a crash after the destination commit but before member evidence;
the original execute command safely finishes that evidence. `--fail-on-incomplete` is suitable for
automation that should remain nonzero until final evidence exists. The status includes verified
present and remaining bytes plus a stable state digest; aggregate progress is withheld if any
member is invalid.

The selected prefixes are the authorization boundary. New releases or staging prefixes outside
the frozen allowlist are never copied or modified. Any source or destination drift inside a
selected prefix fails that member closed, including a pre-existing commit without its complete
canonical graph.

### Mirror operations ledger and resolution

List every destination-side batch without a saved local plan:

```bash
arf run-artifacts-mirror-batch-list \
  --destination-s3-bucket replica-releases \
  --destination-s3-prefix search-r1 \
  --fail-on-unclassified \
  --output artifacts/mirrors/ledger.json
```

The canonical ledger sorts batches by ID, classifies `intent_only`, `member_evidence`, `complete`,
and `resolved` evidence, counts durable member records, reports unknown keys, and provides a stable
state digest. Inspecting a batch by ID revalidates its embedded plan against the source and
destination and reports `pending`, `active`, `blocked`, `complete`, `degraded`, or `resolved`
health:

```bash
arf run-artifacts-mirror-batch-inspect RUN_MIRROR_BATCH_ID \
  --source-s3-bucket primary-releases \
  --destination-s3-bucket replica-releases \
  --public-key trust/release.pub \
  --output artifacts/mirrors/RUN_MIRROR_BATCH_ID.inspection.json
```

`run-artifacts-mirror-batch-resolve` is inspection-only unless `--execute` is supplied. Resolution
requires the exact status digest plus resolver and reason. `cancelled` applies to a reviewed but
entirely unstarted intent; `superseded` also accepts a blocked invalid intent and requires a
different `--replacement-plan-id`:

```bash
arf run-artifacts-mirror-batch-resolve RUN_MIRROR_BATCH_ID \
  --source-s3-bucket primary-releases \
  --destination-s3-bucket replica-releases \
  --public-key trust/release.pub \
  --execute --kind superseded \
  --confirm-status-digest STATUS_DIGEST \
  --replacement-plan-id RUN_MIRROR_BATCH_PLAN_ID \
  --resolver release@example.com \
  --reason "replace a permanently drifted mirror operation"
```

Completion and resolution conditionally claim the same immutable decision key, so both outcomes
cannot win. The resolution embeds the confirmed state counts and exact invalid-member details.
Active partial, destination-complete, or completed batches are ineligible. Resolution is not a
rollback: quiesce executors first when possible, because in-flight release copies may still leave
valid objects or member records even though they cannot publish a completed batch decision.

### Status and preview-confirmed garbage collection

Inspect any release prefix without changing it:

```bash
arf run-artifacts-status RUN_ARCHIVE_ID \
  --store-root artifacts/release-store
```

The state is `absent`, `staged`, `committed`, `gc_in_progress`, `garbage_collected`, or `invalid`.
Committed status requires the complete manifest, checksum, optional attestation, and chunk graph to
validate. A malformed decision or completion record is reported as invalid rather than treated as
deletable staging data.

Garbage collection is preview-only by default:

```bash
arf run-artifacts-gc RUN_ARCHIVE_ID \
  --store-root artifacts/release-store \
  --min-age-seconds 86400 \
  --output artifacts/gc/RUN_ARCHIVE_ID.preview.json
```

An eligible preview inventories every staged object with its key, size, SHA-256, ETag, and
last-modified time. Its stable `state_digest` binds that exact inventory, the retention threshold,
and the computed eligibility time; observation time is deliberately excluded. No object is removed
during preview.

After review, execute only with the exact digest and durable operator context:

```bash
arf run-artifacts-gc RUN_ARCHIVE_ID \
  --store-root artifacts/release-store \
  --min-age-seconds 86400 \
  --execute \
  --confirm-state-digest STATE_DIGEST \
  --operator release@example.com \
  --reason "publication host was retired after upload interruption" \
  --output artifacts/gc/RUN_ARCHIVE_ID.record.json
```

The collector first competes for the same immutable `commit.json` decision key used by publishers.
A valid release commit and a GC tombstone therefore cannot both win. Once a tombstone wins, a late
publisher cannot make the prefix visible. Before deletion, the collector rechecks that current
objects are an exact subset of the preview with unchanged identities; new or mutated objects abort
the operation. Every delete is conditional, and committed graphs are never eligible regardless of
age.

The tombstone remains at `run-releases/<archive_id>/commit.json`. Successful completion writes an
immutable evidence record at `run-release-gc/records/<gc_id>.json`, outside the cleaned prefix. If a
delete fails midway, repeating the same digest, operator, reason, and retention threshold resumes
under the existing tombstone. A different retry command is rejected. Provider lifecycle rules may
still be useful as a final backstop, but must never delete committed prefixes or bypass this
decision protocol.

### Store-wide inventory and reviewed retention plans

For a shared store, create a stable lifecycle inventory before choosing individual prefixes:

```bash
arf run-artifacts-inventory \
  --s3-bucket agent-rl-artifacts \
  --s3-prefix experiments/search-r1 \
  --output artifacts/gc/store-inventory.json

arf run-artifacts-gc-plan \
  --s3-bucket agent-rl-artifacts \
  --s3-prefix experiments/search-r1 \
  --min-age-seconds 86400 \
  --output artifacts/gc/retention-plan.json
```

The inventory scans every `run-releases/<archive_id>/` prefix, validates its lifecycle status,
summarizes object and byte counts by state, and produces a state digest that excludes observation
time. Unclassified keys and invalid release states prevent plan creation. The plan includes the
complete reviewed inventory, only eligible staged previews, aggregate reclaimable bytes, and a
content-derived plan ID. Committed, tombstoned, recent, absent, and invalid releases cannot enter
the candidate set.

Validate a saved plan without a storage provider or mutation:

```bash
arf run-artifacts-gc-batch artifacts/gc/retention-plan.json
```

Execute only after reviewing the exact candidate list and aggregate bytes:

```bash
arf run-artifacts-gc-batch artifacts/gc/retention-plan.json \
  --s3-bucket agent-rl-artifacts \
  --s3-prefix experiments/search-r1 \
  --execute \
  --confirm-plan-id RUN_GC_PLAN_ID \
  --operator release@example.com \
  --reason "quarterly cleanup of expired interrupted publications" \
  --workers 4 \
  --output artifacts/gc/batch-record.json
```

Batch execution writes an immutable intent before starting candidates and runs at most `--workers`
prefixes concurrently. Every member still uses its own preview digest, tombstone, identity-checked
deletes, and completion record. A transient failure may leave some members completed and others in
progress; repeating the same plan, operator, and reason reuses those records and resumes only the
unfinished members. The final batch record commits only after every reviewed candidate has durable
member evidence. Releases created after planning are not targets and cannot be deleted by the
batch.

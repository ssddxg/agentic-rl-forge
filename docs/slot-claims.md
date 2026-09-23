# Planned-slot claims

Deterministic rollout plans make completed work reusable, but storage conflict checks alone do not
stop two hosts from starting the same missing model call. `SlotClaimCoordinator` adds a small,
provider-neutral coordination layer over `ConditionalBlobStore` for planned rollouts.

## Lifecycle

For each missing slot, the scheduler performs five steps:

1. acquire an immutable claim epoch before model inference;
2. append immutable renewal heartbeats while model, environment, and callbacks are running;
3. execute the agent loop with the claim ID in trajectory provenance;
4. verify that the claim is still current immediately before and after persistence callbacks;
5. append a `completed` release after callbacks succeed, or an `abandoned` release on failure.

An active claim owned by another worker raises `SlotClaimConflictError`. Collection fails and joins
its sibling jobs; it does not wait indefinitely or continue duplicate work. A later invocation can
reuse the other worker's committed trajectory or acquire a new epoch after release or expiry.

## Immutable layout

```text
slot-claims/
└── plans/<plan_id>/slots/<slot_id>/
    ├── claims/
    │   ├── 00000000000000000001.json
    │   └── 00000000000000000002.json
    ├── renewals/
    │   └── 00000000000000000001/
    │       ├── 00000000000000000001.json
    │       └── 00000000000000000002.json
    └── releases/
        ├── 00000000000000000001.json
        └── 00000000000000000002.json
```

Claim and release objects are created conditionally and never overwritten. The epoch is a fencing
generation: after takeover, `assert_current` rejects an older worker even if that worker resumes.
Repeated acquisition by the same owner while its claim is active is idempotent. Repeating an
identical release is also idempotent; changing its outcome is rejected.

Renewals are also conditionally created and never overwrite the original claim. Their index must be
contiguous and each effective expiry must strictly extend the prior claim or renewal. Concurrent
writers for the same index converge on one immutable winner. Acquisition, fencing, and status use
the latest valid renewal expiry.

## Storage contract

The backing store must provide all of the following:

- atomic create-if-absent for one key;
- strong read-after-write behavior for `get` and `head`;
- strong read-after-write behavior for `list` under the slot prefix;
- one shared namespace visible to every participating worker.

Construction requires
`ClaimStoreConsistency.STRONG_READ_AFTER_WRITE_AND_LIST` as an explicit acknowledgement. This does
not test the provider. Confirm the behavior of the exact filesystem, object service, gateway, and
replication configuration before deployment. Do not use claims over an eventually consistent list.

`LocalBlobStore` satisfies the contract on a POSIX filesystem with atomic hard links. The
`S3ConditionalBlobStore` adapter supplies conditional object creation, but the caller remains
responsible for confirming the read and listing guarantees of the specific S3-compatible service.

## Expiry, renewal, and clocks

Each claim and renewal grants a fixed ownership window. The scheduler renews before half that window
elapses. `slot_claim_renewal_interval_s` must not exceed half of `slot_claim_ttl_s`; leave enough
margin for scheduler delay and conditional-store latency. Workers should use synchronized clocks.

Renewal stops before the completed release is appended. A renewal write failure cancels the active
rollout, joins it, and records an abandoned release when possible. Once the effective expiry passes,
the old owner cannot renew, and a higher epoch can take over.

The scheduler checks ownership before and after callbacks, but the claim store and trajectory stores
do not form a distributed transaction. Renewal failure or takeover during a callback can still leave
a partial persistence attempt. Immutable trajectory IDs, conditional shards, and run-lease-first
callback ordering remain the final conflict and stale-writer defenses. Claims reduce duplicate
inference; they do not promise exactly-once execution.

## Search-R1 defaults

The checked-in collection configuration enables claims, grants a 900-second ownership window, and
renews every 300 seconds:

```yaml
enable_slot_claims: true
slot_claim_ttl_s: 900
slot_claim_renewal_interval_s: 300
```

Without a custom coordinator, the pipeline writes claims to
`<output>/coordination/slot-claims`. Multiple local processes or hosts must therefore mount the same
collection root with the required POSIX semantics. Python callers can pass a custom
`slot_claim_coordinator` backed by another conditional store.

Disable claims only when duplicate in-flight model calls are acceptable or when the available
storage cannot meet the consistency contract. Completed-trajectory reuse, deterministic planning,
and persistence conflict detection remain active independently.

## Read-only status

Inspect all current slot states for one immutable plan:

```bash
arf slot-claim-status \
  artifacts/collections/plans/PLAN_ID.json \
  artifacts/collections
```

The default claim prefix is `coordination/slot-claims`, matching the Search-R1 pipeline. Use
`--claim-prefix` for a custom namespace. The command reports the latest claim, renewal, and release
records, owner, epoch, effective expiry, and state counts. It does not acquire, release, renew, or
otherwise write coordination state.

Monitoring options control only the exit code:

- `--fail-on-active` exits 1 when at least one current claim is active;
- `--fail-on-expired` exits 1 when at least one unreleased claim has expired.

Malformed object identities, claim or renewal indexes that disagree with their immutable filename,
non-monotonic renewal histories, and release records that disagree with their claim are rejected
instead of being summarized.

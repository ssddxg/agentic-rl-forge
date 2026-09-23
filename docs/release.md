# Reproducibility and release gates

## Dataset fingerprints

Build one collection manifest from all experiment splits:

```bash
arf dataset-manifest artifacts/dataset-manifest.json --name search-r1-run \
  --split-file train=artifacts/train.jsonl \
  --split-file validation=artifacts/validation.jsonl \
  --id-field extra_info.task_id \
  --root "$PWD"
```

For every JSONL file, the manifest records its relative path, byte size, SHA-256, record count,
unique ID count, duplicate count, and sorted-ID digest. The collection records all sorted IDs and
rejects any ID owned by more than one split. Rebuilding from identical bytes and paths produces the
same manifest ID.

The manifest fingerprints raw bytes intentionally. Reformatting JSON or changing record order
changes the content digest even if semantic records appear equivalent.

## Ed25519 signatures

Install the optional signing dependency:

```bash
pip install -e ".[signing]"
```

Generate a key pair and sign a collection manifest:

```bash
arf manifest-keygen secrets/dataset.key artifacts/dataset.pub
arf manifest-sign artifacts/dataset-manifest.json artifacts/dataset-manifest.signed.json \
  --private-key secrets/dataset.key
arf manifest-verify artifacts/dataset-manifest.signed.json
```

The private key is written with mode `0600`. Do not commit it. Commit or publish only the public key
and signed manifest. Verification checks both the Ed25519 signature and the canonical payload
SHA-256.

The same key format signs published run archives. After packing a run, create and verify an exact
receipt attestation:

```bash
arf run-artifacts-sign artifacts/releases/RUN_ID.tar.gz \
  artifacts/releases/RUN_ID.attestation.json \
  --private-key secrets/release.key

arf run-artifacts-signature-verify artifacts/releases/RUN_ID.tar.gz \
  artifacts/releases/RUN_ID.attestation.json \
  --public-key artifacts/release.pub
```

The public key supplied to verification is the trust anchor. The public key embedded in the signed
file is checked for identity consistency but is not trusted by itself. During rotation, repeat
`--public-key` for each currently trusted release key.

The same key format can attest a selected experiment's portable metadata archive:

```bash
arf experiment-promotion-pack \
  artifacts/experiment-promotions/search-r1-candidate \
  artifacts/releases/search-r1-candidate.promotion.tar.gz

arf experiment-promotion-sign \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --private-key secrets/release.key

arf experiment-promotion-signature-verify \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --public-key artifacts/release.pub
```

This signature authenticates the publisher and exact archive receipt. It does not convert the
operator or approver strings inside `decision.json` into cryptographically verified identities.

## Release audit

Run the complete local gate immediately before the release commit:

```bash
arf release-audit --project . --run-checks --build-wheel \
  --output artifacts/release-audit.json
```

The audit fails when any error-level finding remains:

| Gate | Evidence |
| --- | --- |
| Repository files | README, changelog, license, security policy, contribution guide, and CI workflows |
| Package metadata | Name, version, description, Python constraint, readme, and license |
| Documentation | No release placeholders and all local Markdown links resolve |
| Claims | Numeric README performance claims are surfaced for evidence review |
| Security | Common private-key and provider-token patterns are absent |
| Provenance | No generation-trace phrases are present in repository text |
| Git | Initial commit exists, worktree is clean, and origin is reported |
| Quality | Ruff lint and formatting, strict Mypy, Pytest, and recipe syntax pass |
| Packaging | A wheel is built successfully from the current source tree |

Warnings do not fail the report but must be reviewed. When preparing a new repository for its
first public push, create one clean root commit, configure `origin`, rerun the strict audit, and
preserve the resulting JSON report with the release artifacts. This repository already uses its
canonical GitHub URLs; an absent-remote warning only applies to source archives or local checkouts
that have not been connected to a remote.

For published collection examples, verify and pack each run into an independent release unit:

```bash
arf run-artifacts-verify \
  artifacts/collections/runs/RUN_ID/artifact-manifest.json

arf run-artifacts-pack \
  artifacts/collections/runs/RUN_ID/artifact-manifest.json \
  artifacts/releases/RUN_ID.tar.gz
```

Publish `RUN_ID.tar.gz`, `RUN_ID.tar.gz.sha256`, and the signed attestation when release
authenticity is required. Receivers can use `arf run-artifacts-unpack` with `--attestation` and a
trusted `--public-key` to authenticate the publisher, reject unsafe archive structures, and
recursively verify the extracted run before it becomes visible at the destination path.

For large release artifacts or unreliable links, publish the same archive through the resumable
conditional-store protocol:

```bash
arf run-artifacts-publish artifacts/releases/RUN_ID.tar.gz \
  --s3-bucket agent-rl-artifacts \
  --s3-prefix releases/v0.1 \
  --attestation artifacts/releases/RUN_ID.attestation.json \
  --workers 4
```

Record the returned archive ID and commit ID in release notes. Receivers use
`run-artifacts-fetch`, trusted public keys, and `--require-attestation`. A release is not
discoverable until its final commit object exists, so retrying an interrupted publication cannot
expose a partially uploaded archive.

For a reviewed replica or provider migration, create a `run-artifacts-mirror-plan` using an
independently distributed release public key. Archive the canonical plan before execution. It binds
the authenticated source commit and transport graph, exact source and destination identities,
copy/reuse actions, unrelated destination objects, and byte totals. Run `run-artifacts-mirror`
without `--execute` to validate the saved contract offline, then execute with the exact plan ID,
operator, reason, the same trust keys, and an appropriate worker bound. Keep the resulting
`RunArtifactMirrorRecord` with release-operations evidence. Interrupted execution must resume from
the original plan; do not silently replace it with a broader or newer snapshot. The destination
commit is always copied last and evidence is created only after complete authenticated destination
verification.

For a multi-release migration, preserve the output of `run-artifacts-mirror-batch-plan` as the
review artifact. Choose either an explicit archive allowlist or `--all-committed`; the latter means
all releases committed in the captured source inventory, not releases created later. Confirm the
exact batch plan ID and keep its intent, per-release mirror records, final batch record, and status
snapshots together. An interrupted batch must resume with the original operator, reason, trust set,
and plan. Successful members remain valid, selected-prefix drift fails closed, and unselected new
releases are never silently added to the operation.

Set and review `--max-releases` and `--max-copy-bytes`; both limits are part of the batch identity.
Use `run-artifacts-mirror-batch-list` to preserve a canonical destination operations ledger and
`run-artifacts-mirror-batch-inspect` for lookup by batch ID. If an intent must be abandoned, preview
`run-artifacts-mirror-batch-resolve`, quiesce executors, and confirm the exact status digest. Only
entirely unstarted or invalid batches can be cancelled or superseded. Completion and resolution
compete for the same immutable decision key, so a resolved operation cannot later publish final
batch completion even if an in-flight release copy leaves member-level progress.

Before applying provider lifecycle deletion to an interrupted prefix, inspect it with
`run-artifacts-status` and run `run-artifacts-gc` in preview mode. Preserve the preview and supply
its exact state digest, retention threshold, operator, and reason to `--execute`. Cleanup competes
with publication on the final decision key, refuses committed graphs, detects post-preview changes,
and writes a durable completion record outside the release prefix. Repeating the exact execute
command resumes an interrupted cleanup without weakening the original confirmation.

For periodic maintenance across a shared prefix, archive the output of `run-artifacts-inventory`,
review a canonical `run-artifacts-gc-plan`, and execute it only through
`run-artifacts-gc-batch --execute --confirm-plan-id`. Keep the plan, batch intent, member GC
records, and final batch record together as release-operations evidence.

The top-level run artifact manifest is independent of Git release auditing: it verifies experiment
outputs, while `release-audit` verifies the source repository and package.

## Publishing the Python package

The `Release` GitHub Actions workflow builds both the wheel and source distribution, requires the
tag to equal `v` plus the version in `pyproject.toml`, runs strict package metadata checks, publishes
to PyPI through trusted publishing, and creates a GitHub release containing the exact distributions.
No long-lived PyPI API token is stored in the repository.

Before pushing a release tag:

Dependency audits use `pip-audit --skip-editable`: the unpublished checkout itself is excluded from
PyPI vulnerability lookup, while its installed non-editable third-party dependencies remain in
scope.

1. Move the intended changes under a dated heading in `CHANGELOG.md` and set the same version in
   `pyproject.toml` and `agentic_rl_forge.__version__`.
2. Run `make release-check` and the strict `arf release-audit` command above from a clean commit.
3. Configure the public repository as `origin`, add its source/issue URLs to `[project.urls]`, and
   configure the PyPI project to trust the repository's `release.yml` workflow in environment
   `pypi`.
4. Create and push an annotated `vMAJOR.MINOR.PATCH` tag. Protect the `pypi` environment when a
   manual release approval is required.

PyPI ownership and trusted-publisher configuration are external state. Confirm both before the
repository becomes public, and never push a release tag until the package name is controlled by the
project maintainers.

For matrix-driven experiments, archive the canonical `experiment-plan` output, its state directory,
and the deterministic `experiment-run` report. CI should use separate incomplete, failed, and
regression gates so a valid but worse result is not confused with infrastructure failure. A changed
artifact beneath an immutable success record must block release; create and review a new experiment
plan instead of deleting evidence or reusing the old plan ID.

For releases selected from multiple plans, also archive the canonical `experiment-index` and
`experiment-analyze` JSON outputs. They bind the exact report IDs, parameter/metric table, objective
directions and weights, eligibility policy, baseline, ranking, improvements, and Pareto front.
Markdown, CSV, and HTML are presentation derivatives; retain the two canonical JSON contracts as
the verification boundary. A ranking score is not a substitute for the experiment's paired
confidence report or declared regression gates.

For a selected release candidate, retain the promotion directory as one evidence unit. Its
`decision.json` is authoritative and embeds the exact preview, technical checks, policy, and
reproducibility manifest. `record.json`, `manifest.json`, `model-card.md`, `plan.json`,
`report.json`, `index.json`, and `analysis.json` are deterministic sidecars that an exact retry can
restore. The manifest references heavyweight artifacts by scoped path/URI, size, and SHA-256 rather
than copying them; archive or publish those payloads through their native verified transport.

Approval identities in a promotion record are durable assertions, not cryptographic signatures.
Use `experiment-promotion-pack` to emit deterministic metadata plus its checksum, and optionally
`experiment-promotion-sign` to authenticate the publisher. Receivers should use
`experiment-promotion-inspect` in CI and `experiment-promotion-unpack` with independently pinned
public keys before publishing the extracted directory. If release governance requires signed human
approval or quorum, countersign the decision in an external identity-aware workflow before treating
it as authorization; the archive publisher signature is intentionally a different control.

On each receiver, preserve the canonical `experiment-promotion-acquire` plan before confirming it.
The plan binds the trusted archive receipt, receiver roots, destination, per-artifact byte and native
contract findings, checkpoint payload/parent/dataset ancestry, unresolved remote references, and the
maximum materialized byte budget. Preserve the final acquisition record with the plan. Its
content-addressed directory can resume an exact partial copy, while `record.json` is the commit
boundary. A remote URI is evidence, not permission to download. Use
`experiment-promotion-fetch-remote` to create a separate reviewed authorization plan with exact
HTTPS authority or S3 bucket allowlists, byte limits, source validators, and cache identity. Execute
that plan with exact ID confirmation, retain its immutable per-object receipts and final record, and
pass the record to acquisition with `--remote-record`. Acquisition itself remains network-free.

After receiver verification, use `experiment-promotion-lifecycle` to preview and confirm the exact
`candidate`, `staging`, and `production` transitions. Preserve each preview and resulting event with
release evidence. Point deployment environments through `experiment-promotion-alias`; its
generation stream provides compare-and-swap updates, and rollback is valid only to a promotion that
actually appeared earlier in the same environment. Before and after every change, archive
`experiment-promotion-registry-status --fail-on-issues`. Do not retire a promotion until all active
environment aliases have moved away from it. Publisher trust authenticates archive bytes; the
operator and authorizer quorum remain a separate governance decision.

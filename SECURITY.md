# Security Policy

Agent tools may execute code, mutate external systems, access authenticated browser sessions,
or expose retrieved private data. Treat every tool response and model-generated argument as
untrusted input.

## Reporting

Do not open a public issue for a vulnerability that enables unauthorized code execution,
credential exposure, cross-session state access, sandbox escape, or destructive external tool
calls. Before making the repository public, enable GitHub private vulnerability reporting. Report
these issues with the repository's **Security → Report a vulnerability** form so the discussion and
fix remain private.

## Deployment guidance

- Isolate browser and code-execution environments from rollout and training workers.
- Use per-session credentials and revoke them when the session closes.
- Deny network access by default in code sandboxes.
- Validate tool arguments against JSON Schema before execution.
- Require explicit policy for tools with write or external side effects.
- Remove secrets and private verifier state from model-visible observations.
- Apply request limits, timeouts, output limits, and audit logging at tool gateways.
- Do not expose the development retrieval server directly to untrusted networks.
- Receive published runs through `run-artifacts-unpack`; do not pass untrusted archives to a generic
  tar extractor. Keep the expected SHA-256 in a separate trusted channel when authenticity matters.
- Pin release public keys outside the archive channel and pass them explicitly when verifying signed
  run attestations. An embedded public key proves signature consistency, not publisher identity.
- Treat only archive transport prefixes with a validated final commit object as releases; never
  infer publication from chunk presence. Reclaim old staging prefixes through preview-confirmed
  `run-artifacts-gc`, preserve its operator evidence, and require conditional-delete support from
  the configured object service. Never aim a generic lifecycle rule at committed release prefixes.
- For store-wide cleanup, execute only a canonical reviewed `run-artifacts-gc-plan` with its exact
  plan ID. Preserve batch intent and completion records; do not regenerate a broader plan during an
  interrupted run or treat newly discovered prefixes as implicitly approved targets.
- Mirror releases only from a completely validated committed graph. Require independently pinned
  public keys for authenticated releases, review the canonical mirror plan, and confirm its exact
  plan ID. A signature key embedded beside the source objects is not a trust anchor.
- Resume an interrupted mirror with the original plan, operator, reason, and trust set. Do not
  accept source drift, destination drift, unplanned destination keys, or conflicting bytes as
  progress. Preserve the immutable destination mirror record as operational evidence.
- Treat a mirror batch's sorted release selection as its authorization boundary. Review aggregate
  bytes and every member trust result, confirm the exact batch plan ID, and never regenerate a plan
  merely to absorb selected-prefix drift or newly published releases.
- Preserve mirror batch intent, member records, final evidence, and status digests. An invalid
  member must block completion; a destination-complete member still requires the original execute
  command to publish its missing evidence.
- Bind explicit maximum release and copy-byte budgets into every mirror batch plan. Audit the
  destination operations ledger for unclassified keys and use by-ID inspection before taking an
  administrative action.
- Quiesce executors before cancelling or superseding an eligible mirror batch, confirm the exact
  inspected status digest, and preserve the immutable resolution. Completion and resolution share
  one conditional-create decision key, but resolution is not rollback: release objects and member
  evidence already written by in-flight workers may remain valid.
- Treat experiment matrix files as executable specifications. Review the rendered canonical plan
  and confirm its exact ID before execution. Commands bypass a shell but still inherit process
  credentials and environment variables; never place secrets in parameters, configs, commands,
  artifact summaries, or captured logs.
- Keep experiment artifacts below the selected execution root and preserve immutable stage records.
  A missing or changed successful output is evidence drift, not permission to rerun and overwrite
  history. Create a new plan identity for intentional changes.
- Treat the experiment state directory as an evidence namespace. Cross-plan indexing reports
  unrecognized top-level entries and invalid canonical plans instead of silently ignoring them.
  Metric extraction rechecks artifact digests and the project-root boundary before reading values.
- Static experiment dashboards escape parameter and metric text and contain no scripts or remote
  assets. CSV export prefixes formula-like string parameters before spreadsheet use. These files can
  still disclose parameter values, artifact identities, and operational metadata; inspect the
  canonical index before publishing and never store credentials in experiment inputs.
- Experiment promotion previews are read-only and must be recomputed at execution. Keep promotion
  output outside the experiment state directory, confirm the exact preview ID, and use independent
  approver identities unless a stronger external approval system explicitly replaces that control.
- Promotion rejects local checkpoint paths outside the project root and secret-bearing remote URIs.
  Model-card text is escaped, but promotion records deliberately preserve operator, approver, and
  rationale text; never include credentials, private incident details, or access tokens.
- Approval identities are not signatures. A promotion decision proves what this process checked and
  recorded, not that the named people cryptographically approved it.
- Transfer promotion evidence with `experiment-promotion-pack` and receive it only through
  `experiment-promotion-inspect` or `experiment-promotion-unpack`; do not use a generic tar extractor
  for untrusted input. Keep an expected SHA-256 or trusted public key outside the archive channel.
- A promotion archive attestation authenticates the publisher of exact metadata bytes, not the
  approvers named inside the decision. Pin publisher keys independently, repeat `--public-key` only
  for intentional rotation, and treat an embedded signing key as untrusted until it matches that set.
- Promotion acquisition never fetches remote URIs. Resolve project and experiment-state references
  only below explicit receiver roots, keep the destination separate from every source cache, review
  every policy relaxation, and treat unresolved remote evidence as incomplete.
- Fetch remote promotion artifacts only with `experiment-promotion-fetch-remote`. Pin exact HTTPS
  authorities or S3 buckets, refuse redirects, review provider identity and byte limits, and confirm
  the exact plan ID. Prefer version IDs or strong validators; relaxing validator requirements still
  detects final digest mismatch but weakens protection against a source changing between ranges.
- Keep remote fetch cache records and chunks immutable. Unknown files, symbolic links, conflicting
  chunks, changed source metadata, unexpected range responses, and digest mismatches are security
  failures, not resumable progress. Never place credentials in a signed URI; HTTPS public objects
  and the normal S3 SDK credential chain are the supported authentication paths.
- Treat an acquisition plan ID as a write authorization for one exact destination and byte budget.
  Unknown files, symbolic links, conflicting bytes, changed sources, invalid native contracts,
  missing checkpoint payloads or parents, and broken dataset lineage must fail closed. Preserve the
  final acquisition record; do not create it manually or use its absence as permission to overwrite
  partial receiver data.
- Treat the promotion registry root as an append-only evidence namespace. Do not insert notes,
  mutable aliases, or out-of-band files. Require a clean `experiment-promotion-registry-status`
  before mutation and preserve exact preview IDs for lifecycle and environment changes.
- Publisher trust and deployment authorization are separate controls. A valid archive signature is
  not permission to stage, deploy, retire, or roll back a promotion. Require independent authorizers
  and use an external identity-aware approval system when strings are insufficient.
- Never emulate rollback by rewriting or deleting alias generations. Use `--action rollback`; the
  registry verifies that the target previously occupied the same environment. Move active aliases
  before retiring a promotion so no environment resolves to terminal evidence.

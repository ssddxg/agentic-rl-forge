# Reproducible experiment matrices

AgenticRLForge can expand, execute, resume, and inspect local experiment matrices without adding a
second tracking platform. The workflow composes the repository's existing collection configs,
dataset fingerprints, rollout plans, checkpoint manifests, trainer-batch manifests, benchmark
reports, and paired comparison reports into one content-addressed experiment plan.

The design borrows four proven ideas while keeping the current Pydantic, Typer, PyYAML, and local
artifact stack: configuration composition and multi-run expansion from
[Hydra](https://github.com/hydra-ecosystem/hydra), local reproducible experiment comparison from
[DVC](https://github.com/iterative/dvc), parameter/metric/artifact lineage from
[MLflow](https://github.com/mlflow/mlflow), and explicit trials plus result inspection from
[Ray Tune](https://github.com/ray-project/ray). None of those projects is a runtime dependency.

## Matrix specification

Start from a validated collection config and bind matrix parameters to existing dotted paths:

```yaml
schema_version: 1
name: search-r1-sampling-ablation
base_config: ../search_r1_collection.yaml

fixed_parameters:
  rollouts_per_task: 5

axes:
  seed: [42, 43]
  temperature: [0.8, 1.0]

exclude:
  - seed: 43
    temperature: 0.8

config_bindings:
  seed: seed
  temperature: temperature
  rollouts_per_task: rollouts_per_task

stages:
  - name: collect
    command:
      - arf
      - collect-search-r1
      - data/qa.jsonl
      - artifacts/experiments/{trial_id}/collection
      - --config
      - "{config_path}"
    inputs:
      - kind: dataset
        path: data/qa.jsonl
      - kind: collection_config
        path: "{config_path}"
    outputs:
      - kind: other
        path: artifacts/experiments/{trial_id}/collection/trajectories.db

max_trials: 8
```

Axis names are sorted before Cartesian expansion, axis values remain in their reviewed order, and
type-aware exclusion rules remove exact parameter combinations. Duplicate axis values, unknown
bindings, missing base-config paths, empty matrices, unsafe artifact paths, forward dependencies,
and expansions beyond `max_trials` are rejected.

`{trial_id}` and parameter names can appear in commands and artifact paths. `{config_path}` points
to the immutable resolved JSON config materialized under the experiment state directory; JSON is
valid input for the existing YAML config loader. Format conversions and arbitrary template
expressions are intentionally unsupported.

See [`configs/experiments/search_r1_matrix.yaml`](../configs/experiments/search_r1_matrix.yaml) for
a complete collection, rollout-plan snapshot, evaluation, and regression-gate example.

## Dry-run expansion

Expand and review the full plan without running a stage:

```bash
arf experiment-plan configs/experiments/search_r1_matrix.yaml \
  --output artifacts/experiments/search-r1.plan.json
```

Every trial ID binds the experiment name, type-preserving parameter values, and resolved config
digest. The plan ID additionally binds the source matrix bytes, base-config bytes, rendered command
arguments, dependency graph, declared inputs and outputs, and regression gates. Reformatting or
changing either source file therefore requires a new review even if an accidental semantic
equivalence remains.

The plan is canonical JSON and contains no observation timestamp, so expanding unchanged inputs is
byte deterministic.

## Preview, execute, and resume

Inspect current state without starting commands:

```bash
arf experiment-run artifacts/experiments/search-r1.plan.json \
  --root . \
  --state-dir artifacts/experiment-state \
  --fail-on-incomplete
```

Execute only after confirming the reviewed plan ID:

```bash
arf experiment-run artifacts/experiments/search-r1.plan.json \
  --root . \
  --state-dir artifacts/experiment-state \
  --execute \
  --confirm-plan-id EXPERIMENT_PLAN_ID \
  --workers 2 \
  --fail-on-failed \
  --fail-on-regression \
  --output artifacts/experiments/search-r1.report.json
```

Trials may run concurrently, while stages within a trial follow their declared dependency order.
Commands are passed directly to `subprocess` as argument arrays with `shell=False`, an empty stdin,
the selected project root as the working directory, and separate stdout/stderr logs. Matrix files
are executable specifications and must be reviewed like scripts.

Successful stages write immutable evidence containing the exact executed command, input and output
SHA-256 identities, parsed content IDs, timestamps, and log paths. Exact retries revalidate those
files and reuse the record without rerunning the command. A changed or missing artifact makes the
stage `invalid` and is never silently regenerated over historic success evidence.

Failed attempts are append-only. Correct the external condition or input, then rerun the same plan;
the stage can succeed later while retaining prior failure evidence. Dependent stages remain
`blocked` until their prerequisites have immutable success records.

State is stored under:

```text
STATE_DIR/
  EXPERIMENT_PLAN_ID/
    plan.json
    EXPERIMENT_TRIAL_ID/
      resolved-config.json
      stages/STAGE/
        record.json
        attempts/EXPERIMENT_ATTEMPT_ID.json
        logs/*.stdout.log
        logs/*.stderr.log
```

## Artifact lineage

Declared artifacts always receive byte size and SHA-256 evidence. Typed kinds also validate and
summarize repository contracts:

| Kind | Parsed identity and useful lineage |
| --- | --- |
| `collection_config` | Search-R1 plan config digest, dataset, model, policy, and seed |
| `dataset` | Raw content SHA-256 |
| `dataset_manifest` | Dataset manifest ID, split, records, and content digest |
| `dataset_collection_manifest` | Collection ID, split set, record count, and ID digest |
| `rollout_plan` | Plan ID, policy, source/config digests, tasks, and slots |
| `checkpoint_manifest` | Checkpoint/run IDs, step, policy, parent, and dataset lineage |
| `trainer_batch_manifest` | Batch ID, source run, policy, groups, trajectories, and payload digest |
| `benchmark_report` | Report ID, benchmark, run, task/attempt counts, and metric names |
| `comparison_report` | Comparison ID, baseline/candidate, matched tasks, and metric names |
| `other` | Byte size and SHA-256 only |

This does not replace the deeper verifier for a checkpoint payload, trainer object store, or run
artifact bundle. It proves exactly which canonical manifest was consumed or produced by a stage.

## Regression gates

A gate references a declared benchmark or comparison output. Benchmark gates read `MetricValue.value`.
Comparison gates can read `baseline_mean`, `candidate_mean`, `absolute_delta`, `relative_delta`,
`confidence_low`, or `confidence_high`.

```yaml
gates:
  - name: pass-rate-non-regression
    artifact_path: artifacts/experiments/{trial_id}/comparison.json
    metric: task_pass_rate
    statistic: confidence_low
    minimum: -0.01
  - name: error-budget
    artifact_path: artifacts/experiments/{trial_id}/benchmark.json
    metric: error_rate
    statistic: value
    maximum: 0.05
```

The report is deterministic for unchanged evidence and classifies trials as `incomplete`, `failed`,
`regression`, or `complete`. Use `--fail-on-incomplete`, `--fail-on-failed`, and
`--fail-on-regression` independently in CI. A stage can succeed while its trial is classified as a
regression; this distinguishes valid experiment execution from an unacceptable result.

## Cross-plan index and parameter/metric table

Discover every canonical plan in a local state directory without executing commands:

```bash
arf experiment-index \
  --root . \
  --state-dir artifacts/experiment-state \
  --output artifacts/experiments/index.json \
  --csv-output artifacts/experiments/index.csv \
  --fail-on-issues
```

The builder loads each persisted plan, regenerates its report from immutable stage evidence, and
rechecks benchmark artifact paths and SHA-256 identities before extracting values. Every row binds
its plan, current deterministic report, trial, state, parameters, and metrics. The index itself is
content-addressed and contains no scan time.

Metric IDs are derived from:

- experiment name;
- stage name and declared output position;
- benchmark or comparison artifact kind;
- metric and statistic;
- unit.

This lets the same semantic metric line up across compatible plans even when trial-specific output
paths differ. Benchmark artifacts contribute their scalar `value`. Comparison artifacts contribute
baseline and candidate means, absolute and relative deltas, and confidence bounds when present.
The CSV uses stable parameter columns followed by these metric IDs; the JSON metric catalog provides
the human-readable mapping.

The state directory is treated as an evidence namespace. Unknown top-level entries, missing plans,
noncanonical JSON, identity mismatches, and metric extraction failures are recorded as sorted issues
instead of being silently skipped. `--fail-on-issues` exits nonzero after writing the reviewable
index and CSV.

## Ranking, baselines, and Pareto fronts

Use metric IDs from the index to create a deterministic analysis:

```bash
arf experiment-analyze artifacts/experiments/index.json \
  --objective PASS_RATE_METRIC_ID:maximize \
  --objective LATENCY_METRIC_ID:minimize:0.25 \
  --baseline-plan-id EXPERIMENT_PLAN_ID \
  --baseline-trial-id EXPERIMENT_TRIAL_ID \
  --output artifacts/experiments/analysis.json \
  --markdown-output artifacts/experiments/dashboard.md \
  --html-output artifacts/experiments/dashboard.html
```

Objective syntax is `METRIC_ID:maximize[:WEIGHT]` or `METRIC_ID:minimize[:WEIGHT]`. Weights must be
finite and positive. Objectives are type-safe, unique, and canonically ordered by metric ID.

Ranking follows four explicit rules:

1. Trials must be in an eligible completed state and contain every objective. `complete` is the
   default; `--include-regressions` also admits completed trials whose declared gates failed.
2. Each metric is direction-aligned and min-max normalized over the eligible selection. A constant
   metric contributes `1.0` to every candidate rather than changing their relative order.
3. The display score is the weighted mean of normalized objectives. Equal scores use plan ID and
   trial ID as deterministic tie breakers.
4. Pareto membership uses original, unnormalized values and objective directions. A trial is on the
   front exactly when no other eligible trial is at least as good on every objective and strictly
   better on one.

An optional baseline is identified by both plan and trial ID. The analysis records the baseline and
candidate values, raw candidate-minus-baseline delta, and direction-aware improvement for every
objective. This avoids ambiguity when the same content-derived trial appears in more than one plan.

`--experiment-name` and `--plan-id` restrict selection. `--max-candidates` bounds the quadratic
Pareto comparison, and `--fail-on-ineligible` can enforce that every selected trial is rankable.
The canonical analysis ID binds the source index, complete policy, scores, baseline deltas, and
Pareto membership. Contract validation recomputes those relationships rather than trusting stored
rank values.

Markdown and standalone HTML exports include the ranking, objective values, baseline improvements,
metric catalog, report inventory, and discovery issues. They contain no scripts, remote assets, or
timestamps and escape user-controlled parameter text. CSV string parameters that resemble formulas
are prefixed before spreadsheet use. The JSON index and analysis remain the verification contracts;
CSV, Markdown, and HTML are deterministic presentation derivatives.

Weighted scores and Pareto fronts help navigate tradeoffs but do not prove statistical
significance. Use paired comparison reports, bootstrap confidence intervals, and declared gates for
release decisions.

## Experiment promotion

An analysis helps choose a candidate; promotion proves that the chosen candidate still has the same
evidence at the release boundary. The workflow takes design cues from versioned model governance in
[MLflow](https://github.com/mlflow/mlflow) and reviewable documentation in
[Hugging Face Hub model cards](https://github.com/huggingface/hub-docs), while continuing to use the
project's existing Pydantic contracts, Typer CLI, conditional local writes, and content-addressed
evidence. No registry or model-card package is added as a runtime dependency.

Preview one exact plan/trial pair:

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

Preview does not create the promotion directory. It performs the following checks against current
bytes rather than only the saved dashboard:

- rebuild the complete experiment index and require the saved index ID to remain current;
- regenerate the selected plan's report and require its report ID to match the analyzed row;
- require the selected row to remain eligible and enforce complete/regression state policy;
- enforce optional maximum-rank and Pareto-front selection rules;
- require configured artifact kinds and reconstruct one deduplicated artifact graph;
- parse checkpoint manifests, verify local checkpoint payload path, size, and SHA-256, and reject
  project-root escapes;
- require checkpoint dataset lineage to match a promoted raw dataset or dataset-manifest digest;
- report discovery issues, multiple checkpoints, remote payloads, missing payloads, and mismatches as
  explicit policy checks.

The preview includes a content-addressed reproducibility manifest. Each artifact has a scope:

| Scope | Meaning |
| --- | --- |
| `project` | Safe path relative to the selected project root |
| `state` | Safe path relative to the experiment state directory |
| `remote` | Secret-free `https`, `s3`, `gs`, or `az` URI with declared size and SHA-256 |

References preserve artifact kind, content ID, size, digest, producing/consuming stages, and roles.
Checkpoint payload files are references, not copies. The manifest also binds resolved model, policy,
dataset, config digest, parameters, index, analysis, plan, report, and trial identities.

After review, execute with the same policy and exact preview ID:

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

Execution recomputes the preview immediately before writing. The default policy requires one
independently verified approver; `--minimum-approvals` can raise the count, and the operator cannot
also approve unless `--operator-may-approve` is explicit. Identities are sorted and unique.

The promotion name is an immutable decision namespace:

```text
PROMOTION_DIR/
  NAME/
    decision.json
    record.json
    manifest.json
    model-card.md
    plan.json
    report.json
    index.json
    analysis.json
```

`decision.json` is created first and embeds the complete preview and manifest. It is authoritative;
the other files are deterministic sidecars. An exact retry recreates a missing sidecar, while any
different candidate, policy, operator, rationale, approval set, or model-card digest conflicts with
the existing decision and cannot replace it.

The generated model card has stable metadata, model/dataset identity, experiment and report IDs,
objective directions, weights and values, rank, Pareto status, operator rationale, approvers,
reproducibility instructions, and limitations. User-controlled text is escaped. It contains no
observation time, remote assets, executable script, or copied model payload.

The promotion record is strong engineering evidence but not a cryptographic human signature.
Organizations that require signed approvals, quorum, or identity-provider authorization should sign
or countersign the immutable decision through their trusted release system.

### Portable metadata archive and publisher attestation

Package the complete promotion directory after the decision has been created:

```bash
arf experiment-promotion-pack \
  artifacts/experiment-promotions/search-r1-candidate \
  artifacts/releases/search-r1-candidate.promotion.tar.gz

arf experiment-promotion-inspect \
  artifacts/releases/search-r1-candidate.promotion.tar.gz
```

Packing accepts exactly the eight documented promotion files. It reparses every JSON contract,
requires canonical bytes, checks that `decision.json` and `record.json` are identical, validates all
plan/report/index/analysis identities against the decision, and regenerates the model card from the
recorded evidence. The emitted tar headers, member order, gzip header, file modes, and compression
settings are canonical, so identical promotion metadata produces identical archive bytes and the
same adjacent `.sha256` file. Heavyweight artifacts referenced by `manifest.json` remain outside the
archive.

Install the `signing` extra, generate or reuse a release key, and attest the exact archive receipt:

```bash
arf manifest-keygen secrets/promotion-release.key trust/promotion-release.pub

arf experiment-promotion-sign \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --private-key secrets/promotion-release.key

arf experiment-promotion-inspect \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  --attestation artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --public-key trust/promotion-release.pub
```

The attestation binds the archive digest, promotion ID, reproducibility-manifest ID, signer key ID,
and signing time. Trust comes only from public keys supplied independently by the receiver; the key
embedded in the attestation is not a trust anchor. Repeat `--public-key` during key rotation.

A receiver can require the same trust check before any files become visible:

```bash
arf experiment-promotion-unpack \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  received/search-r1-candidate \
  --attestation artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --public-key trust/promotion-release.pub
```

Extraction rejects links, devices, path changes, duplicate or extra members, noncanonical metadata,
noncanonical gzip/tar bytes, malformed contracts, and inconsistent sidecars. Files are written into
an isolated temporary directory, revalidated, and atomically renamed only after every check passes.
The publisher signature authenticates who released these exact bytes; it does not authenticate the
human approver strings recorded in the promotion decision.

### Append-only lifecycle and environment aliases

Register the signed archive as a candidate with a separate deployment authorization:

```bash
arf experiment-promotion-lifecycle \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --registry-dir artifacts/promotion-registry \
  --target-stage candidate \
  --operator release-operator \
  --reason "register the reviewed candidate" \
  --authorizer release-owner \
  --public-key trust/promotion-release.pub \
  --preview-output artifacts/releases/candidate-lifecycle-preview.json \
  --fail-on-ineligible
```

Review the complete preview and confirm its exact ID:

```bash
arf experiment-promotion-lifecycle \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --registry-dir artifacts/promotion-registry \
  --target-stage candidate \
  --operator release-operator \
  --reason "register the reviewed candidate" \
  --authorizer release-owner \
  --public-key trust/promotion-release.pub \
  --execute --confirm-preview-id PROMOTION_LIFECYCLE_PREVIEW_ID
```

Repeat the preview/confirm cycle for `staging` and then `production`. The allowed lifecycle is:

```text
new → candidate → staging → production → retired
           └──────────────→ retired
```

A staging candidate can also be retired without reaching production. Events are stored under the
promotion ID with contiguous eight-digit sequence numbers. Each event commits the preceding event
ID, so the latest event transitively binds the complete history. Concurrent decisions for the same
next sequence compete for one conditional-create key; a different winner makes the stale preview
unexecutable.

Point an environment at a production promotion:

```bash
arf experiment-promotion-alias \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  --registry-dir artifacts/promotion-registry \
  --environment production --action assign \
  --operator deployment-operator \
  --reason "deploy the approved production candidate" \
  --authorizer deployment-owner \
  --public-key trust/promotion-release.pub \
  --preview-output artifacts/releases/production-alias-preview.json
```

Confirm with the same options plus `--execute --confirm-preview-id`. Environment aliases use their
own contiguous generation stream and compare-and-swap against the current event. By default only a
promotion at lifecycle stage `production` is eligible. Repeat `--allowed-stage` to define an
explicit environment policy, for example a candidate-only evaluation environment.

Rollback uses the old promotion archive and requires real ancestry in that environment:

```bash
arf experiment-promotion-alias \
  artifacts/releases/previous.promotion.tar.gz \
  artifacts/releases/previous.promotion.attestation.json \
  --registry-dir artifacts/promotion-registry \
  --environment production --action rollback \
  --operator deployment-operator \
  --reason "rollback after the monitored deployment gate failed" \
  --authorizer incident-commander \
  --public-key trust/promotion-release.pub
```

The preview records both the current alias event being rolled back and the earlier event that proved
the target promotion previously occupied this environment. A promotion cannot be retired while any
active environment alias still references it; move or roll back every alias first.

Inspect the complete registry in CI:

```bash
arf experiment-promotion-registry-status artifacts/promotion-registry \
  --output artifacts/releases/promotion-registry-status.json \
  --fail-on-issues
```

Status discovery rejects gaps, noncanonical records, broken predecessor links, fabricated rollback
ancestry, missing lifecycle evidence, retired active targets, and unrecognized registry files. The
status ID is deterministic and commits the latest event-chain heads for every promotion and
environment. Publisher trust is revalidated from the signed archive for every preview and execute,
but deployment authorization remains a separate quorum of explicit `--authorizer` identities.

### Receiver-side artifact acquisition

After authenticating the metadata archive, resolve its reproducibility manifest against explicit
receiver roots:

```bash
arf experiment-promotion-acquire \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  artifacts/received-promotions \
  --root /srv/agentic-rl-forge \
  --state-dir /srv/agentic-rl-forge/artifacts/experiment-state \
  --public-key trust/promotion-release.pub \
  --plan-output artifacts/releases/search-r1-acquisition-plan.json \
  --fail-on-ineligible
```

Preview is read-only, including not creating the destination root. It re-inspects the canonical
archive, verifies publisher trust, then resolves every `project` and `state` locator without
following symbolic links. Each regular file is streamed through its declared size and SHA-256.
Recognized collection configs, dataset manifests, rollout plans, checkpoint manifests, trainer
batches, benchmark reports, and comparison reports are reparsed and checked against their recorded
content IDs; canonical JSON is required for native JSON contracts.

Checkpoint payloads are resolved by artifact name, size, and digest rather than trusting the
sender's absolute local URI. Parent checkpoints must be present in the acquired manifest graph,
belong to the same run, have a lower step, preserve config and dataset lineage, and form an acyclic
chain. Checkpoint dataset digests must also appear in the acquired dataset evidence graph.

Remote `https`, `s3`, `gs`, and `az` references are reported as unresolved. The command performs no
network request and is ineligible by default. A reviewed handoff may use
`--allow-unresolved-remote`; remote checkpoint payloads additionally require
`--allow-missing-checkpoint-payloads`. Every relaxation is part of the plan ID.

After reviewing the canonical plan, confirm it exactly:

```bash
arf experiment-promotion-acquire \
  artifacts/releases/search-r1-candidate.promotion.tar.gz \
  artifacts/releases/search-r1-candidate.promotion.attestation.json \
  artifacts/received-promotions \
  --root /srv/agentic-rl-forge \
  --state-dir /srv/agentic-rl-forge/artifacts/experiment-state \
  --public-key trust/promotion-release.pub \
  --execute --confirm-plan-id PROMOTION_ACQUISITION_PLAN_ID \
  --record-output artifacts/releases/search-r1-acquisition-record.json
```

Execution recomputes the plan before writing. Files are streamed into
`acquisitions/PROMOTION_ACQUISITION_ID/{project,state}/...` and claimed with exclusive hard links;
existing bytes are accepted only when they match exactly. `record.json` is the final commit marker.
An interrupted attempt can resume from any exact subset, while unknown files, changed sources,
conflicting receiver bytes, a stale plan, or an over-budget graph fail closed.

## Safety boundaries

- Artifact paths must stay below `--root`; resolved symlinks that escape it are rejected.
- Commands never pass through a shell, but they can still execute arbitrary programs with the
  current process credentials.
- Environment variables are inherited so existing service credentials continue to work. Do not
  place secrets in matrix parameters, commands, configs, artifact summaries, or logs.
- Success evidence is immutable. If an output intentionally changes, create a new matrix/plan
  identity or use a new state directory instead of deleting evidence to force a rerun.
- Promotion acquisition destinations must be separate from both source roots. Do not add files
  inside a content-addressed acquisition prefix; unknown entries block completion and exact retry.
- Trial concurrency is bounded to 64. Stage-internal concurrency remains the responsibility of the
  invoked command and its reviewed config.

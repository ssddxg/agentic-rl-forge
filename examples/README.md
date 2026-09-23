# Examples

## Complete offline pipeline

Run the full CPU workflow without a model server or GPU:

```bash
arf offline-pipeline examples/data/qa.jsonl examples/data/corpus.jsonl \
  artifacts/offline-example
```

The repository wrapper remains available as
`python examples/offline_pipeline.py --output-dir artifacts/offline-example`.

The example performs grouped rollouts with deterministic success and failure variation, persists
each trajectory to SQLite and recoverable shards, records metrics, aggregates benchmark results,
filters groups by reward signal and semantic diversity, and produces a verified trainer batch.

Generated artifacts:

```text
artifacts/offline-example/
├── trajectories.db
├── benchmark-report.json
├── filtered-trajectories.jsonl
├── shards/
└── trainer-store/
```

The seeded policy is intentionally simple and exists to demonstrate data and control flow. It is
not a model-quality benchmark.

The command accepts arbitrary validated English or Chinese QA rows with `id`, `question`, and
`answer`, plus a corpus containing `id` and `contents`. It refuses to mix with an existing output
directory and requires each question to retrieve evidence containing an accepted answer. Use
`arf corpus-build` to create the corpus from a text or Markdown directory. Set
`--rollouts-per-task` between 2 and 4; these bounded deterministic variations guarantee that the
smoke workflow can actually exercise its learning-signal checks.

On Windows, keep the absolute output directory at 133 ordinary characters or fewer. The command
checks this before writing so that nested shard and trainer artifacts remain readable by standard
Windows tools; a short path such as `C:\arf-runs\run-1` is recommended for deeply nested projects.

To replace the seeded policy with a real OpenAI-compatible vLLM or SGLang endpoint, use
`arf collect-search-r1` with [`configs/search_r1_collection.yaml`](../configs/search_r1_collection.yaml).
The operational workflow is documented in [`docs/collection.md`](../docs/collection.md).

## Retriever

Build and search a corpus directly without starting a service:

```bash
arf corpus-build ./documents artifacts/corpus.jsonl
arf search "your question" --corpus artifacts/corpus.jsonl
```

Serve the same corpus over the Search-R1-compatible HTTP protocol:

```bash
arf serve-retriever examples/data/corpus.jsonl --port 8000
```

Request:

```bash
curl -s http://127.0.0.1:8000/retrieve \
  -H 'content-type: application/json' \
  -d '{"queries":["capital of France"],"topk":2,"return_scores":true}'
```

## Dataset manifest

```bash
arf dataset-manifest artifacts/example-dataset.json --name offline-example \
  --split-file demo=examples/data/qa.jsonl --id-field id --root "$PWD"
```

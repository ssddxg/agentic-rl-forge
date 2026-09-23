# Local knowledge-base search

AgenticRLForge includes a CPU-only local search workflow that does not require a model server,
GPU, Docker, or external database. It is suitable for searching project notes and for validating
the retrieval side of a Search-R1 pipeline.

## Build a corpus from files

```bash
arf corpus-build ./documents ./data/corpus.jsonl
```

The builder recursively reads UTF-8 `.txt`, `.md`, `.markdown`, and `.rst` files. It:

- ignores hidden paths and symbolic links;
- rejects unreadable, non-UTF-8, and oversized input;
- creates deterministic overlapping chunks and IDs;
- records each chunk's source, title, index, and chunk count;
- removes duplicate chunk contents across files;
- writes through a temporary file and publishes atomically;
- refuses to replace an existing output unless `--force` is explicit.

Useful controls:

```bash
arf corpus-build ./documents ./data/corpus.jsonl \
  --chunk-size 1600 --chunk-overlap 160 \
  --extension .txt --extension .md --max-file-mb 25
```

## Validate and identify a corpus

```bash
arf corpus-check ./data/corpus.jsonl
arf corpus-check ./data/corpus.jsonl --json
```

Validation rejects malformed JSON, non-object rows, blank IDs or contents, duplicate IDs, and an
empty corpus. Errors include the file and line number. The report includes both the exact file
SHA-256 and an order-independent corpus identity over document content and metadata.

The accepted JSONL shape is:

```json
{"id":"doc-1","contents":"Searchable text","source":"notes.md"}
```

`id` and `contents` are required. Other JSON-compatible fields are preserved as result metadata.

## Search without a service

```bash
arf search "deployment instructions" --corpus ./data/corpus.jsonl --top-k 5
arf search "部署说明" --corpus ./data/corpus.jsonl --json
```

The built-in tokenizer applies Unicode NFKC normalization, case folding, Latin word splitting, and
CJK unigram/bigram tokenization. Search results therefore work with English, Chinese, Japanese,
Korean, and mixed text. BM25 uses an in-memory inverted index and stable document-ID tie breaking.

## Serve the corpus over HTTP

```bash
arf serve-retriever ./data/corpus.jsonl --port 8000
```

Direct launches bind to `127.0.0.1` by default. The HTTP service has no built-in authentication,
so keep it on loopback unless an authenticated reverse proxy and suitable network controls protect
it. Do not expose it directly to the internet or an untrusted network.

Endpoints:

- `GET /health` provides a stable liveness response.
- `GET /stats` reports the active generation, corpus/index identities, index size, tokenizer, and
  latest reload outcome.
- `GET /metrics` exposes Prometheus text metrics.
- `POST /retrieve` implements the Search-R1 batch protocol.

Example request:

```json
{"queries":["deployment instructions"],"topk":5,"return_scores":true}
```

A request may contain 1-64 queries, each 1-4096 characters after trimming, and `topk` must be
between 1 and 100. Search work runs outside the asynchronous HTTP event loop and concurrent search
batches are bounded; configure the limit with `--max-concurrent-searches`.

Enable safe polling reloads when the corpus is replaced by another process:

```bash
arf serve-retriever ./data/corpus.jsonl --reload-interval 5
```

The service builds and validates a complete replacement index before switching generations. A bad
replacement is reported in `/stats` and metrics while the last valid index remains available. Each
poll computes a content digest, so larger corpora should use a moderate interval rather than
sub-second polling.

## Docker

The repository's default `docker compose up --build` command now starts the complete Studio for
ordinary users. To run only this developer-facing retrieval endpoint, build the same image and
override its command:

```bash
docker build -t agentic-rl-forge:local .
docker run --rm \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/data:/app/corpus:ro" \
  agentic-rl-forge:local \
  arf serve-retriever /app/corpus/corpus.jsonl --host 0.0.0.0 --port 8000
```

Mounting the containing directory is intentional: when a builder atomically replaces the host
file, the container can observe the new directory entry. Add `--reload-interval 5` to the final
command if hot reload is needed. The example publishes only to `127.0.0.1`; the same
no-authentication warning applies if that binding is changed.

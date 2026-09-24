# Changelog

All notable changes to AgenticRLForge are documented here. The project follows semantic
versioning. Versions before 0.3.0 were pre-public development milestones; public releases use Git
tags in the form `vMAJOR.MINOR.PATCH`.

## 0.3.0 — 2026-09-24

First public source release.

- Added polished GitHub Release notes, installable wheel and source assets, checksums, release-audit
  evidence, and versioned GitHub Container Registry images for the public launch.
- Repositioned Studio around the actual Agent RL engine: runtime diagnostics, a real inspectable
  agent trajectory, custom QA/corpus datasets, background offline RL pipeline runs, persistent
  results, and explicit separation between laptop validation and GPU weight training.
- Added a local, browser-based Studio for ordinary users with persistent knowledge bases,
  drag-and-drop document import, multilingual search, cited answers, and model settings.
- Added safe PDF, DOCX, HTML, Markdown, reStructuredText, and plain-text ingestion with source
  management, duplicate protection, size limits, deletion, and index rebuilding.
- Added one-click Windows startup, cross-platform launch scripts, and a persistent Docker Compose
  deployment that opens the complete Studio instead of only the developer retrieval endpoint.
- Kept local search fully usable without an AI account; optional OpenAI-compatible models can be
  configured for grounded answers without exposing API keys to the browser or API responses.
- Removed internal development-session logs and local runtime artifacts from the public project
  surface; release auditing now focuses on publishable repository evidence.

## 0.2.0 — 2026-09-22

- Added deterministic text and Markdown corpus building with strict validation, metadata,
  deduplication, safe chunking, content identities, and atomic output.
- Added a local `arf search` command and multilingual retrieval with Unicode normalization plus
  CJK unigram and bigram support.
- Reworked the built-in BM25 backend around an inverted index and added bounded, non-blocking HTTP
  retrieval, service statistics, concurrency protection, and optional safe corpus hot reload.
- Added a reusable offline pipeline command for validating arbitrary English or Chinese QA data
  through real answer-bearing retrieval evidence, rollout, persistence, evaluation, filtering,
  and verified trainer-batch export; output manifests now fingerprint both QA and corpus inputs.
- Expanded `arf doctor` with core, server, and training profiles, strict exit codes, and stable JSON
  reports suitable for setup scripts and CI.
- Hardened cross-filesystem corpus publication, Windows deep-path offline staging,
  locale-independent JSON output, cancelled-request concurrency accounting, Docker
  directory-mounted hot reload, and Windows/Linux one-command setup smoke coverage.

## 0.1.1 — 2026-09-22

- Added one-command setup and verification scripts for Windows, Linux, and macOS.
- Added a hardened Docker Compose quick start for the local Search-R1-compatible retriever.
- Added cross-platform packaging and smoke-test coverage for GitHub-hosted development.
- Improved Windows compatibility for paths, archives, atomic publication, and platform-specific
  filesystem behavior.

## 0.1.0 — 2026-09-20

Pre-public development milestone.

- Added typed, stateful tool environments and asynchronous multi-turn agent rollouts.
- Added Search-R1-compatible retrieval, grouped GRPO collection, verl export, PRM, MCTS,
  rejection sampling, Nash-MD utilities, and hindsight relabeling.
- Added SQLite and sharded trajectory persistence, evaluation, comparison, checkpoint, dataset,
  trainer-batch, and observability workflows.
- Added deterministic experiment matrices, analysis, promotion, signed archives, lifecycle and
  environment aliases, receiver acquisition, and lineage verification.
- Added conditional local/S3 artifact transport, resumable release mirroring and retention, plus
  explicitly allowlisted HTTPS/S3 promotion artifact fetching.
- Added a CPU-only offline pipeline, strict typing, release auditing, security guidance, and a
  comprehensive automated test suite.

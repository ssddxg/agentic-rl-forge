## Problem

Describe the user, research, or operational problem this change addresses.

## Changes

- Describe the main implementation changes here.

## Verification

- [ ] Tests cover the changed behavior and important failure cases.
- [ ] `ruff check .` passes.
- [ ] `ruff format --check src tests examples` passes.
- [ ] `mypy src examples/offline_pipeline.py` passes.
- [ ] Relevant documentation and examples are updated.
- [ ] No credentials, private data, generated artifacts, or local absolute paths are included.

## RL and data integrity

- [ ] Not applicable.
- [ ] Policy and environment versions are recorded for new trajectories.
- [ ] Derived or replayed data has explicit lineage and cannot silently enter on-policy batches.
- [ ] Metric denominators, dataset versions, and benchmark settings are documented.

## Operational impact

Note compatibility, migration, resource, security, and deployment effects.

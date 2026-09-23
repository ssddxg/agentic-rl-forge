# Contributing

Contributions are welcome when they preserve reproducibility, data provenance, and the
separation between on-policy rollouts and derived data.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,studio,research,signing]"
make check
```

On Windows, activate with `.venv\Scripts\Activate.ps1`, then run the same install command and
`python -m pytest`. See the commands in `Makefile` or `scripts/check.ps1` for the complete local
quality suite.

## Change requirements

- Add tests for behavior changes and failure cases.
- Keep the core package usable without CUDA.
- Do not import verl, vLLM, SGLang, WebArena, or browser runtimes from core modules.
- Record policy and environment versions for new trajectory producers.
- Attach parent IDs and transform names to all replayed or transformed data.
- Define metric denominators and benchmark versions in reports.
- Document configuration and migration changes.

## Pull requests

Describe the problem, design, tests, and operational impact. Algorithm changes should include
an ablation plan; performance changes should include the exact hardware, model, sequence
length, batch size, rollout count, and whether environment latency is included.

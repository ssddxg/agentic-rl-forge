# GPU and verl validation runbook

This runbook is the acceptance gate for claims that require a real training environment. Preserve
the exact command output and artifact paths with the private run artifacts, and publish only a
sanitized summary. Do not infer throughput or benchmark gains from a CPU-only run.

## 1. Freeze the environment

Use a dedicated machine or container with a supported NVIDIA driver and CUDA runtime. Before
installation, record:

```bash
nvidia-smi
python --version
git -C /path/to/verl rev-parse HEAD
```

Install the project and the chosen verl rollout backend in that environment:

```bash
python -m pip install -e ".[data,server]"
# Install verl and exactly one validated rollout backend using their pinned environment files.
```

Do not install unpinned framework revisions into an existing production environment. Store the
verl commit, PyTorch/CUDA versions, vLLM or SGLang version, GPU model, GPU count, and driver in the
run manifest metadata.

## 2. Preflight

```bash
arf doctor
python -c "from agentic_rl_forge.integrations.verl_search_tool import AgenticRLForgeSearchTool"
bash -n recipes/verl/run_search_r1_grpo.sh
```

Acceptance criteria:

- `torch`, `verl`, and the selected rollout engine are importable;
- the native search tool imports against the pinned verl revision;
- every visible GPU passes a small allocation test;
- no other process consumes enough memory to invalidate the smoke test.

## 3. Prepare a bounded dataset and retriever

Start with a small, non-sensitive QA slice containing enough tasks to fill at least two GRPO
batches. Keep train and validation source IDs disjoint.

```bash
arf prepare-search-r1 data/train.jsonl artifacts/train.parquet --dataset smoke-train
arf prepare-search-r1 data/validation.jsonl artifacts/validation.parquet \
  --dataset smoke-validation
arf serve-retriever data/corpus.jsonl --host 0.0.0.0 --port 8000
```

The retriever has no built-in authentication. Binding to `0.0.0.0` is intended only for an
isolated, trusted training network; otherwise keep the default `127.0.0.1` binding or place an
authenticated reverse proxy in front of the service.

Before training, send a manual `/retrieve` request and verify that the expected passage ranks in
the configured top-k results.

## 4. Two-step training smoke test

Use the normal recipe with small overrides. Exact micro-batch values depend on GPU memory and the
pinned verl revision; record every override.

```bash
export TRAIN_FILES="$PWD/artifacts/train.parquet"
export VAL_FILES="$PWD/artifacts/validation.parquet"
export MODEL_PATH="/models/your-search-policy"
export ROLLOUT_ENGINE="vllm"
export GPUS_PER_NODE=2
export EXPERIMENT_NAME="search-r1-smoke"

bash recipes/verl/run_search_r1_grpo.sh \
  data.train_batch_size=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  trainer.total_training_steps=2 \
  trainer.save_freq=1 \
  trainer.test_freq=1
```

Acceptance criteria:

- two optimizer steps complete without OOM, deadlock, or worker restart;
- rollout logs show real search calls and observations;
- observation tokens are excluded from the policy response mask;
- each GRPO group contains the configured number of samples from one task and policy version;
- rewards, KL, response length, tool calls, rollout latency, and GPU memory are finite;
- at least one checkpoint can be loaded into the rollout engine and used for inference.

## 5. Register and verify checkpoints

```bash
arf checkpoint-register artifacts/checkpoints \
  --run-id RUN_ID --step 2 --policy-version POLICY_VERSION \
  --config configs/search_r1.yaml \
  --artifact actor=/path/to/checkpoint/actor.safetensors \
  --artifact optimizer=/path/to/checkpoint/optimizer.pt
```

Run `arf checkpoint-verify` after copying or uploading artifacts. A remote upload is not considered
verified until its downloaded bytes match the registered SHA-256 and size.

## 6. Scale and benchmark

Only increase batch size, group size, response length, concurrency, or node count one dimension at
a time. For every experiment, persist trajectories and compare the candidate policy on the same
task IDs and attempt counts:

```bash
arf compare artifacts/trajectories.db \
  --baseline-policy BASELINE --candidate-policy CANDIDATE \
  --benchmark search-r1 --output artifacts/comparison.json
```

Report throughput with the evaluated task count, attempts per task, hardware, policy and
environment versions, prompt/response limits, and aggregation definition. Report WebArena gains
only after the authenticated browser environment and official evaluator complete successfully.

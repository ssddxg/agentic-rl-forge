from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import orjson

from agentic_rl_forge.pipelines import run_offline_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete offline Agent RL data path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "qa.jsonl",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parent / "data" / "corpus.jsonl",
    )
    parser.add_argument("--rollouts-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    arguments = parser.parse_args()
    summary = asyncio.run(
        run_offline_pipeline(
            arguments.output_dir,
            arguments.data.resolve(strict=True),
            arguments.corpus.resolve(strict=True),
            rollouts_per_task=arguments.rollouts_per_task,
            seed=arguments.seed,
            max_concurrency=arguments.max_concurrency,
        )
    )
    print(orjson.dumps(summary.model_dump(mode="json"), option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()

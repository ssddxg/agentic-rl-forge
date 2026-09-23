from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_offline_pipeline_produces_verified_training_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    output_dir = tmp_path / "offline-pipeline"
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "examples" / "offline_pipeline.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)

    assert summary["trajectory_count"] == 8
    assert summary["accepted_trajectory_count"] == 8
    assert summary["group_pass_rate"] == 1.0
    assert summary["learning_signal_group_rate"] == 1.0
    assert summary["trainer_batch_valid"] is True
    trajectory_counters = {
        key: value
        for key, value in summary["metrics"]["counters"].items()
        if key.startswith("arf_trajectories_total")
    }
    assert sum(trajectory_counters.values()) == 8.0
    assert summary["metrics"]["counters"]["arf_observation_tokens_total"] > 0
    assert len((output_dir / "filtered-trajectories.jsonl").read_text().splitlines()) == 8
    assert (output_dir / "trajectories.db").is_file()
    assert (output_dir / "benchmark-report.json").is_file()
    assert list((output_dir / "shards").rglob("manifests/*.json"))
    assert list((output_dir / "trainer-store").rglob("manifest.json"))


def test_offline_pipeline_accepts_custom_chinese_data_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[1]
    data_path = tmp_path / "问答.jsonl"
    corpus_path = tmp_path / "资料.jsonl"
    output_dir = tmp_path / "自定义运行"
    data_path.write_text(
        json.dumps(
            {"id": "cn-1", "question": "法国的首都是哪里?", "answer": "巴黎"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps(
            {"id": "fr", "contents": "巴黎是法国的首都。"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(project_root / "examples" / "offline_pipeline.py"),
        "--output-dir",
        str(output_dir),
        "--data",
        str(data_path),
        "--corpus",
        str(corpus_path),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)

    assert summary["trajectory_count"] == 4
    assert summary["accepted_trajectory_count"] == 4
    assert summary["trainer_batch_valid"] is True
    assert output_dir.is_dir()

    repeated = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "output directory already exists" in repeated.stderr

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import orjson

from agentic_rl_forge.contracts import (
    RolloutPlan,
    RunArtifactManifest,
    RunArtifactRef,
    RunArtifactVerification,
    RunManifest,
    RunStatus,
    ShardVerification,
    Trajectory,
    TrajectoryShardManifest,
)
from agentic_rl_forge.evaluation import BenchmarkReport
from agentic_rl_forge.storage.blobs import LocalBlobStore


class RunArtifactBundle:
    _ARTIFACT_SPECS = (
        ("rollout_plan", "application/json"),
        ("run_manifest", "application/json"),
        ("trajectories", "application/x-ndjson"),
        ("benchmark_report", "application/json"),
        ("metrics", "text/plain; version=0.0.4"),
        ("summary", "application/json"),
        ("shard_manifest", "application/json"),
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def build(
        self,
        *,
        run_id: str,
        plan_id: str,
        shard_manifest_id: str,
        created_at: datetime,
    ) -> RunArtifactManifest:
        relative_paths = self._expected_paths(run_id, plan_id, shard_manifest_id)
        references = []
        for name, media_type in self._ARTIFACT_SPECS:
            relative_path = relative_paths[name]
            path = self.root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"required run artifact is missing: {relative_path}")
            payload = path.read_bytes()
            references.append(
                RunArtifactRef(
                    name=name,
                    relative_path=relative_path,
                    media_type=media_type,
                    size_bytes=len(payload),
                    content_digest=hashlib.sha256(payload).hexdigest(),
                )
            )
        artifacts = tuple(references)
        manifest_id = RunArtifactManifest.expected_manifest_id(
            run_id=run_id,
            plan_id=plan_id,
            shard_manifest_id=shard_manifest_id,
            created_at=created_at,
            artifacts=artifacts,
        )
        manifest = RunArtifactManifest(
            manifest_id=manifest_id,
            run_id=run_id,
            plan_id=plan_id,
            shard_manifest_id=shard_manifest_id,
            created_at=created_at,
            artifacts=artifacts,
        )
        LocalBlobStore(self.root).put_if_absent(
            self.manifest_relative_path(run_id),
            manifest.canonical_bytes() + b"\n",
        )
        return manifest

    def load(self, path: Path | str) -> RunArtifactManifest:
        return RunArtifactManifest.model_validate_json(Path(path).read_bytes())

    def verify(self, manifest: RunArtifactManifest) -> RunArtifactVerification:
        missing: list[str] = []
        mismatched: list[str] = []
        verified: dict[str, Path] = {}
        for artifact in manifest.artifacts:
            path = self.root / artifact.relative_path
            if not path.is_file():
                missing.append(artifact.name)
                continue
            payload = path.read_bytes()
            if (
                len(payload) != artifact.size_bytes
                or hashlib.sha256(payload).hexdigest() != artifact.content_digest
            ):
                mismatched.append(artifact.name)
                continue
            verified[artifact.name] = path

        semantic_errors: list[str] = []
        shard_verification: ShardVerification | None = None
        if len(verified) == len(manifest.artifacts):
            try:
                plan = RolloutPlan.model_validate_json(verified["rollout_plan"].read_bytes())
                run = RunManifest.model_validate_json(verified["run_manifest"].read_bytes())
                trajectories = self._load_trajectories(verified["trajectories"])
                report = BenchmarkReport.model_validate_json(
                    verified["benchmark_report"].read_bytes()
                )
                summary = orjson.loads(verified["summary"].read_bytes())
                shard_manifest = TrajectoryShardManifest.model_validate_json(
                    verified["shard_manifest"].read_bytes()
                )
                self._validate_linkage(
                    manifest,
                    plan=plan,
                    run=run,
                    trajectories=trajectories,
                    report=report,
                    summary=summary,
                    shard_manifest=shard_manifest,
                    errors=semantic_errors,
                )
                shard_verification = self._verify_shards(shard_manifest)
            except (KeyError, TypeError, ValueError) as error:
                semantic_errors.append(f"artifact_contract_error:{type(error).__name__}:{error}")

        valid = (
            not missing
            and not mismatched
            and not semantic_errors
            and shard_verification is not None
            and shard_verification.valid
            and shard_verification.complete
        )
        return RunArtifactVerification(
            manifest_id=manifest.manifest_id,
            valid=valid,
            verified_artifact_count=len(verified),
            missing_artifacts=tuple(missing),
            mismatched_artifacts=tuple(mismatched),
            semantic_errors=tuple(semantic_errors),
            shard_verification=shard_verification,
        )

    @staticmethod
    def manifest_relative_path(run_id: str) -> str:
        return f"runs/{run_id}/artifact-manifest.json"

    @staticmethod
    def _expected_paths(run_id: str, plan_id: str, shard_manifest_id: str) -> dict[str, str]:
        return {
            "rollout_plan": f"plans/{plan_id}.json",
            "run_manifest": f"runs/{run_id}/run-manifest.json",
            "trajectories": f"runs/{run_id}/trajectories.jsonl",
            "benchmark_report": f"runs/{run_id}/benchmark-report.json",
            "metrics": f"runs/{run_id}/metrics.prom",
            "summary": f"runs/{run_id}/summary.json",
            "shard_manifest": (f"shards/runs/{run_id}/manifests/{shard_manifest_id}.json"),
        }

    @staticmethod
    def _load_trajectories(path: Path) -> tuple[Trajectory, ...]:
        trajectories = []
        for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
            if not line.strip():
                continue
            try:
                trajectories.append(Trajectory.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid trajectory on line {line_number}: {error}") from error
        if not trajectories:
            raise ValueError("trajectory artifact is empty")
        return tuple(trajectories)

    @staticmethod
    def _validate_linkage(
        manifest: RunArtifactManifest,
        *,
        plan: RolloutPlan,
        run: RunManifest,
        trajectories: tuple[Trajectory, ...],
        report: BenchmarkReport,
        summary: object,
        shard_manifest: TrajectoryShardManifest,
        errors: list[str],
    ) -> None:
        if plan.plan_id != manifest.plan_id:
            errors.append("rollout_plan_id_mismatch")
        if run.run_id != manifest.run_id:
            errors.append("run_manifest_id_mismatch")
        if run.status is RunStatus.RUNNING:
            errors.append("run_manifest_not_terminal")
        if run.metadata.get("shard_manifest_id") != manifest.shard_manifest_id:
            errors.append("run_manifest_shard_id_mismatch")
        if shard_manifest.run_id != manifest.run_id:
            errors.append("shard_manifest_run_id_mismatch")
        if shard_manifest.manifest_id != manifest.shard_manifest_id:
            errors.append("shard_manifest_id_mismatch")
        trajectory_ids = {item.trajectory_id for item in trajectories}
        if len(trajectory_ids) != len(trajectories):
            errors.append("trajectory_artifact_has_duplicate_ids")
        if trajectory_ids != {slot.trajectory_id for slot in plan.slots}:
            errors.append("trajectory_artifact_does_not_match_plan")
        if trajectory_ids != {item.trajectory_id for item in shard_manifest.shards}:
            errors.append("trajectory_artifact_does_not_match_shards")
        if report.run_id != manifest.run_id:
            errors.append("benchmark_report_run_id_mismatch")
        if report.attempt_count != len(trajectories):
            errors.append("benchmark_report_attempt_count_mismatch")
        if not isinstance(summary, dict):
            errors.append("summary_is_not_an_object")
            return
        for field, expected in (
            ("run_id", manifest.run_id),
            ("plan_id", manifest.plan_id),
            ("shard_manifest_id", manifest.shard_manifest_id),
            ("trajectory_count", len(trajectories)),
        ):
            if summary.get(field) != expected:
                errors.append(f"summary_{field}_mismatch")

    def _verify_shards(self, manifest: TrajectoryShardManifest) -> ShardVerification:
        run_path = self.root / "shards" / "runs" / manifest.run_id
        missing = []
        mismatched = []
        verified = 0
        expected_paths = {item.relative_path for item in manifest.shards}
        for shard in manifest.shards:
            path = run_path / shard.relative_path
            if not path.is_file():
                missing.append(shard.trajectory_id)
                continue
            payload = path.read_bytes()
            if (
                len(payload) != shard.size_bytes
                or hashlib.sha256(payload).hexdigest() != shard.content_digest
            ):
                mismatched.append(shard.trajectory_id)
                continue
            trajectory = Trajectory.model_validate_json(payload)
            if trajectory.trajectory_id != shard.trajectory_id:
                mismatched.append(shard.trajectory_id)
                continue
            verified += 1
        shard_path = run_path / "shards"
        actual_paths = (
            {
                path.relative_to(run_path).as_posix()
                for path in shard_path.glob("*.json")
                if path.is_file()
            }
            if shard_path.is_dir()
            else set()
        )
        unexpected = tuple(sorted(actual_paths - expected_paths))
        valid = not missing and not mismatched
        return ShardVerification(
            manifest_id=manifest.manifest_id,
            valid=valid,
            complete=valid and not unexpected,
            verified_count=verified,
            missing_trajectory_ids=tuple(missing),
            mismatched_trajectory_ids=tuple(mismatched),
            unexpected_paths=unexpected,
        )

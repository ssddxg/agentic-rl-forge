from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import orjson

from agentic_rl_forge.contracts import (
    ShardVerification,
    Trajectory,
    TrajectoryShardManifest,
    TrajectoryShardRef,
)


class ShardedTrajectoryStore:
    """Process-safe trajectory shards for shared filesystems and interrupted rollouts."""

    def __init__(self, root: Path | str, *, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = self._validate_identifier(run_id, label="run ID")
        self.run_path = self.root / "runs" / self.run_id
        self.shard_path = self.run_path / "shards"
        self.manifest_path = self.run_path / "manifests"
        self.shard_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path.mkdir(parents=True, exist_ok=True)

    def put(self, trajectory: Trajectory) -> bool:
        target = self._trajectory_path(trajectory.trajectory_id)
        payload = trajectory.canonical_bytes() + b"\n"
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(
                    f"trajectory {trajectory.trajectory_id!r} already exists with other content"
                )
            return False
        return self._link_payload(target, payload, label=trajectory.trajectory_id)

    def get(self, trajectory_id: str) -> Trajectory | None:
        path = self._trajectory_path(trajectory_id)
        if not path.exists():
            return None
        trajectory = Trajectory.model_validate_json(path.read_bytes())
        if trajectory.trajectory_id != trajectory_id:
            raise ValueError(f"trajectory reference {trajectory_id!r} contains another ID")
        return trajectory

    def list_trajectories(self) -> tuple[Trajectory, ...]:
        trajectories = [
            Trajectory.model_validate_json(path.read_bytes())
            for path in sorted(self.shard_path.glob("*.json"))
        ]
        return tuple(sorted(trajectories, key=lambda item: item.trajectory_id))

    def finalize(
        self,
        *,
        expected_policy_version: str | None = None,
        require_single_policy: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> TrajectoryShardManifest:
        trajectories = self.list_trajectories()
        if not trajectories:
            raise ValueError("cannot finalize an empty trajectory run")
        policy_versions = tuple(sorted({item.policy_version for item in trajectories}))
        if require_single_policy and len(policy_versions) != 1:
            raise ValueError("trajectory run contains multiple policy versions")
        if expected_policy_version is not None:
            for trajectory in trajectories:
                trajectory.require_on_policy(expected_policy_version)
        shards = tuple(self._shard_ref(trajectory) for trajectory in trajectories)
        identity = hashlib.sha256()
        identity.update(self.run_id.encode("utf-8"))
        for shard in shards:
            identity.update(shard.trajectory_id.encode("utf-8"))
            identity.update(shard.content_digest.encode("ascii"))
        manifest_metadata = metadata or {}
        identity.update(orjson.dumps(manifest_metadata, option=orjson.OPT_SORT_KEYS))
        manifest = TrajectoryShardManifest(
            manifest_id=f"manifest_{identity.hexdigest()[:24]}",
            run_id=self.run_id,
            created_at=max(
                trajectory.completed_at or trajectory.started_at for trajectory in trajectories
            ),
            policy_versions=policy_versions,
            environment_versions=tuple(sorted({item.environment_version for item in trajectories})),
            trajectory_count=len(trajectories),
            task_count=len({item.task_id for item in trajectories}),
            group_count=len({item.group_id for item in trajectories}),
            shards=shards,
            metadata=manifest_metadata,
        )
        target = self.manifest_path / f"{manifest.manifest_id}.json"
        self._link_payload(target, manifest.canonical_bytes() + b"\n", label=manifest.manifest_id)
        return manifest

    def get_manifest(self, manifest_id: str) -> TrajectoryShardManifest | None:
        safe_id = self._validate_identifier(manifest_id, label="manifest ID")
        path = self.manifest_path / f"{safe_id}.json"
        if not path.exists():
            return None
        return TrajectoryShardManifest.model_validate_json(path.read_bytes())

    def list_manifests(self) -> tuple[TrajectoryShardManifest, ...]:
        manifests = [
            TrajectoryShardManifest.model_validate_json(path.read_bytes())
            for path in self.manifest_path.glob("manifest_*.json")
        ]
        return tuple(
            sorted(
                manifests,
                key=lambda item: (
                    item.trajectory_count,
                    item.created_at,
                    item.manifest_id,
                ),
            )
        )

    def latest_manifest(self) -> TrajectoryShardManifest | None:
        manifests = self.list_manifests()
        return manifests[-1] if manifests else None

    def verify(self, manifest: TrajectoryShardManifest) -> ShardVerification:
        if manifest.run_id != self.run_id:
            raise ValueError("manifest belongs to another run")
        missing = []
        mismatched = []
        verified = 0
        expected_paths = {shard.relative_path for shard in manifest.shards}
        for shard in manifest.shards:
            path = self.run_path / shard.relative_path
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
        actual_paths = {
            path.relative_to(self.run_path).as_posix() for path in self.shard_path.glob("*.json")
        }
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

    def export_jsonl(
        self,
        manifest: TrajectoryShardManifest,
        output: Path,
    ) -> int:
        verification = self.verify(manifest)
        if not verification.valid:
            raise ValueError("cannot export a manifest with missing or mismatched shards")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for shard in manifest.shards:
                    destination.write((self.run_path / shard.relative_path).read_bytes())
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return manifest.trajectory_count

    def _shard_ref(self, trajectory: Trajectory) -> TrajectoryShardRef:
        path = self._trajectory_path(trajectory.trajectory_id)
        payload = path.read_bytes()
        return TrajectoryShardRef(
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            group_id=trajectory.group_id,
            content_digest=hashlib.sha256(payload).hexdigest(),
            relative_path=path.relative_to(self.run_path).as_posix(),
            size_bytes=len(payload),
        )

    def _trajectory_path(self, trajectory_id: str) -> Path:
        digest = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()
        return self.shard_path / f"{digest}.json"

    def _link_payload(self, target: Path, payload: bytes, *, label: str) -> bool:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise ValueError(f"{label!r} already exists with other content") from None
                return False
        finally:
            if temporary.exists():
                temporary.unlink()
        return True

    @staticmethod
    def _validate_identifier(value: str, *, label: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        if (
            not value
            or len(value) > 128
            or value in {".", ".."}
            or any(character not in allowed for character in value)
        ):
            raise ValueError(f"invalid {label}")
        return value

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from agentic_rl_forge.contracts import (
    ArtifactLocation,
    CheckpointManifest,
    CheckpointVerification,
)


class CheckpointRegistry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, manifest: CheckpointManifest) -> bool:
        target = self._manifest_path(manifest.checkpoint_id)
        payload = manifest.canonical_bytes() + b"\n"
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(
                    f"checkpoint {manifest.checkpoint_id!r} already exists with other content"
                )
            return False
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path,
            prefix=".checkpoint-",
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
                    raise ValueError(
                        f"checkpoint {manifest.checkpoint_id!r} already exists with other content"
                    ) from None
                return False
        finally:
            if temporary.exists():
                temporary.unlink()
        return True

    def get(self, checkpoint_id: str) -> CheckpointManifest | None:
        path = self._manifest_path(checkpoint_id)
        if not path.exists():
            return None
        return CheckpointManifest.model_validate_json(path.read_bytes())

    def list(
        self,
        *,
        run_id: str | None = None,
        policy_version: str | None = None,
    ) -> tuple[CheckpointManifest, ...]:
        manifests = []
        for path in self.path.glob("*.json"):
            manifest = CheckpointManifest.model_validate_json(path.read_bytes())
            if run_id is not None and manifest.run_id != run_id:
                continue
            if policy_version is not None and manifest.policy_version != policy_version:
                continue
            manifests.append(manifest)
        return tuple(
            sorted(
                manifests,
                key=lambda item: (item.step, item.created_at, item.checkpoint_id),
            )
        )

    def latest(
        self,
        *,
        run_id: str | None = None,
        policy_version: str | None = None,
    ) -> CheckpointManifest | None:
        manifests = self.list(run_id=run_id, policy_version=policy_version)
        return manifests[-1] if manifests else None

    def verify(
        self,
        manifest: CheckpointManifest,
        *,
        root: Path | None = None,
    ) -> CheckpointVerification:
        verified = []
        missing = []
        mismatched = []
        unverified = []
        for artifact in manifest.artifacts:
            if artifact.location is ArtifactLocation.REMOTE:
                unverified.append(artifact.name)
                continue
            path = Path(artifact.uri)
            if not path.is_absolute() and root is not None:
                path = root / path
            if not path.is_file():
                missing.append(artifact.name)
                continue
            if path.stat().st_size != artifact.size_bytes:
                mismatched.append(artifact.name)
                continue
            digest = self._file_digest(path)
            if digest != artifact.sha256:
                mismatched.append(artifact.name)
                continue
            verified.append(artifact.name)
        valid = not missing and not mismatched
        return CheckpointVerification(
            checkpoint_id=manifest.checkpoint_id,
            valid=valid,
            fully_verified=valid and not unverified,
            verified_artifacts=tuple(verified),
            missing_artifacts=tuple(missing),
            mismatched_artifacts=tuple(mismatched),
            unverified_artifacts=tuple(unverified),
        )

    def _manifest_path(self, checkpoint_id: str) -> Path:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        if (
            not checkpoint_id
            or len(checkpoint_id) > 128
            or checkpoint_id in {".", ".."}
            or any(character not in allowed for character in checkpoint_id)
        ):
            raise ValueError("invalid checkpoint ID")
        return self.path / f"{checkpoint_id}.json"

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from agentic_rl_forge.contracts import (
    TrainerBatchManifest,
    TrainerBatchVerification,
    TrainerTrajectoryRef,
    Trajectory,
)
from agentic_rl_forge.integrations.verl import (
    VerlTrajectoryRecord,
    trajectory_to_verl_record,
    validate_grpo_batch,
)
from agentic_rl_forge.storage.blobs import ConditionalBlobStore


class TrainerBatchExporter:
    def __init__(
        self,
        store: ConditionalBlobStore,
        *,
        prefix: str = "trainer",
    ) -> None:
        self._store = store
        self._prefix = prefix.strip("/")

    def export(
        self,
        trajectories: Sequence[Trajectory],
        *,
        expected_policy_version: str,
        expected_group_size: int,
        source_run_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TrainerBatchManifest:
        validate_grpo_batch(
            trajectories,
            expected_policy_version=expected_policy_version,
            expected_group_size=expected_group_size,
        )
        ordered = tuple(sorted(trajectories, key=lambda item: (item.group_id, item.trajectory_id)))
        records = tuple(trajectory_to_verl_record(item) for item in ordered)
        payload = b"".join(record.canonical_bytes() + b"\n" for record in records)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        identity = hashlib.sha256()
        identity.update(expected_policy_version.encode("utf-8"))
        identity.update(str(expected_group_size).encode("ascii"))
        identity.update(payload_sha256.encode("ascii"))
        batch_id = f"batch_{identity.hexdigest()[:24]}"
        base_key = self._key(f"batches/{batch_id}")
        payload_key = f"{base_key}/trajectories.jsonl"
        self._store.put_if_absent(
            payload_key,
            payload,
            metadata={
                "batch_id": batch_id,
                "policy_version": expected_policy_version,
                "format": "verl_trajectory_jsonl_v1",
            },
        )
        manifest = TrainerBatchManifest(
            batch_id=batch_id,
            created_at=max(
                trajectory.completed_at or trajectory.started_at for trajectory in ordered
            ),
            policy_version=expected_policy_version,
            environment_versions=tuple(
                sorted({trajectory.environment_version for trajectory in ordered})
            ),
            source_run_id=source_run_id,
            group_size=expected_group_size,
            group_count=len({trajectory.group_id for trajectory in ordered}),
            trajectory_count=len(ordered),
            payload_key=payload_key,
            payload_sha256=payload_sha256,
            payload_size_bytes=len(payload),
            trajectories=tuple(
                TrainerTrajectoryRef(
                    trajectory_id=trajectory.trajectory_id,
                    task_id=trajectory.task_id,
                    group_id=trajectory.group_id,
                    content_digest=trajectory.digest(),
                    origin=trajectory.provenance.origin,
                    reward=trajectory.total_reward,
                )
                for trajectory in ordered
            ),
            metadata=metadata or {},
        )
        self._store.put_if_absent(
            f"{base_key}/manifest.json",
            manifest.canonical_bytes() + b"\n",
            metadata={"batch_id": batch_id, "kind": "trainer-batch-manifest"},
        )
        return manifest

    def load_manifest(self, key: str) -> TrainerBatchManifest:
        return TrainerBatchManifest.model_validate_json(self._store.get(key))

    def load_records(
        self,
        manifest: TrainerBatchManifest,
    ) -> tuple[VerlTrajectoryRecord, ...]:
        payload = self._store.get(manifest.payload_key)
        return tuple(
            VerlTrajectoryRecord.model_validate_json(line)
            for line in payload.splitlines()
            if line.strip()
        )

    def verify(self, manifest: TrainerBatchManifest) -> TrainerBatchVerification:
        info = self._store.head(manifest.payload_key)
        if info is None:
            return TrainerBatchVerification(
                batch_id=manifest.batch_id,
                valid=False,
                payload_exists=False,
                payload_digest_matches=False,
                payload_size_matches=False,
                record_count_matches=False,
                trajectory_ids_match=False,
                trajectory_contents_match=False,
            )
        payload = self._store.get(manifest.payload_key)
        digest_matches = hashlib.sha256(payload).hexdigest() == manifest.payload_sha256
        size_matches = len(payload) == manifest.payload_size_bytes
        try:
            records = tuple(
                VerlTrajectoryRecord.model_validate_json(line)
                for line in payload.splitlines()
                if line.strip()
            )
        except ValueError:
            records = ()
        count_matches = len(records) == manifest.trajectory_count
        ids_match = tuple(record.trajectory_id for record in records) == tuple(
            item.trajectory_id for item in manifest.trajectories
        )
        references = {item.trajectory_id: item for item in manifest.trajectories}
        contents_match = count_matches and ids_match
        if contents_match:
            for record in records:
                reference = references[record.trajectory_id]
                try:
                    trajectory = Trajectory.model_validate(record.trajectory)
                except ValueError:
                    contents_match = False
                    break
                if (
                    trajectory.digest() != reference.content_digest
                    or trajectory.task_id != reference.task_id
                    or trajectory.group_id != reference.group_id
                    or trajectory.policy_version != manifest.policy_version
                    or trajectory.provenance.origin is not reference.origin
                    or trajectory.total_reward != reference.reward
                ):
                    contents_match = False
                    break
        valid = digest_matches and size_matches and count_matches and ids_match and contents_match
        return TrainerBatchVerification(
            batch_id=manifest.batch_id,
            valid=valid,
            payload_exists=True,
            payload_digest_matches=digest_matches,
            payload_size_matches=size_matches,
            record_count_matches=count_matches,
            trajectory_ids_match=ids_match,
            trajectory_contents_match=contents_match,
        )

    def _key(self, value: str) -> str:
        return f"{self._prefix}/{value}" if self._prefix else value

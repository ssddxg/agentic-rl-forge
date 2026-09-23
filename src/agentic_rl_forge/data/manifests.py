from __future__ import annotations

import base64
import hashlib
import importlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson

from agentic_rl_forge.contracts import (
    DatasetCollectionManifest,
    DatasetFileManifest,
    DatasetManifest,
    JsonObject,
    ManifestSignature,
    SignedDatasetCollectionManifest,
)


class DatasetManifestBuilder:
    def build_split(
        self,
        paths: Sequence[Path],
        *,
        name: str,
        split: str,
        id_field: str = "task_id",
        root: Path | None = None,
        metadata: JsonObject | None = None,
    ) -> DatasetManifest:
        if not paths:
            raise ValueError("dataset split requires at least one file")
        resolved_root = root.resolve() if root is not None else None
        files = []
        all_ids = []
        content_identity = hashlib.sha256()
        seen_paths: set[str] = set()
        for path in sorted((item.resolve(strict=True) for item in paths), key=str):
            relative_path = self._relative_path(path, resolved_root)
            if relative_path in seen_paths:
                raise ValueError(f"duplicate dataset path {relative_path!r}")
            seen_paths.add(relative_path)
            payload = path.read_bytes()
            file_ids = self._jsonl_ids(payload, id_field=id_field, path=path)
            counts = Counter(file_ids)
            unique_ids = tuple(sorted(counts))
            file_digest = hashlib.sha256(payload).hexdigest()
            files.append(
                DatasetFileManifest(
                    relative_path=relative_path,
                    size_bytes=len(payload),
                    sha256=file_digest,
                    record_count=len(file_ids),
                    unique_id_count=len(unique_ids),
                    duplicate_id_count=len(file_ids) - len(unique_ids),
                    record_ids_digest=self._ids_digest(unique_ids),
                )
            )
            all_ids.extend(file_ids)
            content_identity.update(relative_path.encode("utf-8"))
            content_identity.update(file_digest.encode("ascii"))
        counts = Counter(all_ids)
        record_ids = tuple(sorted(counts))
        record_ids_digest = self._ids_digest(record_ids)
        content_digest = content_identity.hexdigest()
        manifest_identity = hashlib.sha256()
        manifest_identity.update(name.encode("utf-8"))
        manifest_identity.update(split.encode("utf-8"))
        manifest_identity.update(id_field.encode("utf-8"))
        manifest_identity.update(content_digest.encode("ascii"))
        manifest_identity.update(record_ids_digest.encode("ascii"))
        return DatasetManifest(
            manifest_id=f"dataset_{manifest_identity.hexdigest()[:24]}",
            name=name,
            split=split,
            format="jsonl",
            id_field=id_field,
            files=tuple(files),
            record_count=len(all_ids),
            unique_id_count=len(record_ids),
            duplicate_id_count=len(all_ids) - len(record_ids),
            record_ids_digest=record_ids_digest,
            record_ids=record_ids,
            content_digest=content_digest,
            metadata=metadata or {},
        )

    def build_collection(
        self,
        splits: Mapping[str, Sequence[Path]],
        *,
        name: str,
        id_field: str = "task_id",
        root: Path | None = None,
        metadata: JsonObject | None = None,
    ) -> DatasetCollectionManifest:
        if not splits:
            raise ValueError("dataset collection requires at least one split")
        manifests = {
            split: self.build_split(
                paths,
                name=name,
                split=split,
                id_field=id_field,
                root=root,
            )
            for split, paths in sorted(splits.items())
        }
        owners: dict[str, str] = {}
        overlaps = []
        for split, manifest in manifests.items():
            for record_id in manifest.record_ids:
                previous = owners.setdefault(record_id, split)
                if previous != split:
                    overlaps.append((record_id, previous, split))
        if overlaps:
            preview = ", ".join(
                f"{record_id} ({left}/{right})" for record_id, left, right in overlaps[:10]
            )
            raise ValueError(f"dataset splits overlap: {preview}")
        all_ids = tuple(sorted(owners))
        identity = hashlib.sha256()
        identity.update(name.encode("utf-8"))
        for split, manifest in manifests.items():
            identity.update(split.encode("utf-8"))
            identity.update(manifest.digest().encode("ascii"))
        return DatasetCollectionManifest(
            collection_id=f"collection_{identity.hexdigest()[:24]}",
            name=name,
            splits=manifests,
            total_record_count=sum(item.record_count for item in manifests.values()),
            total_unique_id_count=len(all_ids),
            record_ids_digest=self._ids_digest(all_ids),
            metadata=metadata or {},
        )

    @staticmethod
    def _jsonl_ids(payload: bytes, *, id_field: str, path: Path) -> list[str]:
        ids = []
        fields = id_field.split(".")
        for line_number, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                continue
            value: Any = orjson.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            for field in fields:
                if not isinstance(value, dict) or field not in value:
                    raise ValueError(f"{path}:{line_number} does not contain ID field {id_field!r}")
                value = value[field]
            if not isinstance(value, str | int):
                raise ValueError(f"{path}:{line_number} has a non-scalar record ID")
            ids.append(str(value))
        return ids

    @staticmethod
    def _relative_path(path: Path, root: Path | None) -> str:
        if root is None:
            return path.name
        try:
            return path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"dataset file {path} is outside manifest root {root}") from error

    @staticmethod
    def _ids_digest(record_ids: Sequence[str]) -> str:
        digest = hashlib.sha256()
        for record_id in record_ids:
            digest.update(record_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


class Ed25519ManifestSigner:
    def __init__(self, private_key: Any) -> None:
        self._private_key = private_key

    @classmethod
    def generate(cls) -> Ed25519ManifestSigner:
        ed25519, _ = cls._modules()
        return cls(ed25519.Ed25519PrivateKey.generate())

    @classmethod
    def from_private_key_base64(cls, value: str) -> Ed25519ManifestSigner:
        ed25519, _ = cls._modules()
        raw = base64.b64decode(value, validate=True)
        return cls(ed25519.Ed25519PrivateKey.from_private_bytes(raw))

    @property
    def private_key_base64(self) -> str:
        _, serialization = self._modules()
        raw = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return base64.b64encode(raw).decode("ascii")

    @property
    def public_key_base64(self) -> str:
        _, serialization = self._modules()
        raw = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw).decode("ascii")

    @property
    def key_id(self) -> str:
        return self.public_key_id(self.public_key_base64)

    @staticmethod
    def public_key_id(public_key_base64: str) -> str:
        raw = base64.b64decode(public_key_base64, validate=True)
        if len(raw) != 32:
            raise ValueError("Ed25519 public key must contain 32 bytes")
        return f"ed25519_{hashlib.sha256(raw).hexdigest()[:24]}"

    def sign_payload(self, payload: bytes) -> ManifestSignature:
        signature = self._private_key.sign(payload)
        return ManifestSignature(
            public_key_base64=self.public_key_base64,
            signature_base64=base64.b64encode(signature).decode("ascii"),
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )

    @classmethod
    def verify_payload(cls, signature: ManifestSignature, payload: bytes) -> bool:
        ed25519, _ = cls._modules()
        if hashlib.sha256(payload).hexdigest() != signature.payload_sha256:
            return False
        try:
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                base64.b64decode(signature.public_key_base64, validate=True)
            )
            public_key.verify(
                base64.b64decode(signature.signature_base64, validate=True),
                payload,
            )
        except Exception:
            return False
        return True

    def sign(
        self,
        manifest: DatasetCollectionManifest,
    ) -> SignedDatasetCollectionManifest:
        payload = manifest.canonical_bytes()
        return SignedDatasetCollectionManifest(
            manifest=manifest,
            signature=self.sign_payload(payload),
        )

    @classmethod
    def verify(cls, signed: SignedDatasetCollectionManifest) -> bool:
        payload = signed.manifest.canonical_bytes()
        return cls.verify_payload(signed.signature, payload)

    @staticmethod
    def _modules() -> tuple[Any, Any]:
        try:
            ed25519 = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ed25519")
            serialization = importlib.import_module("cryptography.hazmat.primitives.serialization")
        except ImportError as error:
            raise RuntimeError("install agentic-rl-forge[signing] for Ed25519 manifests") from error
        return ed25519, serialization

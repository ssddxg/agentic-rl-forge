from __future__ import annotations

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, JsonObject


class DatasetFileManifest(ContractModel):
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=0)
    unique_id_count: int = Field(ge=0)
    duplicate_id_count: int = Field(ge=0)
    record_ids_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(ContractModel):
    manifest_id: str = Field(pattern=r"^dataset_[0-9a-f]{24}$")
    name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    format: str
    id_field: str = Field(min_length=1)
    files: tuple[DatasetFileManifest, ...] = Field(min_length=1)
    record_count: int = Field(ge=0)
    unique_id_count: int = Field(ge=0)
    duplicate_id_count: int = Field(ge=0)
    record_ids_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_ids: tuple[str, ...]
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> DatasetManifest:
        if self.record_count != sum(item.record_count for item in self.files):
            raise ValueError("record_count must match file records")
        if self.unique_id_count != len(self.record_ids):
            raise ValueError("unique_id_count must match record_ids")
        if tuple(sorted(set(self.record_ids))) != self.record_ids:
            raise ValueError("record_ids must be sorted and unique")
        if self.duplicate_id_count != self.record_count - self.unique_id_count:
            raise ValueError("duplicate_id_count is inconsistent")
        return self


class DatasetCollectionManifest(ContractModel):
    collection_id: str = Field(pattern=r"^collection_[0-9a-f]{24}$")
    name: str = Field(min_length=1)
    splits: dict[str, DatasetManifest] = Field(min_length=1)
    total_record_count: int = Field(ge=0)
    total_unique_id_count: int = Field(ge=0)
    record_ids_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_collection(self) -> DatasetCollectionManifest:
        if any(name != manifest.split for name, manifest in self.splits.items()):
            raise ValueError("split keys must match embedded manifest split names")
        if self.total_record_count != sum(
            manifest.record_count for manifest in self.splits.values()
        ):
            raise ValueError("total_record_count must match split records")
        all_ids = [
            record_id for manifest in self.splits.values() for record_id in manifest.record_ids
        ]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("dataset splits contain overlapping record IDs")
        if self.total_unique_id_count != len(all_ids):
            raise ValueError("total_unique_id_count must match split IDs")
        return self


class ManifestSignature(ContractModel):
    algorithm: str = "ed25519"
    public_key_base64: str = Field(min_length=1)
    signature_base64: str = Field(min_length=1)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SignedDatasetCollectionManifest(ContractModel):
    manifest: DatasetCollectionManifest
    signature: ManifestSignature

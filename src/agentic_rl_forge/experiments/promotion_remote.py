from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Collection
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import httpx
import orjson
from pydantic import Field, model_validator

from agentic_rl_forge.contracts import ContractModel
from agentic_rl_forge.data.manifests import Ed25519ManifestSigner
from agentic_rl_forge.experiments.promotion import (
    ExperimentPromotionArtifactRef,
    ExperimentPromotionArtifactScope,
    ExperimentReproducibilityManifest,
)
from agentic_rl_forge.experiments.promotion_archives import (
    ExperimentPromotionArchive,
    ExperimentPromotionArchiveAttestationVerification,
    ExperimentPromotionArchiveAttestor,
    ExperimentPromotionArchiveReceipt,
    SignedExperimentPromotionArchiveAttestation,
)
from agentic_rl_forge.storage.blobs import BlobConflictError, LocalBlobStore

_AUTHORITY = re.compile(r"^[A-Za-z0-9.-]+(?::[0-9]{1,5})?$")
_BUCKET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,62}$")
_SIGNER_KEY_ID = re.compile(r"^ed25519_[0-9a-f]{24}$")


class ExperimentPromotionRemoteFetchError(ValueError):
    pass


class ExperimentPromotionRemoteProvider(str, Enum):
    HTTPS = "https"
    S3 = "s3"
    GS = "gs"
    AZ = "az"


class ExperimentPromotionRemotePolicy(ContractModel):
    allowed_https_authorities: tuple[str, ...] = ()
    allowed_s3_buckets: tuple[str, ...] = ()
    chunk_size_bytes: int = Field(default=8 * 1024 * 1024, ge=64 * 1024, le=64 * 1024 * 1024)
    max_artifact_bytes: int = Field(default=1 << 39, ge=1)
    max_total_bytes: int = Field(default=1 << 40, ge=1)
    require_source_validator: bool = True

    @model_validator(mode="after")
    def validate_policy(self) -> ExperimentPromotionRemotePolicy:
        authorities = tuple(sorted(set(self.allowed_https_authorities)))
        buckets = tuple(sorted(set(self.allowed_s3_buckets)))
        if self.allowed_https_authorities != authorities or any(
            _AUTHORITY.fullmatch(item) is None or item != item.casefold() for item in authorities
        ):
            raise ValueError("HTTPS authorities must be normalized, sorted, and unique")
        if self.allowed_s3_buckets != buckets or any(
            _BUCKET.fullmatch(item) is None or item != item.casefold() for item in buckets
        ):
            raise ValueError("S3 buckets must be normalized, sorted, and unique")
        return self


class ExperimentPromotionRemoteSource(ContractModel):
    provider: str = Field(pattern=r"^(https|s3)$")
    provider_identity: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    etag: str | None = None
    version_id: str | None = None
    last_modified: str | None = None

    @property
    def has_validator(self) -> bool:
        return any((self.etag, self.version_id, self.last_modified))


class ExperimentPromotionRemoteArtifactPlan(ContractModel):
    reference: ExperimentPromotionArtifactRef
    artifact_id: str = Field(pattern=r"^promotion_remote_artifact_[0-9a-f]{24}$")
    provider: str = Field(pattern=r"^(https|s3|gs|az)$")
    source: ExperimentPromotionRemoteSource | None = None
    chunk_count: int = Field(ge=0)
    cache_prefix: str = Field(min_length=1)
    issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_artifact(self) -> ExperimentPromotionRemoteArtifactPlan:
        if self.reference.scope is not ExperimentPromotionArtifactScope.REMOTE:
            raise ValueError("remote fetch artifact must use remote scope")
        expected_provider = urlsplit(self.reference.locator).scheme
        if self.provider != expected_provider:
            raise ValueError("remote fetch provider does not match the artifact URI")
        expected_id = self.expected_artifact_id(self.reference)
        if self.artifact_id != expected_id:
            raise ValueError("remote fetch artifact ID does not match its reference")
        if self.cache_prefix != f"artifacts/{self.artifact_id}":
            raise ValueError("remote fetch cache prefix is inconsistent")
        if self.issues != tuple(sorted(set(self.issues))):
            raise ValueError("remote fetch artifact issues must be sorted and unique")
        if self.source is not None:
            if self.source.provider != self.provider:
                raise ValueError("remote source provider does not match its artifact")
            if self.chunk_count < 0:
                raise ValueError("remote source chunk count is invalid")
        elif self.chunk_count != 0:
            raise ValueError("unresolved remote artifact cannot declare chunks")
        if not self.issues and self.source is None:
            raise ValueError("ready remote artifact requires source evidence")
        return self

    @staticmethod
    def expected_artifact_id(reference: ExperimentPromotionArtifactRef) -> str:
        digest = hashlib.sha256(reference.canonical_bytes()).hexdigest()
        return f"promotion_remote_artifact_{digest[:24]}"


class ExperimentPromotionRemoteCheck(ContractModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")
    passed: bool
    detail: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_check(self) -> ExperimentPromotionRemoteCheck:
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise ValueError("remote fetch check evidence must be sorted and unique")
        return self


class ExperimentPromotionRemoteFetchPlan(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    plan_id: str = Field(pattern=r"^promotion_remote_fetch_plan_[0-9a-f]{24}$")
    receipt: ExperimentPromotionArchiveReceipt
    attestation: SignedExperimentPromotionArchiveAttestation
    attestation_verification: ExperimentPromotionArchiveAttestationVerification
    trusted_signer_key_ids: tuple[str, ...]
    manifest: ExperimentReproducibilityManifest
    cache_root: str = Field(min_length=1)
    policy: ExperimentPromotionRemotePolicy
    artifacts: tuple[ExperimentPromotionRemoteArtifactPlan, ...]
    remote_artifact_count: int = Field(ge=0)
    total_size_bytes: int = Field(ge=0)
    checks: tuple[ExperimentPromotionRemoteCheck, ...]
    failed_check_count: int = Field(ge=0)
    eligible: bool

    @model_validator(mode="after")
    def validate_plan(self) -> ExperimentPromotionRemoteFetchPlan:
        root = Path(self.cache_root)
        if not root.is_absolute() or root.as_posix() != self.cache_root:
            raise ValueError("remote fetch cache root must be a normalized absolute path")
        _validate_trust_evidence(
            self.receipt,
            self.attestation,
            self.attestation_verification,
            self.trusted_signer_key_ids,
        )
        if (
            self.receipt.promotion_name != self.manifest.promotion_name
            or self.receipt.reproducibility_manifest_id != self.manifest.manifest_id
        ):
            raise ValueError("remote fetch archive receipt does not match its manifest")
        remote_refs = tuple(
            item
            for item in self.manifest.artifacts
            if item.scope is ExperimentPromotionArtifactScope.REMOTE
        )
        if tuple(item.reference for item in self.artifacts) != remote_refs:
            raise ValueError("remote fetch artifacts do not cover the remote manifest references")
        for item in self.artifacts:
            expected_chunks = (
                math.ceil(item.reference.size_bytes / self.policy.chunk_size_bytes)
                if item.source is not None
                else 0
            )
            if item.chunk_count != expected_chunks:
                raise ValueError("remote fetch artifact chunk count is inconsistent")
        if self.remote_artifact_count != len(self.artifacts):
            raise ValueError("remote fetch artifact count is inconsistent")
        expected_total = sum(item.reference.size_bytes for item in self.artifacts)
        if self.total_size_bytes != expected_total:
            raise ValueError("remote fetch byte count is inconsistent")
        expected_checks = self.expected_checks(
            verification=self.attestation_verification,
            policy=self.policy,
            artifacts=self.artifacts,
            total_size_bytes=self.total_size_bytes,
        )
        if self.checks != expected_checks:
            raise ValueError("remote fetch checks do not match their evidence")
        failed = sum(not item.passed for item in self.checks)
        if self.failed_check_count != failed or self.eligible != (failed == 0):
            raise ValueError("remote fetch plan eligibility is inconsistent")
        expected_id = self.expected_plan_id(
            receipt=self.receipt,
            attestation=self.attestation,
            trusted_signer_key_ids=self.trusted_signer_key_ids,
            manifest=self.manifest,
            cache_root=self.cache_root,
            policy=self.policy,
            artifacts=self.artifacts,
            checks=self.checks,
        )
        if self.plan_id != expected_id:
            raise ValueError("remote fetch plan ID does not match its contents")
        return self

    @staticmethod
    def expected_checks(
        *,
        verification: ExperimentPromotionArchiveAttestationVerification,
        policy: ExperimentPromotionRemotePolicy,
        artifacts: tuple[ExperimentPromotionRemoteArtifactPlan, ...],
        total_size_bytes: int,
    ) -> tuple[ExperimentPromotionRemoteCheck, ...]:
        def evidence_for(prefix: str) -> tuple[str, ...]:
            return tuple(
                sorted(
                    item.reference.locator
                    for item in artifacts
                    if any(
                        issue == prefix or issue.startswith(f"{prefix}:") for issue in item.issues
                    )
                )
            )

        unsupported = evidence_for("unsupported_provider")
        unauthorized = evidence_for("unauthorized")
        unavailable = evidence_for("unavailable")
        mismatched = evidence_for("size_mismatch")
        missing_validator = evidence_for("missing_validator")
        oversized = evidence_for("artifact_too_large")
        checks = (
            ExperimentPromotionRemoteCheck(
                code="archive.signature",
                passed=(
                    verification.archive_receipt_matches
                    and verification.signer_identity_valid
                    and verification.payload_digest_valid
                    and verification.signature_valid
                ),
                detail="publisher attestation cryptographically matches the archive receipt",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionRemoteCheck(
                code="archive.trusted",
                passed=verification.trusted_signer,
                detail="publisher key belongs to the supplied remote-fetch trust set",
                evidence=(verification.attestation_id,),
            ),
            ExperimentPromotionRemoteCheck(
                code="sources.supported",
                passed=not unsupported,
                detail="every remote URI uses an implemented transport",
                evidence=unsupported,
            ),
            ExperimentPromotionRemoteCheck(
                code="sources.authorized",
                passed=not unauthorized,
                detail="every remote source is covered by an explicit authority allowlist",
                evidence=unauthorized,
            ),
            ExperimentPromotionRemoteCheck(
                code="sources.available",
                passed=not unavailable,
                detail="every authorized remote source exposes readable metadata",
                evidence=unavailable,
            ),
            ExperimentPromotionRemoteCheck(
                code="sources.size",
                passed=not mismatched,
                detail="remote source sizes match the signed promotion manifest",
                evidence=mismatched,
            ),
            ExperimentPromotionRemoteCheck(
                code="sources.validator",
                passed=not policy.require_source_validator or not missing_validator,
                detail="remote sources expose validators for conditional range reads",
                evidence=missing_validator,
            ),
            ExperimentPromotionRemoteCheck(
                code="limits.artifact",
                passed=not oversized,
                detail="every remote artifact fits within the per-artifact byte limit",
                evidence=oversized,
            ),
            ExperimentPromotionRemoteCheck(
                code="limits.total",
                passed=total_size_bytes <= policy.max_total_bytes,
                detail="the complete remote graph fits within the approved byte limit",
                evidence=tuple(sorted({str(total_size_bytes), str(policy.max_total_bytes)})),
            ),
        )
        return tuple(sorted(checks, key=lambda item: item.code))

    @staticmethod
    def expected_plan_id(
        *,
        receipt: ExperimentPromotionArchiveReceipt,
        attestation: SignedExperimentPromotionArchiveAttestation,
        trusted_signer_key_ids: tuple[str, ...],
        manifest: ExperimentReproducibilityManifest,
        cache_root: str,
        policy: ExperimentPromotionRemotePolicy,
        artifacts: tuple[ExperimentPromotionRemoteArtifactPlan, ...],
        checks: tuple[ExperimentPromotionRemoteCheck, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "receipt": receipt.model_dump(mode="json"),
                "attestation_id": attestation.attestation.attestation_id,
                "trusted_signer_key_ids": trusted_signer_key_ids,
                "manifest_id": manifest.manifest_id,
                "cache_root": cache_root,
                "policy": policy.model_dump(mode="json"),
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "checks": [item.model_dump(mode="json") for item in checks],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"promotion_remote_fetch_plan_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionRemoteChunk(ContractModel):
    index: int = Field(ge=0)
    offset: int = Field(ge=0)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    key: str = Field(min_length=1)


class ExperimentPromotionRemoteArtifactReceipt(ContractModel):
    receipt_id: str = Field(pattern=r"^promotion_remote_receipt_[0-9a-f]{24}$")
    artifact: ExperimentPromotionRemoteArtifactPlan
    chunks: tuple[ExperimentPromotionRemoteChunk, ...]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> ExperimentPromotionRemoteArtifactReceipt:
        if self.artifact.issues or self.artifact.source is None:
            raise ValueError("remote artifact receipt requires a ready plan artifact")
        expected_chunks = tuple(range(len(self.chunks)))
        if tuple(item.index for item in self.chunks) != expected_chunks:
            raise ValueError("remote artifact receipt chunks must be contiguous")
        offset = 0
        for chunk in self.chunks:
            if chunk.offset != offset:
                raise ValueError("remote artifact receipt chunk offsets are not contiguous")
            expected_key = f"{self.artifact.cache_prefix}/chunks/{chunk.index:08d}.bin"
            if chunk.key != expected_key:
                raise ValueError("remote artifact receipt chunk key is inconsistent")
            offset += chunk.size_bytes
        if offset != self.size_bytes or self.size_bytes != self.artifact.reference.size_bytes:
            raise ValueError("remote artifact receipt size is inconsistent")
        if self.sha256 != self.artifact.reference.sha256:
            raise ValueError("remote artifact receipt digest does not match the manifest")
        expected_key = f"{self.artifact.cache_prefix}/receipt.json"
        if self.receipt_key != expected_key:
            raise ValueError("remote artifact receipt key is inconsistent")
        expected_id = self.expected_receipt_id(self.artifact, self.chunks)
        if self.receipt_id != expected_id:
            raise ValueError("remote artifact receipt ID does not match its evidence")
        return self

    @staticmethod
    def expected_receipt_id(
        artifact: ExperimentPromotionRemoteArtifactPlan,
        chunks: tuple[ExperimentPromotionRemoteChunk, ...],
    ) -> str:
        payload = orjson.dumps(
            {
                "artifact": artifact.model_dump(mode="json"),
                "chunks": [item.model_dump(mode="json") for item in chunks],
            },
            option=orjson.OPT_SORT_KEYS,
        )
        return f"promotion_remote_receipt_{hashlib.sha256(payload).hexdigest()[:24]}"


class ExperimentPromotionRemoteFetchRecord(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    fetch_id: str = Field(pattern=r"^promotion_remote_fetch_[0-9a-f]{24}$")
    plan: ExperimentPromotionRemoteFetchPlan
    artifact_receipts: tuple[ExperimentPromotionRemoteArtifactReceipt, ...]
    fetched_artifact_count: int = Field(ge=0)
    fetched_chunk_count: int = Field(ge=0)
    fetched_size_bytes: int = Field(ge=0)
    record_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_record(self) -> ExperimentPromotionRemoteFetchRecord:
        if not self.plan.eligible:
            raise ValueError("ineligible remote fetch plan cannot become a record")
        expected_fetch_id = self.expected_fetch_id(self.plan.plan_id)
        if self.fetch_id != expected_fetch_id:
            raise ValueError("remote fetch ID does not match its plan")
        if tuple(item.artifact for item in self.artifact_receipts) != self.plan.artifacts:
            raise ValueError("remote fetch receipts do not cover the plan artifacts")
        for receipt in self.artifact_receipts:
            for chunk in receipt.chunks[:-1]:
                if chunk.size_bytes != self.plan.policy.chunk_size_bytes:
                    raise ValueError("remote fetch receipt chunk size is inconsistent")
            if receipt.chunks and (
                receipt.chunks[-1].size_bytes > self.plan.policy.chunk_size_bytes
            ):
                raise ValueError("remote fetch final chunk exceeds the plan limit")
        if (
            self.fetched_artifact_count != len(self.artifact_receipts)
            or self.fetched_chunk_count != sum(len(item.chunks) for item in self.artifact_receipts)
            or self.fetched_size_bytes != sum(item.size_bytes for item in self.artifact_receipts)
        ):
            raise ValueError("remote fetch record counts are inconsistent")
        if self.record_key != f"promotion-fetches/{self.fetch_id}/record.json":
            raise ValueError("remote fetch record key is inconsistent")
        return self

    @staticmethod
    def expected_fetch_id(plan_id: str) -> str:
        digest = hashlib.sha256(plan_id.encode("ascii")).hexdigest()
        return f"promotion_remote_fetch_{digest[:24]}"


class ExperimentPromotionRemoteReader(Protocol):
    def inspect(self, locator: str) -> ExperimentPromotionRemoteSource: ...

    def read_range(
        self,
        locator: str,
        *,
        start: int,
        end: int,
        source: ExperimentPromotionRemoteSource,
    ) -> bytes: ...


class ExperimentPromotionRemoteReaderRouter:
    def __init__(
        self,
        *,
        allowed_https_authorities: Collection[str],
        allowed_s3_buckets: Collection[str],
        s3_endpoint_url: str | None = None,
        s3_region_name: str | None = None,
        timeout_seconds: float = 60.0,
        http_client: httpx.Client | None = None,
        s3_client: Any | None = None,
    ) -> None:
        self.allowed_https_authorities = frozenset(allowed_https_authorities)
        self.allowed_s3_buckets = frozenset(allowed_s3_buckets)
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_region_name = s3_region_name
        self._http_client = http_client or httpx.Client(
            follow_redirects=False,
            timeout=timeout_seconds,
            headers={"Accept-Encoding": "identity"},
        )
        self._owns_http_client = http_client is None
        self._s3_client = s3_client

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> ExperimentPromotionRemoteReaderRouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def inspect(self, locator: str) -> ExperimentPromotionRemoteSource:
        parsed = urlsplit(locator)
        if parsed.scheme == ExperimentPromotionRemoteProvider.HTTPS:
            authority = parsed.netloc.casefold()
            if authority not in self.allowed_https_authorities:
                raise ExperimentPromotionRemoteFetchError("HTTPS authority is not allowlisted")
            response = self._http_client.head(locator)
            self._validate_http_response(response, expected_status=200)
            content_length = response.headers.get("content-length")
            if content_length is None:
                raise ExperimentPromotionRemoteFetchError(
                    "HTTPS source does not declare Content-Length"
                )
            return ExperimentPromotionRemoteSource(
                provider=ExperimentPromotionRemoteProvider.HTTPS,
                provider_identity=authority,
                size_bytes=int(content_length),
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
            )
        if parsed.scheme == ExperimentPromotionRemoteProvider.S3:
            bucket, key = _s3_location(locator)
            if bucket not in self.allowed_s3_buckets:
                raise ExperimentPromotionRemoteFetchError("S3 bucket is not allowlisted")
            response = self._s3().head_object(Bucket=bucket, Key=key)
            modified = response.get("LastModified")
            return ExperimentPromotionRemoteSource(
                provider=ExperimentPromotionRemoteProvider.S3,
                provider_identity=_s3_provider_identity(bucket, self.s3_endpoint_url),
                size_bytes=int(response["ContentLength"]),
                etag=_clean_etag(response.get("ETag")),
                version_id=_optional_string(response.get("VersionId")),
                last_modified=(modified.isoformat() if isinstance(modified, datetime) else None),
            )
        raise ExperimentPromotionRemoteFetchError("remote provider is not implemented")

    def read_range(
        self,
        locator: str,
        *,
        start: int,
        end: int,
        source: ExperimentPromotionRemoteSource,
    ) -> bytes:
        if start < 0 or end < start:
            raise ValueError("remote byte range is invalid")
        parsed = urlsplit(locator)
        expected_size = end - start + 1
        if parsed.scheme == ExperimentPromotionRemoteProvider.HTTPS:
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
            if source.etag is not None:
                headers["If-Match"] = source.etag
            elif source.last_modified is not None:
                headers["If-Unmodified-Since"] = source.last_modified
            response = self._http_client.get(locator, headers=headers)
            full_object = start == 0 and expected_size == source.size_bytes
            self._validate_http_response(
                response,
                expected_status=200 if full_object and response.status_code == 200 else 206,
            )
            if response.status_code == 206:
                expected_range = f"bytes {start}-{end}/{source.size_bytes}"
                if response.headers.get("content-range") != expected_range:
                    raise ExperimentPromotionRemoteFetchError(
                        "HTTPS source returned an unexpected Content-Range"
                    )
            payload = response.content
        elif parsed.scheme == ExperimentPromotionRemoteProvider.S3:
            bucket, key = _s3_location(locator)
            request: dict[str, object] = {
                "Bucket": bucket,
                "Key": key,
                "Range": f"bytes={start}-{end}",
            }
            if source.version_id is not None:
                request["VersionId"] = source.version_id
            elif source.etag is not None:
                request["IfMatch"] = source.etag
            response = self._s3().get_object(**request)
            body = response["Body"]
            try:
                payload = bytes(body.read())
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
        else:
            raise ExperimentPromotionRemoteFetchError("remote provider is not implemented")
        if len(payload) != expected_size:
            raise ExperimentPromotionRemoteFetchError("remote byte range has an unexpected size")
        return payload

    @staticmethod
    def _validate_http_response(response: httpx.Response, *, expected_status: int) -> None:
        if 300 <= response.status_code < 400:
            raise ExperimentPromotionRemoteFetchError("HTTPS redirects are not allowed")
        if response.status_code != expected_status:
            raise ExperimentPromotionRemoteFetchError(
                f"HTTPS source returned status {response.status_code}"
            )
        encoding = response.headers.get("content-encoding")
        if encoding not in {None, "", "identity"}:
            raise ExperimentPromotionRemoteFetchError(
                "HTTPS source returned content encoding despite identity request"
            )

    def _s3(self) -> Any:
        if self._s3_client is not None:
            return self._s3_client
        try:
            boto3 = importlib.import_module("boto3")
        except ImportError as error:
            raise ExperimentPromotionRemoteFetchError(
                "install agentic-rl-forge[object-store] for S3 remote fetches"
            ) from error
        self._s3_client = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            region_name=self.s3_region_name,
        )
        return self._s3_client


class ExperimentPromotionRemoteFetcher:
    def __init__(self, reader: ExperimentPromotionRemoteReader) -> None:
        self.reader = reader

    def preview(
        self,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        cache_root: Path | str,
        *,
        trusted_public_keys: Collection[str],
        policy: ExperimentPromotionRemotePolicy,
    ) -> ExperimentPromotionRemoteFetchPlan:
        cache = self._cache_root(cache_root)
        receipt, promotion = ExperimentPromotionArchive.inspect_record(
            archive,
            expected_sha256=attestation.attestation.receipt.content_digest,
        )
        trusted_keys = tuple(sorted(set(trusted_public_keys)))
        trusted_ids = tuple(
            sorted(Ed25519ManifestSigner.public_key_id(item) for item in trusted_keys)
        )
        verification = ExperimentPromotionArchiveAttestor.verify(
            attestation,
            receipt,
            trusted_public_keys=trusted_keys,
        )
        manifest = promotion.preview.manifest
        artifacts = tuple(
            self._inspect_artifact(reference, policy)
            for reference in manifest.artifacts
            if reference.scope is ExperimentPromotionArtifactScope.REMOTE
        )
        total = sum(item.reference.size_bytes for item in artifacts)
        checks = ExperimentPromotionRemoteFetchPlan.expected_checks(
            verification=verification,
            policy=policy,
            artifacts=artifacts,
            total_size_bytes=total,
        )
        plan_id = ExperimentPromotionRemoteFetchPlan.expected_plan_id(
            receipt=receipt,
            attestation=attestation,
            trusted_signer_key_ids=trusted_ids,
            manifest=manifest,
            cache_root=cache.as_posix(),
            policy=policy,
            artifacts=artifacts,
            checks=checks,
        )
        failed = sum(not item.passed for item in checks)
        return ExperimentPromotionRemoteFetchPlan(
            plan_id=plan_id,
            receipt=receipt,
            attestation=attestation,
            attestation_verification=verification,
            trusted_signer_key_ids=trusted_ids,
            manifest=manifest,
            cache_root=cache.as_posix(),
            policy=policy,
            artifacts=artifacts,
            remote_artifact_count=len(artifacts),
            total_size_bytes=total,
            checks=checks,
            failed_check_count=failed,
            eligible=failed == 0,
        )

    def execute(
        self,
        plan: ExperimentPromotionRemoteFetchPlan,
        archive: Path | str,
        attestation: SignedExperimentPromotionArchiveAttestation,
        *,
        trusted_public_keys: Collection[str],
        confirm_plan_id: str,
    ) -> ExperimentPromotionRemoteFetchRecord:
        if confirm_plan_id.strip() != plan.plan_id:
            raise ExperimentPromotionRemoteFetchError(
                "remote fetch confirmation does not match the plan ID"
            )
        current = self.preview(
            archive,
            attestation,
            plan.cache_root,
            trusted_public_keys=trusted_public_keys,
            policy=plan.policy,
        )
        if current != plan:
            raise ExperimentPromotionRemoteFetchError("remote fetch evidence changed after preview")
        if not plan.eligible:
            raise ExperimentPromotionRemoteFetchError("remote fetch plan is not eligible")
        cache = self._prepare_cache(plan.cache_root)
        fetch_id = ExperimentPromotionRemoteFetchRecord.expected_fetch_id(plan.plan_id)
        prefix = cache / "promotion-fetches" / fetch_id
        expected_paths = {"record.json"}
        for artifact in plan.artifacts:
            expected_paths.add(f"{artifact.cache_prefix}/receipt.json")
            expected_paths.update(
                f"{artifact.cache_prefix}/chunks/{index:08d}.bin"
                for index in range(artifact.chunk_count)
            )
        self._ensure_safe_prefix(cache, prefix)
        self._validate_cache_prefix(prefix, expected_paths)
        store = LocalBlobStore(prefix)
        receipts = tuple(
            self._fetch_artifact(store, item, plan.policy.chunk_size_bytes)
            for item in plan.artifacts
        )
        record = ExperimentPromotionRemoteFetchRecord(
            fetch_id=fetch_id,
            plan=plan,
            artifact_receipts=receipts,
            fetched_artifact_count=len(receipts),
            fetched_chunk_count=sum(len(item.chunks) for item in receipts),
            fetched_size_bytes=sum(item.size_bytes for item in receipts),
            record_key=f"promotion-fetches/{fetch_id}/record.json",
        )
        try:
            store.put_if_absent("record.json", record.canonical_bytes() + b"\n")
        except BlobConflictError as error:
            raise ExperimentPromotionRemoteFetchError(
                "remote fetch record conflicts with existing evidence"
            ) from error
        self.verify_record(cache / record.record_key)
        return record

    @staticmethod
    def verify_record(path: Path | str) -> ExperimentPromotionRemoteFetchRecord:
        record_path = Path(path)
        if record_path.is_symlink() or not record_path.is_file():
            raise ExperimentPromotionRemoteFetchError("remote fetch record must be a regular file")
        payload = record_path.read_bytes()
        try:
            record = ExperimentPromotionRemoteFetchRecord.model_validate_json(payload)
        except ValueError as error:
            raise ExperimentPromotionRemoteFetchError("remote fetch record is invalid") from error
        if payload != record.canonical_bytes() + b"\n":
            raise ExperimentPromotionRemoteFetchError("remote fetch record is not canonical")
        expected_path = Path(record.plan.cache_root) / record.record_key
        if record_path.resolve(strict=True) != expected_path.resolve(strict=True) and (
            expected_path.is_symlink()
            or not expected_path.is_file()
            or expected_path.read_bytes() != payload
        ):
            raise ExperimentPromotionRemoteFetchError(
                "remote fetch record copy has no matching canonical cache record"
            )
        root = expected_path.parent
        expected_paths = {"record.json"}
        for artifact in record.plan.artifacts:
            expected_paths.add(f"{artifact.cache_prefix}/receipt.json")
            expected_paths.update(
                f"{artifact.cache_prefix}/chunks/{index:08d}.bin"
                for index in range(artifact.chunk_count)
            )
        ExperimentPromotionRemoteFetcher._validate_cache_prefix(root, expected_paths)
        store = LocalBlobStore(root)
        for receipt in record.artifact_receipts:
            receipt_path = root / receipt.receipt_key
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ExperimentPromotionRemoteFetchError(
                    "cached remote artifact receipt is not a regular file"
                )
            receipt_payload = store.get(receipt.receipt_key)
            if receipt_payload != receipt.canonical_bytes() + b"\n":
                raise ExperimentPromotionRemoteFetchError(
                    "cached remote artifact receipt conflicts with the record"
                )
            digest = hashlib.sha256()
            size = 0
            for chunk in receipt.chunks:
                chunk_path = root / chunk.key
                if chunk_path.is_symlink() or not chunk_path.is_file():
                    raise ExperimentPromotionRemoteFetchError(
                        "cached remote artifact chunk is not a regular file"
                    )
                data = store.get(chunk.key)
                if (
                    len(data) != chunk.size_bytes
                    or hashlib.sha256(data).hexdigest() != chunk.sha256
                ):
                    raise ExperimentPromotionRemoteFetchError(
                        "cached remote artifact chunk conflicts with its receipt"
                    )
                size += len(data)
                digest.update(data)
            if size != receipt.size_bytes or digest.hexdigest() != receipt.sha256:
                raise ExperimentPromotionRemoteFetchError(
                    "cached remote artifact does not match its signed reference"
                )
        return record

    def _inspect_artifact(
        self,
        reference: ExperimentPromotionArtifactRef,
        policy: ExperimentPromotionRemotePolicy,
    ) -> ExperimentPromotionRemoteArtifactPlan:
        parsed = urlsplit(reference.locator)
        provider = parsed.scheme
        artifact_id = ExperimentPromotionRemoteArtifactPlan.expected_artifact_id(reference)
        issues: set[str] = set()
        source: ExperimentPromotionRemoteSource | None = None
        if provider not in {
            ExperimentPromotionRemoteProvider.HTTPS,
            ExperimentPromotionRemoteProvider.S3,
        }:
            issues.add("unsupported_provider")
        elif (
            provider == ExperimentPromotionRemoteProvider.HTTPS
            and parsed.netloc.casefold() not in policy.allowed_https_authorities
        ) or (
            provider == ExperimentPromotionRemoteProvider.S3
            and parsed.netloc.casefold() not in policy.allowed_s3_buckets
        ):
            issues.add("unauthorized")
        else:
            try:
                source = self.reader.inspect(reference.locator)
            except Exception as error:
                issues.add(f"unavailable:{type(error).__name__}")
        if source is not None:
            if source.size_bytes != reference.size_bytes:
                issues.add("size_mismatch")
            if policy.require_source_validator and not source.has_validator:
                issues.add("missing_validator")
        if reference.size_bytes > policy.max_artifact_bytes:
            issues.add("artifact_too_large")
        chunk_count = (
            math.ceil(reference.size_bytes / policy.chunk_size_bytes) if source is not None else 0
        )
        return ExperimentPromotionRemoteArtifactPlan(
            reference=reference,
            artifact_id=artifact_id,
            provider=provider,
            source=source,
            chunk_count=chunk_count,
            cache_prefix=f"artifacts/{artifact_id}",
            issues=tuple(sorted(issues)),
        )

    def _fetch_artifact(
        self,
        store: LocalBlobStore,
        artifact: ExperimentPromotionRemoteArtifactPlan,
        chunk_size_bytes: int,
    ) -> ExperimentPromotionRemoteArtifactReceipt:
        if artifact.issues or artifact.source is None:
            raise ExperimentPromotionRemoteFetchError("cannot fetch an unresolved remote artifact")
        chunks: list[ExperimentPromotionRemoteChunk] = []
        digest = hashlib.sha256()
        offset = 0
        for index in range(artifact.chunk_count):
            end = min(
                artifact.reference.size_bytes - 1,
                offset + chunk_size_bytes - 1,
            )
            key = f"{artifact.cache_prefix}/chunks/{index:08d}.bin"
            existing = store.head(key)
            if existing is None:
                data = self.reader.read_range(
                    artifact.reference.locator,
                    start=offset,
                    end=end,
                    source=artifact.source,
                )
                try:
                    store.put_if_absent(key, data)
                except BlobConflictError as error:
                    raise ExperimentPromotionRemoteFetchError(
                        "remote fetch chunk conflicts with concurrent progress"
                    ) from error
            data = store.get(key)
            expected_size = end - offset + 1
            if len(data) != expected_size:
                raise ExperimentPromotionRemoteFetchError(
                    "cached remote fetch chunk has an unexpected size"
                )
            chunk_digest = hashlib.sha256(data).hexdigest()
            chunks.append(
                ExperimentPromotionRemoteChunk(
                    index=index,
                    offset=offset,
                    size_bytes=len(data),
                    sha256=chunk_digest,
                    key=key,
                )
            )
            digest.update(data)
            offset += len(data)
        if (
            offset != artifact.reference.size_bytes
            or digest.hexdigest() != artifact.reference.sha256
        ):
            raise ExperimentPromotionRemoteFetchError(
                "remote artifact bytes do not match the signed promotion manifest"
            )
        chunk_tuple = tuple(chunks)
        receipt = ExperimentPromotionRemoteArtifactReceipt(
            receipt_id=ExperimentPromotionRemoteArtifactReceipt.expected_receipt_id(
                artifact,
                chunk_tuple,
            ),
            artifact=artifact,
            chunks=chunk_tuple,
            size_bytes=offset,
            sha256=digest.hexdigest(),
            receipt_key=f"{artifact.cache_prefix}/receipt.json",
        )
        try:
            store.put_if_absent(receipt.receipt_key, receipt.canonical_bytes() + b"\n")
        except BlobConflictError as error:
            raise ExperimentPromotionRemoteFetchError(
                "remote artifact receipt conflicts with existing progress"
            ) from error
        return receipt

    @staticmethod
    def _cache_root(value: Path | str) -> Path:
        path = Path(value)
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError("remote fetch cache root must be a regular directory")
        return path.resolve(strict=False)

    @classmethod
    def _prepare_cache(cls, value: str) -> Path:
        path = cls._cache_root(value)
        path.mkdir(parents=True, exist_ok=True)
        if path.resolve(strict=True).as_posix() != value:
            raise ExperimentPromotionRemoteFetchError(
                "remote fetch cache identity changed after preview"
            )
        return path

    @staticmethod
    def _ensure_safe_prefix(cache: Path, prefix: Path) -> None:
        current = cache
        for part in prefix.relative_to(cache).parts:
            current /= part
            try:
                current.mkdir()
            except FileExistsError:
                if current.is_symlink() or not current.is_dir():
                    raise ExperimentPromotionRemoteFetchError(
                        "remote fetch cache prefix is unsafe"
                    ) from None
            current.chmod(0o755)

    @staticmethod
    def _validate_cache_prefix(prefix: Path, expected_paths: set[str]) -> None:
        if not prefix.exists():
            return
        if prefix.is_symlink() or not prefix.is_dir():
            raise ExperimentPromotionRemoteFetchError(
                "remote fetch cache prefix is not a regular directory"
            )
        for entry in prefix.rglob("*"):
            if entry.is_symlink():
                raise ExperimentPromotionRemoteFetchError(
                    "remote fetch cache prefix contains a symbolic link"
                )
            if entry.is_dir():
                continue
            relative = entry.relative_to(prefix).as_posix()
            if relative not in expected_paths:
                raise ExperimentPromotionRemoteFetchError(
                    f"remote fetch cache prefix contains unexpected file {relative!r}"
                )


def remote_artifact_output_path(reference: ExperimentPromotionArtifactRef) -> str:
    if reference.scope is not ExperimentPromotionArtifactScope.REMOTE:
        raise ValueError("remote output path requires a remote artifact reference")
    parsed = urlsplit(reference.locator)
    name = Path(unquote(parsed.path)).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-") or "artifact.bin"
    return f"remote/{reference.sha256}/{safe_name}"


def _validate_trust_evidence(
    receipt: ExperimentPromotionArchiveReceipt,
    attestation: SignedExperimentPromotionArchiveAttestation,
    verification: ExperimentPromotionArchiveAttestationVerification,
    trusted_signer_key_ids: tuple[str, ...],
) -> None:
    if trusted_signer_key_ids != tuple(sorted(set(trusted_signer_key_ids))) or any(
        _SIGNER_KEY_ID.fullmatch(item) is None for item in trusted_signer_key_ids
    ):
        raise ValueError("trusted remote-fetch signer key IDs must be sorted and unique")
    if attestation.attestation.receipt != receipt:
        raise ValueError("remote fetch attestation receipt is inconsistent")
    embedded_key = attestation.signature.public_key_base64
    embedded_id = Ed25519ManifestSigner.public_key_id(embedded_key)
    derived = ExperimentPromotionArchiveAttestor.verify(
        attestation,
        receipt,
        trusted_public_keys=(embedded_key,) if embedded_id in trusted_signer_key_ids else (),
    )
    if verification != derived:
        raise ValueError("remote fetch attestation verification is inconsistent")


def _s3_location(locator: str) -> tuple[str, str]:
    parsed = urlsplit(locator)
    bucket = parsed.netloc.casefold()
    key = unquote(parsed.path.lstrip("/"))
    if not bucket or not key:
        raise ExperimentPromotionRemoteFetchError("S3 URI must include a bucket and object key")
    return bucket, key


def _s3_provider_identity(bucket: str, endpoint_url: str | None) -> str:
    endpoint = endpoint_url.rstrip("/") if endpoint_url is not None else "aws"
    return f"{endpoint}/{bucket}"


def _clean_etag(value: object) -> str | None:
    return str(value).strip('"') if value is not None else None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


__all__ = [
    "ExperimentPromotionRemoteArtifactPlan",
    "ExperimentPromotionRemoteArtifactReceipt",
    "ExperimentPromotionRemoteCheck",
    "ExperimentPromotionRemoteChunk",
    "ExperimentPromotionRemoteFetchError",
    "ExperimentPromotionRemoteFetchPlan",
    "ExperimentPromotionRemoteFetchRecord",
    "ExperimentPromotionRemoteFetcher",
    "ExperimentPromotionRemotePolicy",
    "ExperimentPromotionRemoteProvider",
    "ExperimentPromotionRemoteReader",
    "ExperimentPromotionRemoteReaderRouter",
    "ExperimentPromotionRemoteSource",
    "remote_artifact_output_path",
]

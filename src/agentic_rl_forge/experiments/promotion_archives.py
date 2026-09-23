from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import io
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zlib
from collections.abc import Collection, Mapping
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import Field, model_validator

from agentic_rl_forge.contracts import ContractModel, ManifestSignature, utc_now
from agentic_rl_forge.data.manifests import Ed25519ManifestSigner
from agentic_rl_forge.experiments.models import ExperimentPlan, ExperimentReport
from agentic_rl_forge.experiments.operations import (
    ExperimentAnalysis,
    ExperimentOperationsIndex,
)
from agentic_rl_forge.experiments.promotion import (
    ExperimentPromotionRecord,
    ExperimentReproducibilityManifest,
    render_experiment_model_card,
)
from agentic_rl_forge.storage.blobs import LocalBlobStore


class ExperimentPromotionArchiveError(ValueError):
    pass


class ExperimentPromotionArchiveReceipt(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    archive_id: str = Field(pattern=r"^experiment_promotion_archive_[0-9a-f]{24}$")
    promotion_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    promotion_id: str = Field(pattern=r"^experiment_promotion_[0-9a-f]{24}$")
    reproducibility_manifest_id: str = Field(pattern=r"^experiment_repro_[0-9a-f]{24}$")
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    member_count: Literal[8] = 8

    @model_validator(mode="after")
    def validate_receipt(self) -> ExperimentPromotionArchiveReceipt:
        expected = self.expected_archive_id(self.content_digest)
        if self.archive_id != expected:
            raise ValueError("experiment promotion archive ID does not match its content digest")
        return self

    @staticmethod
    def expected_archive_id(content_digest: str) -> str:
        return f"experiment_promotion_archive_{content_digest[:24]}"


class ExperimentPromotionArchiveAttestation(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    attestation_id: str = Field(pattern=r"^experiment_promotion_attestation_[0-9a-f]{24}$")
    receipt: ExperimentPromotionArchiveReceipt
    signer_key_id: str = Field(pattern=r"^ed25519_[0-9a-f]{24}$")
    signed_at: datetime

    @model_validator(mode="after")
    def validate_attestation(self) -> ExperimentPromotionArchiveAttestation:
        if self.signed_at.tzinfo is None or self.signed_at.utcoffset() is None:
            raise ValueError("experiment promotion attestation time must be timezone-aware")
        expected = self.expected_attestation_id(
            receipt=self.receipt,
            signer_key_id=self.signer_key_id,
            signed_at=self.signed_at,
        )
        if self.attestation_id != expected:
            raise ValueError("experiment promotion attestation ID does not match its contents")
        return self

    @staticmethod
    def expected_attestation_id(
        *,
        receipt: ExperimentPromotionArchiveReceipt,
        signer_key_id: str,
        signed_at: datetime,
    ) -> str:
        payload = b"\0".join(
            (
                receipt.canonical_bytes(),
                signer_key_id.encode("ascii"),
                signed_at.isoformat().encode("ascii"),
            )
        )
        return f"experiment_promotion_attestation_{hashlib.sha256(payload).hexdigest()[:24]}"


class SignedExperimentPromotionArchiveAttestation(ContractModel):
    attestation: ExperimentPromotionArchiveAttestation
    signature: ManifestSignature

    @model_validator(mode="after")
    def validate_signature_identity(self) -> SignedExperimentPromotionArchiveAttestation:
        if self.signature.algorithm != "ed25519":
            raise ValueError("experiment promotion attestation requires Ed25519")
        try:
            public_key = base64.b64decode(
                self.signature.public_key_base64,
                validate=True,
            )
        except (ValueError, binascii.Error) as error:
            raise ValueError("experiment promotion attestation public key is invalid") from error
        if len(public_key) != 32:
            raise ValueError("experiment promotion attestation public key must contain 32 bytes")
        expected_key_id = f"ed25519_{hashlib.sha256(public_key).hexdigest()[:24]}"
        if self.attestation.signer_key_id != expected_key_id:
            raise ValueError("experiment promotion signer key ID does not match its public key")
        payload_digest = hashlib.sha256(self.attestation.canonical_bytes()).hexdigest()
        if self.signature.payload_sha256 != payload_digest:
            raise ValueError("experiment promotion attestation payload digest does not match")
        return self


class ExperimentPromotionArchiveAttestationVerification(ContractModel):
    attestation_id: str = Field(pattern=r"^experiment_promotion_attestation_[0-9a-f]{24}$")
    valid: bool
    archive_receipt_matches: bool
    signer_identity_valid: bool
    payload_digest_valid: bool
    signature_valid: bool
    trusted_signer: bool

    @model_validator(mode="after")
    def validate_result(self) -> ExperimentPromotionArchiveAttestationVerification:
        expected = all(
            (
                self.archive_receipt_matches,
                self.signer_identity_valid,
                self.payload_digest_valid,
                self.signature_valid,
                self.trusted_signer,
            )
        )
        if self.valid is not expected:
            raise ValueError("experiment promotion attestation validity does not match findings")
        return self


class ExperimentPromotionArchive:
    MEMBERS = (
        "decision.json",
        "record.json",
        "manifest.json",
        "model-card.md",
        "plan.json",
        "report.json",
        "index.json",
        "analysis.json",
    )
    _CHUNK_SIZE = 1024 * 1024
    _MAX_MEMBER_BYTES = 16 * 1024 * 1024
    _MAX_TOTAL_BYTES = 64 * 1024 * 1024

    def __init__(self, promotion_directory: Path | str) -> None:
        self.promotion_directory = Path(promotion_directory)

    def pack(self, output: Path | str) -> ExperimentPromotionArchiveReceipt:
        output_path = Path(output)
        if not output_path.name.endswith(".tar.gz"):
            raise ExperimentPromotionArchiveError(
                "experiment promotion archive must use the .tar.gz suffix"
            )
        source, payloads, _ = self._read_directory(self.promotion_directory)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output = output_path.parent.resolve(strict=True) / output_path.name
        if resolved_output.is_relative_to(source):
            raise ExperimentPromotionArchiveError(
                "experiment promotion archive output must stay outside its source directory"
            )

        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=".arf-promotion-",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            self._write_archive_path(temporary, payloads)
            temporary.chmod(0o644)
            content_digest, _ = self._digest_file(temporary)
            receipt = self.inspect(temporary, expected_sha256=content_digest)
            self._validate_existing_checksum(output_path, content_digest)
            self._publish_file(temporary, output_path, content_digest)
            LocalBlobStore(output_path.parent).put_if_absent(
                self.checksum_path(output_path).name,
                f"{content_digest}\n".encode("ascii"),
            )
            output_path.chmod(0o644)
            self.checksum_path(output_path).chmod(0o644)
            return receipt
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def inspect(
        cls,
        archive: Path | str,
        *,
        expected_sha256: str,
    ) -> ExperimentPromotionArchiveReceipt:
        receipt, _ = cls.inspect_record(
            archive,
            expected_sha256=expected_sha256,
        )
        return receipt

    @classmethod
    def inspect_record(
        cls,
        archive: Path | str,
        *,
        expected_sha256: str,
    ) -> tuple[ExperimentPromotionArchiveReceipt, ExperimentPromotionRecord]:
        """Inspect canonical archive bytes and return their authoritative record."""
        archive_path = Path(archive)
        expected = cls.normalize_digest(expected_sha256)
        content_digest, size_bytes = cls._digest_file(archive_path)
        if content_digest != expected:
            raise ExperimentPromotionArchiveError(
                "experiment promotion archive SHA-256 does not match"
            )
        payloads, record = cls._read_archive(archive_path, destination=None)
        canonical = cls._canonical_archive_bytes(payloads)
        if hashlib.sha256(canonical).hexdigest() != content_digest:
            raise ExperimentPromotionArchiveError(
                "experiment promotion archive bytes are not canonical"
            )
        return cls._receipt(record, content_digest, size_bytes), record

    @classmethod
    def unpack(
        cls,
        archive: Path | str,
        destination: Path | str,
        *,
        expected_sha256: str,
    ) -> ExperimentPromotionArchiveReceipt:
        archive_path = Path(archive)
        destination_path = Path(destination)
        if destination_path.exists() or destination_path.is_symlink():
            raise ExperimentPromotionArchiveError("archive destination already exists")
        expected = cls.normalize_digest(expected_sha256)
        content_digest, size_bytes = cls._digest_file(archive_path)
        if content_digest != expected:
            raise ExperimentPromotionArchiveError(
                "experiment promotion archive SHA-256 does not match"
            )

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                dir=destination_path.parent,
                prefix=".arf-promotion-unpack-",
            )
        )
        try:
            payloads, record = cls._read_archive(archive_path, destination=temporary)
            canonical = cls._canonical_archive_bytes(payloads)
            if hashlib.sha256(canonical).hexdigest() != content_digest:
                raise ExperimentPromotionArchiveError(
                    "experiment promotion archive bytes are not canonical"
                )
            temporary.chmod(0o755)
            if destination_path.exists() or destination_path.is_symlink():
                raise ExperimentPromotionArchiveError(
                    "archive destination appeared during extraction"
                )
            os.rename(temporary, destination_path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return cls._receipt(record, content_digest, size_bytes)

    @classmethod
    def verify_directory(cls, directory: Path | str) -> ExperimentPromotionRecord:
        _, _, record = cls._read_directory(Path(directory))
        return record

    @staticmethod
    def checksum_path(archive: Path | str) -> Path:
        path = Path(archive)
        return path.with_name(f"{path.name}.sha256")

    @classmethod
    def read_checksum(cls, path: Path | str) -> str:
        checksum_path = Path(path)
        try:
            payload = checksum_path.read_bytes()
        except OSError as error:
            raise ExperimentPromotionArchiveError(
                f"cannot read experiment promotion archive checksum: {error}"
            ) from error
        if payload.endswith(b"\r\n"):
            digest_payload = payload[:-2]
        elif payload.endswith(b"\n"):
            digest_payload = payload[:-1]
        else:
            raise ExperimentPromotionArchiveError("archive checksum file is not canonical")
        try:
            digest = digest_payload.decode("ascii")
        except UnicodeDecodeError as error:
            raise ExperimentPromotionArchiveError("archive checksum is not ASCII") from error
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ExperimentPromotionArchiveError("archive checksum file is not canonical")
        return digest

    @staticmethod
    def normalize_digest(value: str) -> str:
        digest = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ExperimentPromotionArchiveError(
                "expected archive SHA-256 must contain 64 hex digits"
            )
        return digest

    @classmethod
    def _read_directory(
        cls,
        directory: Path,
    ) -> tuple[Path, dict[str, bytes], ExperimentPromotionRecord]:
        if directory.is_symlink() or not directory.is_dir():
            raise ExperimentPromotionArchiveError(
                "experiment promotion source must be a regular directory"
            )
        try:
            resolved = directory.resolve(strict=True)
            entries = tuple(directory.iterdir())
        except OSError as error:
            raise ExperimentPromotionArchiveError(
                f"cannot read experiment promotion directory: {error}"
            ) from error
        names = {entry.name for entry in entries}
        expected_names = set(cls.MEMBERS)
        if names != expected_names:
            missing = sorted(expected_names - names)
            unexpected = sorted(names - expected_names)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise ExperimentPromotionArchiveError(
                "promotion directory member set is invalid: " + "; ".join(details)
            )
        payloads = {name: cls._read_source_member(directory / name) for name in cls.MEMBERS}
        if sum(len(payload) for payload in payloads.values()) > cls._MAX_TOTAL_BYTES:
            raise ExperimentPromotionArchiveError(
                "experiment promotion metadata exceeds the total safety limit"
            )
        record = cls._validate_payloads(payloads)
        return resolved, payloads, record

    @classmethod
    def _read_source_member(cls, path: Path) -> bytes:
        if path.is_symlink():
            raise ExperimentPromotionArchiveError(
                f"promotion member {path.name!r} must be a regular file"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ExperimentPromotionArchiveError(
                f"cannot open promotion member {path.name!r}: {error}"
            ) from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ExperimentPromotionArchiveError(
                    f"promotion member {path.name!r} must be a regular file"
                )
            if before.st_size > cls._MAX_MEMBER_BYTES:
                raise ExperimentPromotionArchiveError(
                    f"promotion member {path.name!r} exceeds the safety limit"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                payload = source.read(cls._MAX_MEMBER_BYTES + 1)
            after = os.fstat(descriptor)
        except OSError as error:
            raise ExperimentPromotionArchiveError(
                f"cannot read promotion member {path.name!r}: {error}"
            ) from error
        finally:
            os.close(descriptor)
        if len(payload) > cls._MAX_MEMBER_BYTES:
            raise ExperimentPromotionArchiveError(
                f"promotion member {path.name!r} exceeds the safety limit"
            )
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ExperimentPromotionArchiveError(
                f"promotion member {path.name!r} changed while it was read"
            )
        return payload

    @classmethod
    def _write_archive_path(cls, output: Path, payloads: Mapping[str, bytes]) -> None:
        try:
            with output.open("wb") as raw_output:
                cls._write_archive(raw_output, payloads)
                raw_output.flush()
                os.fsync(raw_output.fileno())
        except (OSError, tarfile.TarError, ValueError) as error:
            raise ExperimentPromotionArchiveError(
                f"cannot build canonical experiment promotion archive: {error}"
            ) from error

    @classmethod
    def _canonical_archive_bytes(cls, payloads: Mapping[str, bytes]) -> bytes:
        output = io.BytesIO()
        cls._write_archive(output, payloads)
        return output.getvalue()

    @classmethod
    def _write_archive(
        cls,
        output: BinaryIO,
        payloads: Mapping[str, bytes],
    ) -> None:
        with (
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=output,
                mtime=0,
            ) as compressed,
            tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as archive,
        ):
            for name in cls.MEMBERS:
                payload = payloads[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.type = tarfile.REGTYPE
                archive.addfile(info, io.BytesIO(payload))

    @classmethod
    def _read_archive(
        cls,
        archive_path: Path,
        *,
        destination: Path | None,
    ) -> tuple[dict[str, bytes], ExperimentPromotionRecord]:
        payloads: dict[str, bytes] = {}
        total_size = 0
        expected_offset = 0
        try:
            with tarfile.open(archive_path, mode="r|gz") as archive:
                for name in cls.MEMBERS:
                    member = archive.next()
                    if member is None:
                        raise ExperimentPromotionArchiveError(f"archive is missing member {name!r}")
                    cls._validate_member_metadata(member, expected_offset=expected_offset)
                    expected_offset = cls._next_offset(member)
                    if member.name != name:
                        raise ExperimentPromotionArchiveError(
                            f"archive member order mismatch: expected {name!r}, got {member.name!r}"
                        )
                    if member.size > cls._MAX_MEMBER_BYTES:
                        raise ExperimentPromotionArchiveError(
                            f"archive member {name!r} exceeds the safety limit"
                        )
                    total_size += member.size
                    if total_size > cls._MAX_TOTAL_BYTES:
                        raise ExperimentPromotionArchiveError(
                            "experiment promotion metadata exceeds the total safety limit"
                        )
                    payloads[name] = cls._consume_member(
                        archive,
                        member,
                        destination=destination,
                    )
                unexpected = archive.next()
                if unexpected is not None:
                    raise ExperimentPromotionArchiveError(
                        f"archive contains unexpected member {unexpected.name!r}"
                    )
        except ExperimentPromotionArchiveError:
            raise
        except (OSError, EOFError, tarfile.TarError, ValueError, zlib.error) as error:
            raise ExperimentPromotionArchiveError(
                f"invalid experiment promotion archive: {error}"
            ) from error
        return payloads, cls._validate_payloads(payloads)

    @classmethod
    def _consume_member(
        cls,
        archive: tarfile.TarFile,
        member: tarfile.TarInfo,
        *,
        destination: Path | None,
    ) -> bytes:
        source = archive.extractfile(member)
        if source is None:
            raise ExperimentPromotionArchiveError(
                f"archive member {member.name!r} has no file data"
            )
        target = destination / member.name if destination is not None else None
        captured = bytearray()
        size = 0
        output: BinaryIO | None = target.open("xb") if target is not None else None
        try:
            while True:
                chunk = source.read(cls._CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                captured.extend(chunk)
                if output is not None:
                    output.write(chunk)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        finally:
            source.close()
            if output is not None:
                output.close()
        if size != member.size:
            raise ExperimentPromotionArchiveError(f"archive member {member.name!r} is truncated")
        if target is not None:
            target.chmod(0o644)
        return bytes(captured)

    @classmethod
    def _validate_payloads(
        cls,
        payloads: Mapping[str, bytes],
    ) -> ExperimentPromotionRecord:
        try:
            decision = ExperimentPromotionRecord.model_validate_json(payloads["decision.json"])
            record = ExperimentPromotionRecord.model_validate_json(payloads["record.json"])
            manifest = ExperimentReproducibilityManifest.model_validate_json(
                payloads["manifest.json"]
            )
            plan = ExperimentPlan.model_validate_json(payloads["plan.json"])
            report = ExperimentReport.model_validate_json(payloads["report.json"])
            index = ExperimentOperationsIndex.model_validate_json(payloads["index.json"])
            analysis = ExperimentAnalysis.model_validate_json(payloads["analysis.json"])
        except (KeyError, ValueError) as error:
            raise ExperimentPromotionArchiveError(
                f"promotion archive contains invalid contract JSON: {error}"
            ) from error

        canonical_models = {
            "decision.json": decision,
            "record.json": record,
            "manifest.json": manifest,
            "plan.json": plan,
            "report.json": report,
            "index.json": index,
            "analysis.json": analysis,
        }
        for name, model in canonical_models.items():
            if payloads[name] != model.canonical_bytes() + b"\n":
                raise ExperimentPromotionArchiveError(
                    f"promotion archive member {name!r} is not canonical"
                )
        if decision != record or payloads["decision.json"] != payloads["record.json"]:
            raise ExperimentPromotionArchiveError(
                "promotion decision and record sidecar do not match"
            )
        preview = record.preview
        if manifest != preview.manifest:
            raise ExperimentPromotionArchiveError(
                "promotion manifest sidecar does not match the decision"
            )
        if (
            plan.plan_id != preview.plan_id
            or report.plan_id != preview.plan_id
            or report.report_id != preview.report_id
            or index.index_id != preview.index_id
            or analysis.analysis_id != preview.analysis_id
            or analysis.index_id != index.index_id
        ):
            raise ExperimentPromotionArchiveError(
                "promotion metadata identities do not match the decision"
            )
        plan_trial = next(
            (item for item in plan.trials if item.trial_id == preview.trial_id),
            None,
        )
        report_trial = next(
            (item for item in report.trials if item.trial_id == preview.trial_id),
            None,
        )
        index_row = next(
            (
                item
                for item in index.rows
                if item.plan_id == preview.plan_id and item.trial_id == preview.trial_id
            ),
            None,
        )
        ranked_trial = next(
            (
                item
                for item in analysis.trials
                if item.plan_id == preview.plan_id and item.trial_id == preview.trial_id
            ),
            None,
        )
        index_report = next(
            (item for item in index.reports if item.plan_id == preview.plan_id),
            None,
        )
        if None in (plan_trial, report_trial, index_row, ranked_trial, index_report):
            raise ExperimentPromotionArchiveError(
                "promotion metadata does not contain the selected trial"
            )
        if (
            plan.name != manifest.experiment_name
            or report.name != manifest.experiment_name
            or plan_trial is None
            or plan_trial.config_digest != manifest.config_digest
            or plan_trial.parameters != manifest.parameters
            or report_trial is None
            or report_trial.state is not preview.state
            or index_row is None
            or index_row.report_id != preview.report_id
            or index_row.state is not preview.state
            or ranked_trial is None
            or ranked_trial.report_id != preview.report_id
            or ranked_trial.rank != preview.rank
            or ranked_trial.pareto_front is not preview.pareto_front
            or index_report is None
            or index_report.report_id != preview.report_id
        ):
            raise ExperimentPromotionArchiveError(
                "promotion selected-trial evidence is inconsistent"
            )
        model_card = render_experiment_model_card(
            index,
            analysis,
            preview,
            record.operator,
            record.reason,
            record.approvers,
        ).encode("utf-8")
        if payloads["model-card.md"] != model_card:
            raise ExperimentPromotionArchiveError(
                "promotion model card cannot be reproduced from the decision"
            )
        if hashlib.sha256(model_card).hexdigest() != record.model_card_sha256:
            raise ExperimentPromotionArchiveError(
                "promotion model card digest does not match the decision"
            )
        return record

    @staticmethod
    def _validate_member_metadata(
        member: tarfile.TarInfo,
        *,
        expected_offset: int,
    ) -> None:
        if not member.isreg() or member.type != tarfile.REGTYPE:
            raise ExperimentPromotionArchiveError("archive members must be regular files")
        if (
            member.mode != 0o644
            or member.uid != 0
            or member.gid != 0
            or member.uname != ""
            or member.gname != ""
            or member.mtime != 0
            or member.linkname != ""
            or member.pax_headers
        ):
            raise ExperimentPromotionArchiveError(
                f"archive member {member.name!r} has noncanonical metadata"
            )
        if member.offset != expected_offset or member.offset_data != member.offset + 512:
            raise ExperimentPromotionArchiveError(
                f"archive member {member.name!r} has a noncanonical header layout"
            )

    @staticmethod
    def _next_offset(member: tarfile.TarInfo) -> int:
        blocks = (member.size + 511) // 512
        return member.offset_data + blocks * 512

    @staticmethod
    def _digest_file(path: Path) -> tuple[str, int]:
        content_hash = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(ExperimentPromotionArchive._CHUNK_SIZE):
                    size += len(chunk)
                    content_hash.update(chunk)
        except OSError as error:
            raise ExperimentPromotionArchiveError(
                f"cannot read experiment promotion archive: {error}"
            ) from error
        return content_hash.hexdigest(), size

    @classmethod
    def _publish_file(cls, source: Path, target: Path, content_digest: str) -> None:
        try:
            os.link(source, target)
        except FileExistsError:
            if not target.is_file() or cls._digest_file(target)[0] != content_digest:
                raise ExperimentPromotionArchiveError(
                    "archive output already exists with different content"
                ) from None
        except OSError as error:
            raise ExperimentPromotionArchiveError(
                f"cannot publish experiment promotion archive: {error}"
            ) from error

    @classmethod
    def _validate_existing_checksum(cls, archive: Path, content_digest: str) -> None:
        path = cls.checksum_path(archive)
        if path.exists() and cls.read_checksum(path) != content_digest:
            raise ExperimentPromotionArchiveError(
                "archive checksum output already exists with different content"
            )

    @classmethod
    def _receipt(
        cls,
        record: ExperimentPromotionRecord,
        content_digest: str,
        size_bytes: int,
    ) -> ExperimentPromotionArchiveReceipt:
        return ExperimentPromotionArchiveReceipt(
            archive_id=ExperimentPromotionArchiveReceipt.expected_archive_id(content_digest),
            promotion_name=record.preview.promotion_name,
            promotion_id=record.promotion_id,
            reproducibility_manifest_id=record.preview.manifest.manifest_id,
            content_digest=content_digest,
            size_bytes=size_bytes,
            member_count=8,
        )


class ExperimentPromotionArchiveAttestor:
    def __init__(self, signer: Ed25519ManifestSigner) -> None:
        self._signer = signer

    def sign(
        self,
        receipt: ExperimentPromotionArchiveReceipt,
        *,
        signed_at: datetime | None = None,
    ) -> SignedExperimentPromotionArchiveAttestation:
        signing_time = signed_at or utc_now()
        attestation_id = ExperimentPromotionArchiveAttestation.expected_attestation_id(
            receipt=receipt,
            signer_key_id=self._signer.key_id,
            signed_at=signing_time,
        )
        attestation = ExperimentPromotionArchiveAttestation(
            attestation_id=attestation_id,
            receipt=receipt,
            signer_key_id=self._signer.key_id,
            signed_at=signing_time,
        )
        return SignedExperimentPromotionArchiveAttestation(
            attestation=attestation,
            signature=self._signer.sign_payload(attestation.canonical_bytes()),
        )

    @staticmethod
    def verify(
        signed: SignedExperimentPromotionArchiveAttestation,
        receipt: ExperimentPromotionArchiveReceipt,
        *,
        trusted_public_keys: Collection[str],
    ) -> ExperimentPromotionArchiveAttestationVerification:
        payload = signed.attestation.canonical_bytes()
        signature = signed.signature
        archive_receipt_matches = signed.attestation.receipt == receipt
        try:
            embedded_key_id = Ed25519ManifestSigner.public_key_id(signature.public_key_base64)
        except Exception:
            embedded_key_id = ""
        signer_identity_valid = embedded_key_id == signed.attestation.signer_key_id
        payload_digest_valid = hashlib.sha256(payload).hexdigest() == signature.payload_sha256
        signature_valid = Ed25519ManifestSigner.verify_payload(signature, payload)
        trusted_signer = signature.public_key_base64 in set(trusted_public_keys)
        valid = all(
            (
                archive_receipt_matches,
                signer_identity_valid,
                payload_digest_valid,
                signature_valid,
                trusted_signer,
            )
        )
        return ExperimentPromotionArchiveAttestationVerification(
            attestation_id=signed.attestation.attestation_id,
            valid=valid,
            archive_receipt_matches=archive_receipt_matches,
            signer_identity_valid=signer_identity_valid,
            payload_digest_valid=payload_digest_valid,
            signature_valid=signature_valid,
            trusted_signer=trusted_signer,
        )

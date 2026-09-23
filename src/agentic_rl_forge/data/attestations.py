from __future__ import annotations

import hashlib
from collections.abc import Collection
from datetime import datetime

from agentic_rl_forge.contracts import (
    RunArtifactArchiveAttestation,
    RunArtifactArchiveAttestationVerification,
    RunArtifactArchiveReceipt,
    SignedRunArtifactArchiveAttestation,
    utc_now,
)
from agentic_rl_forge.data.manifests import Ed25519ManifestSigner


class RunArtifactArchiveAttestor:
    def __init__(self, signer: Ed25519ManifestSigner) -> None:
        self._signer = signer

    def sign(
        self,
        receipt: RunArtifactArchiveReceipt,
        *,
        signed_at: datetime | None = None,
    ) -> SignedRunArtifactArchiveAttestation:
        signing_time = signed_at or utc_now()
        attestation_id = RunArtifactArchiveAttestation.expected_attestation_id(
            receipt=receipt,
            signer_key_id=self._signer.key_id,
            signed_at=signing_time,
        )
        attestation = RunArtifactArchiveAttestation(
            attestation_id=attestation_id,
            receipt=receipt,
            signer_key_id=self._signer.key_id,
            signed_at=signing_time,
        )
        return SignedRunArtifactArchiveAttestation(
            attestation=attestation,
            signature=self._signer.sign_payload(attestation.canonical_bytes()),
        )

    @staticmethod
    def verify(
        signed: SignedRunArtifactArchiveAttestation,
        receipt: RunArtifactArchiveReceipt,
        *,
        trusted_public_keys: Collection[str],
    ) -> RunArtifactArchiveAttestationVerification:
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
        return RunArtifactArchiveAttestationVerification(
            attestation_id=signed.attestation.attestation_id,
            valid=valid,
            archive_receipt_matches=archive_receipt_matches,
            signer_identity_valid=signer_identity_valid,
            payload_digest_valid=payload_digest_valid,
            signature_valid=signature_valid,
            trusted_signer=trusted_signer,
        )

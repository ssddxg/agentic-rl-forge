from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from agentic_rl_forge.cli import app
from agentic_rl_forge.experiments import (
    ExperimentPromotionAcquirer,
    ExperimentPromotionRemoteFetcher,
    ExperimentPromotionRemoteFetchError,
    ExperimentPromotionRemoteFetchPlan,
    ExperimentPromotionRemoteFetchRecord,
    ExperimentPromotionRemotePolicy,
    ExperimentPromotionRemoteReaderRouter,
    ExperimentPromotionRemoteSource,
)
from test_experiment_promotion_acquisition import (
    build_acquisition_fixture,
    read_filesystem_bytes,
)


class MemoryRemoteReader:
    def __init__(self, payloads: dict[str, bytes], *, fail_once_at: int | None = None) -> None:
        self.payloads = payloads
        self.fail_once_at = fail_once_at
        self.failed = False
        self.ranges: list[tuple[int, int]] = []

    def inspect(self, locator: str) -> ExperimentPromotionRemoteSource:
        payload = self.payloads[locator]
        return ExperimentPromotionRemoteSource(
            provider="https",
            provider_identity="unreachable.invalid",
            size_bytes=len(payload),
            etag=f'"{hashlib.sha256(payload).hexdigest()}"',
        )

    def read_range(
        self,
        locator: str,
        *,
        start: int,
        end: int,
        source: ExperimentPromotionRemoteSource,
    ) -> bytes:
        assert source.etag is not None
        self.ranges.append((start, end))
        if self.fail_once_at == start and not self.failed:
            self.failed = True
            raise RuntimeError("injected remote interruption")
        return self.payloads[locator][start : end + 1]


def test_remote_fetch_resumes_and_completes_strict_acquisition(tmp_path: Path) -> None:
    payload = b"remote-checkpoint-block" * 7000
    fixture = build_acquisition_fixture(
        tmp_path / "source",
        remote_checkpoint=True,
        remote_payload=payload,
    )
    locator = "https://unreachable.invalid/releases/model.bin"
    chunk_size = 64 * 1024
    reader = MemoryRemoteReader({locator: payload}, fail_once_at=chunk_size)
    fetcher = ExperimentPromotionRemoteFetcher(reader)
    cache = tmp_path / "remote-cache"
    policy = ExperimentPromotionRemotePolicy(
        allowed_https_authorities=("unreachable.invalid",),
        chunk_size_bytes=chunk_size,
        max_artifact_bytes=len(payload),
        max_total_bytes=len(payload),
    )

    plan = fetcher.preview(
        fixture.archive,
        fixture.attestation,
        cache,
        trusted_public_keys=(fixture.public_key,),
        policy=policy,
    )
    assert plan.eligible
    assert plan.remote_artifact_count == 1
    assert plan.artifacts[0].chunk_count == 3
    with pytest.raises(RuntimeError, match="interruption"):
        fetcher.execute(
            plan,
            fixture.archive,
            fixture.attestation,
            trusted_public_keys=(fixture.public_key,),
            confirm_plan_id=plan.plan_id,
        )
    assert reader.ranges[:2] == [(0, chunk_size - 1), (chunk_size, 2 * chunk_size - 1)]

    record = fetcher.execute(
        plan,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=plan.plan_id,
    )
    repeated = fetcher.execute(
        plan,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=plan.plan_id,
    )
    assert repeated == record
    assert record.fetched_chunk_count == 3
    assert reader.ranges.count((0, chunk_size - 1)) == 1
    record_path = cache / record.record_key
    assert ExperimentPromotionRemoteFetcher.verify_record(record_path) == record

    destination = tmp_path / "received"
    acquirer = ExperimentPromotionAcquirer(
        fixture.project_root,
        fixture.state_root,
        remote_records=(record_path,),
    )
    acquisition = acquirer.preview(
        fixture.archive,
        fixture.attestation,
        destination,
        trusted_public_keys=(fixture.public_key,),
    )
    assert acquisition.eligible
    assert acquisition.unresolved_remote_count == 0
    remote_file = next(
        item for item in acquisition.materialized_files if item.scope.value == "remote"
    )
    acquired = acquirer.execute(
        acquisition,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=acquisition.plan_id,
    )
    output = destination / "acquisitions" / acquired.acquisition_id / remote_file.output_path
    assert read_filesystem_bytes(output) == payload


def test_remote_fetch_requires_explicit_authority_and_detects_cache_tampering(
    tmp_path: Path,
) -> None:
    payload = b"remote-checkpoint"
    fixture = build_acquisition_fixture(
        tmp_path / "source",
        remote_checkpoint=True,
        remote_payload=payload,
    )
    locator = "https://unreachable.invalid/releases/model.bin"
    reader = MemoryRemoteReader({locator: payload})
    fetcher = ExperimentPromotionRemoteFetcher(reader)
    cache = tmp_path / "remote-cache"

    denied = fetcher.preview(
        fixture.archive,
        fixture.attestation,
        cache,
        trusted_public_keys=(fixture.public_key,),
        policy=ExperimentPromotionRemotePolicy(),
    )
    assert not denied.eligible
    assert {item.code for item in denied.checks if not item.passed} == {"sources.authorized"}
    assert reader.ranges == []

    policy = ExperimentPromotionRemotePolicy(
        allowed_https_authorities=("unreachable.invalid",),
    )
    plan = fetcher.preview(
        fixture.archive,
        fixture.attestation,
        cache,
        trusted_public_keys=(fixture.public_key,),
        policy=policy,
    )
    with pytest.raises(ExperimentPromotionRemoteFetchError, match="confirmation"):
        fetcher.execute(
            plan,
            fixture.archive,
            fixture.attestation,
            trusted_public_keys=(fixture.public_key,),
            confirm_plan_id="promotion_remote_fetch_plan_" + "0" * 24,
        )

    blocked_cache = tmp_path / "blocked-cache"
    blocked = fetcher.preview(
        fixture.archive,
        fixture.attestation,
        blocked_cache,
        trusted_public_keys=(fixture.public_key,),
        policy=policy,
    )
    blocked_prefix = (
        blocked_cache
        / "promotion-fetches"
        / ExperimentPromotionRemoteFetchRecord.expected_fetch_id(blocked.plan_id)
    )
    blocked_prefix.mkdir(parents=True)
    (blocked_prefix / "notes.txt").write_text("unreviewed", encoding="utf-8")
    with pytest.raises(ExperimentPromotionRemoteFetchError, match="unexpected file"):
        fetcher.execute(
            blocked,
            fixture.archive,
            fixture.attestation,
            trusted_public_keys=(fixture.public_key,),
            confirm_plan_id=blocked.plan_id,
        )

    record = fetcher.execute(
        plan,
        fixture.archive,
        fixture.attestation,
        trusted_public_keys=(fixture.public_key,),
        confirm_plan_id=plan.plan_id,
    )
    record_path = cache / record.record_key
    chunk = record.artifact_receipts[0].chunks[0]
    (record_path.parent / chunk.key).write_bytes(b"tampered")
    with pytest.raises(ExperimentPromotionRemoteFetchError, match="chunk conflicts"):
        ExperimentPromotionRemoteFetcher.verify_record(record_path)
    with pytest.raises(ExperimentPromotionRemoteFetchError, match="chunk conflicts"):
        ExperimentPromotionAcquirer(
            fixture.project_root,
            fixture.state_root,
            remote_records=(record_path,),
        )


def test_remote_fetch_contract_and_cli_fail_closed_without_network(tmp_path: Path) -> None:
    fixture = build_acquisition_fixture(tmp_path / "source", remote_checkpoint=True)
    reader = MemoryRemoteReader(
        {"https://unreachable.invalid/releases/model.bin": b"remote-checkpoint"}
    )
    plan = ExperimentPromotionRemoteFetcher(reader).preview(
        fixture.archive,
        fixture.attestation,
        tmp_path / "cache",
        trusted_public_keys=(fixture.public_key,),
        policy=ExperimentPromotionRemotePolicy(
            allowed_https_authorities=("unreachable.invalid",),
        ),
    )
    payload = plan.model_dump(mode="python")
    with pytest.raises(ValueError, match="byte count"):
        ExperimentPromotionRemoteFetchPlan.model_validate(
            {**payload, "total_size_bytes": plan.total_size_bytes + 1}
        )
    with pytest.raises(ValueError, match="plan ID"):
        ExperimentPromotionRemoteFetchPlan.model_validate(
            {**payload, "plan_id": "promotion_remote_fetch_plan_" + "0" * 24}
        )

    attestation_path = tmp_path / "attestation.json"
    public_key_path = tmp_path / "release.pub"
    plan_path = tmp_path / "denied-plan.json"
    attestation_path.write_bytes(fixture.attestation.canonical_bytes() + b"\n")
    public_key_path.write_text(f"{fixture.public_key}\n", encoding="ascii")
    result = CliRunner().invoke(
        app,
        [
            "experiment-promotion-fetch-remote",
            str(fixture.archive),
            str(attestation_path),
            str(tmp_path / "cli-cache"),
            "--public-key",
            str(public_key_path),
            "--plan-output",
            str(plan_path),
            "--fail-on-ineligible",
        ],
    )
    assert result.exit_code == 1, result.output
    denied = ExperimentPromotionRemoteFetchPlan.model_validate_json(plan_path.read_bytes())
    assert not denied.eligible
    assert not (tmp_path / "cli-cache").exists()


def test_https_reader_rejects_redirects_and_enforces_exact_ranges() -> None:
    payload = b"abcdef"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(payload)), "ETag": '"v1"'},
            )
        assert request.headers["range"] == "bytes=1-3"
        assert request.headers["if-match"] == '"v1"'
        return httpx.Response(
            206,
            content=payload[1:4],
            headers={"Content-Range": "bytes 1-3/6"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        router = ExperimentPromotionRemoteReaderRouter(
            allowed_https_authorities=("files.example",),
            allowed_s3_buckets=(),
            http_client=client,
        )
        source = router.inspect("https://files.example/model.bin")
        assert (
            router.read_range(
                "https://files.example/model.bin",
                start=1,
                end=3,
                source=source,
            )
            == b"bcd"
        )

    redirect = httpx.MockTransport(lambda _: httpx.Response(302, headers={"Location": "/other"}))
    with httpx.Client(transport=redirect, follow_redirects=False) as client:
        router = ExperimentPromotionRemoteReaderRouter(
            allowed_https_authorities=("files.example",),
            allowed_s3_buckets=(),
            http_client=client,
        )
        with pytest.raises(ExperimentPromotionRemoteFetchError, match="redirects"):
            router.inspect("https://files.example/model.bin")


def test_s3_reader_binds_version_and_uses_bounded_ranges() -> None:
    payload = b"0123456789"

    class RangeS3Client:
        def __init__(self) -> None:
            self.last_get: dict[str, Any] = {}

        def head_object(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"Bucket": "models", "Key": "release/model.bin"}
            return {
                "ContentLength": len(payload),
                "ETag": '"etag-v1"',
                "VersionId": "version-1",
                "LastModified": datetime(2026, 9, 20, tzinfo=timezone.utc),
            }

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            self.last_get = kwargs
            raw_range = str(kwargs["Range"]).removeprefix("bytes=")
            start_raw, end_raw = raw_range.split("-", maxsplit=1)
            return {
                "Body": io.BytesIO(payload[int(start_raw) : int(end_raw) + 1]),
            }

    client = RangeS3Client()
    router = ExperimentPromotionRemoteReaderRouter(
        allowed_https_authorities=(),
        allowed_s3_buckets=("models",),
        s3_client=client,
    )
    source = router.inspect("s3://models/release/model.bin")
    assert source.version_id == "version-1"
    assert (
        router.read_range(
            "s3://models/release/model.bin",
            start=2,
            end=5,
            source=source,
        )
        == b"2345"
    )
    assert client.last_get == {
        "Bucket": "models",
        "Key": "release/model.bin",
        "Range": "bytes=2-5",
        "VersionId": "version-1",
    }

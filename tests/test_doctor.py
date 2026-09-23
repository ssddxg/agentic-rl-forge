import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import agentic_rl_forge.quality.doctor as doctor_module
from agentic_rl_forge.quality import (
    DoctorCheck,
    DoctorProfile,
    DoctorReport,
    DoctorStatus,
    run_doctor,
)


def _set_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing: frozenset[str] = frozenset(),
) -> None:
    monkeypatch.setattr(
        doctor_module.importlib.util,
        "find_spec",
        lambda name: None if name in missing else object(),
    )


def _make_project(root: Path) -> None:
    (root / "examples/data").mkdir(parents=True)
    for relative_path in (
        "pyproject.toml",
        "README.md",
        "examples/data/corpus.jsonl",
        "examples/data/qa.jsonl",
    ):
        (root / relative_path).write_text("fixture\n", encoding="utf-8")


def test_core_without_optional_gpu_modules_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_modules(monkeypatch, missing=frozenset({"torch", "vllm", "sglang"}))
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.12.4")

    report = run_doctor()

    assert report.profile is DoctorProfile.CORE
    assert report.status is DoctorStatus.PASS
    assert report.exit_code(strict=True) == 0
    optional = [check for check in report.checks if not check.required]
    assert {check.status for check in optional} == {DoctorStatus.WARN}
    assert {check.name for check in optional} == {
        "optional.module.torch",
        "optional.module.vllm",
        "optional.module.sglang",
    }


def test_server_profile_fails_when_required_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_modules(monkeypatch, missing=frozenset({"uvicorn"}))
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.11.9")

    report = run_doctor(DoctorProfile.SERVER)

    assert report.status is DoctorStatus.FAIL
    assert report.exit_code(strict=False) == 0
    assert report.exit_code(strict=True) == 1
    uvicorn = next(check for check in report.checks if check.name == "server.module.uvicorn")
    assert uvicorn.required
    assert uvicorn.status is DoctorStatus.FAIL
    assert "[server]" in (uvicorn.remediation or "")


def test_training_profile_requires_torch_and_verl(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_modules(monkeypatch, missing=frozenset({"torch", "verl", "vllm", "sglang"}))
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.10.14")

    report = run_doctor("training")

    failures = {check.name for check in report.checks if check.status is DoctorStatus.FAIL}
    assert report.status is DoctorStatus.FAIL
    assert failures == {"training.module.torch", "training.module.verl"}


def test_project_checks_require_checkout_files_and_do_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_modules(monkeypatch, missing=frozenset({"torch", "vllm", "sglang"}))
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(doctor_module.os, "access", lambda path, mode: mode == os.W_OK)
    (tmp_path / "pyproject.toml").write_text("fixture\n", encoding="utf-8")
    before = tuple(tmp_path.rglob("*"))

    report = run_doctor(project=tmp_path)

    after = tuple(tmp_path.rglob("*"))
    assert report.status is DoctorStatus.FAIL
    assert report.project_path == str(tmp_path.resolve())
    assert before == after
    missing = {
        check.name
        for check in report.checks
        if check.required and check.status is DoctorStatus.FAIL
    }
    assert missing == {
        "project.file.readme-md",
        "project.file.examples-data-corpus-jsonl",
        "project.file.examples-data-qa-jsonl",
    }


def test_complete_project_report_has_stable_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _make_project(tmp_path)
    _set_modules(monkeypatch, missing=frozenset({"torch", "vllm", "sglang"}))
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.12.1")
    monkeypatch.setattr(doctor_module.os, "access", lambda path, mode: True)

    first = run_doctor(project=tmp_path)
    second = run_doctor(project=tmp_path)
    payload = json.loads(first.canonical_bytes())

    assert first.status is DoctorStatus.PASS
    assert first.canonical_bytes() == second.canonical_bytes()
    assert payload["profile"] == "core"
    assert payload["status"] == "pass"
    assert payload["project_path"] == str(tmp_path.resolve())


def test_python_version_and_report_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_modules(monkeypatch)
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.13.0")

    report = run_doctor()

    assert report.status is DoctorStatus.FAIL
    assert next(check for check in report.checks if check.name == "python.version").required
    with pytest.raises(ValidationError, match="status must reflect required checks"):
        DoctorReport(
            profile=DoctorProfile.CORE,
            status=DoctorStatus.PASS,
            checks=report.checks,
            package_version=report.package_version,
            python_version=report.python_version,
        )
    with pytest.raises(ValidationError):
        DoctorCheck(
            name="core.invalid",
            status=DoctorStatus.PASS,
            purpose="validate extra-field rejection",
            detail="valid",
            remediation=None,
            required=True,
            unexpected=True,
        )

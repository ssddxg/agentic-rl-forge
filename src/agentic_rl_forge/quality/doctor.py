from __future__ import annotations

import importlib.util
import os
import platform
from enum import Enum
from pathlib import Path
from typing import Final

from pydantic import Field, model_validator

from agentic_rl_forge import __version__
from agentic_rl_forge.contracts.base import ContractModel


class DoctorProfile(str, Enum):
    CORE = "core"
    SERVER = "server"
    TRAINING = "training"


class DoctorStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class DoctorCheck(ContractModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    status: DoctorStatus
    purpose: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    remediation: str | None = None
    required: bool


class DoctorReport(ContractModel):
    profile: DoctorProfile
    status: DoctorStatus
    checks: tuple[DoctorCheck, ...]
    package_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    project_path: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> DoctorReport:
        required_statuses = {check.status for check in self.checks if check.required}
        if DoctorStatus.FAIL in required_statuses:
            expected = DoctorStatus.FAIL
        elif DoctorStatus.WARN in required_statuses:
            expected = DoctorStatus.WARN
        else:
            expected = DoctorStatus.PASS
        if self.status is not expected:
            raise ValueError("status must reflect required checks")
        return self

    def exit_code(self, strict: bool) -> int:
        return int(strict and self.status is DoctorStatus.FAIL)


_CORE_MODULES: Final[tuple[tuple[str, str], ...]] = (
    ("agentic_rl_forge.contracts", "typed runtime contracts"),
    ("agentic_rl_forge.rollout", "agent rollout orchestration"),
    ("agentic_rl_forge.storage", "trajectory and artifact persistence"),
)
_SERVER_MODULES: Final[tuple[tuple[str, str], ...]] = (
    ("fastapi", "HTTP service application"),
    ("uvicorn", "HTTP service process"),
)
_TRAINING_MODULES: Final[tuple[tuple[str, str], ...]] = (
    ("torch", "model training runtime"),
    ("verl", "distributed reinforcement-learning training"),
)
_OPTIONAL_CORE_MODULES: Final[tuple[tuple[str, str], ...]] = (
    ("torch", "local GPU and tensor support"),
    ("vllm", "optional vLLM rollout serving"),
    ("sglang", "optional SGLang rollout serving"),
)
_PROJECT_FILES: Final[tuple[tuple[str, str], ...]] = (
    ("pyproject.toml", "Python package and dependency metadata"),
    ("README.md", "installation and usage documentation"),
    ("examples/data/corpus.jsonl", "offline retrieval example corpus"),
    ("examples/data/qa.jsonl", "offline question-answer example data"),
)


def run_doctor(
    profile: DoctorProfile | str = DoctorProfile.CORE,
    project: Path | None = None,
) -> DoctorReport:
    """Inspect a local installation without importing optional modules or writing files."""
    selected_profile = DoctorProfile(profile)
    python_version = platform.python_version()
    checks = [_python_check(python_version)]
    checks.extend(
        _module_check(module, purpose=purpose, required=True, profile="core")
        for module, purpose in _CORE_MODULES
    )

    if selected_profile is DoctorProfile.CORE:
        checks.extend(
            _module_check(module, purpose=purpose, required=False, profile="optional")
            for module, purpose in _OPTIONAL_CORE_MODULES
        )
    elif selected_profile is DoctorProfile.SERVER:
        checks.extend(
            _module_check(module, purpose=purpose, required=True, profile="server")
            for module, purpose in _SERVER_MODULES
        )
    else:
        checks.extend(
            _module_check(module, purpose=purpose, required=True, profile="training")
            for module, purpose in _TRAINING_MODULES
        )
        checks.extend(
            _module_check(module, purpose=purpose, required=False, profile="optional")
            for module, purpose in _OPTIONAL_CORE_MODULES[1:]
        )

    project_path: str | None = None
    if project is not None:
        resolved_project = project.expanduser().resolve()
        project_path = str(resolved_project)
        checks.extend(_project_checks(resolved_project))

    frozen_checks = tuple(checks)
    report_status = _report_status(frozen_checks)
    return DoctorReport(
        profile=selected_profile,
        status=report_status,
        checks=frozen_checks,
        package_version=__version__,
        python_version=python_version,
        project_path=project_path,
    )


def _python_check(version: str) -> DoctorCheck:
    try:
        major, minor = (int(part) for part in version.split(".", maxsplit=2)[:2])
        supported = major == 3 and 10 <= minor <= 12
    except (TypeError, ValueError):
        supported = False
    return DoctorCheck(
        name="python.version",
        status=DoctorStatus.PASS if supported else DoctorStatus.FAIL,
        purpose="run AgenticRLForge on a supported Python interpreter",
        detail=(
            f"Python {version} is supported."
            if supported
            else f"Python {version} is outside the supported 3.10-3.12 range."
        ),
        remediation=None if supported else "Install Python 3.10, 3.11, or 3.12.",
        required=True,
    )


def _module_check(
    module: str,
    *,
    purpose: str,
    required: bool,
    profile: str,
) -> DoctorCheck:
    try:
        available = importlib.util.find_spec(module) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        available = False
    if available:
        status = DoctorStatus.PASS
        detail = f"Python module {module!r} is available."
        remediation = None
    elif required:
        status = DoctorStatus.FAIL
        detail = f"Required Python module {module!r} is not available."
        remediation = _module_remediation(module, profile)
    else:
        status = DoctorStatus.WARN
        detail = f"Optional Python module {module!r} is not installed."
        remediation = _module_remediation(module, profile)
    normalized_name = module.replace("_", "-").replace(".", "-")
    return DoctorCheck(
        name=f"{profile}.module.{normalized_name}",
        status=status,
        purpose=purpose,
        detail=detail,
        remediation=remediation,
        required=required,
    )


def _module_remediation(module: str, profile: str) -> str:
    if profile == "server":
        return 'Install the server dependencies with `pip install "agentic-rl-forge[server]"`.'
    if profile == "training":
        return (
            f"Install {module!r} using the versions and platform instructions for your "
            "training environment."
        )
    if profile == "core":
        return "Reinstall AgenticRLForge and verify the installed dependency set."
    return f"Install {module!r} only if this optional capability is needed."


def _project_checks(project: Path) -> tuple[DoctorCheck, ...]:
    checks = []
    is_directory = project.is_dir()
    checks.append(
        DoctorCheck(
            name="project.directory",
            status=DoctorStatus.PASS if is_directory else DoctorStatus.FAIL,
            purpose="locate a complete AgenticRLForge source checkout",
            detail=(
                f"Project directory exists at {project}."
                if is_directory
                else f"Project directory does not exist at {project}."
            ),
            remediation=None if is_directory else "Provide the path to a complete checkout.",
            required=True,
        )
    )
    for relative_path, purpose in _PROJECT_FILES:
        path = project / relative_path
        exists = path.is_file()
        checks.append(
            DoctorCheck(
                name=("project.file." + relative_path.replace("/", "-").replace(".", "-").lower()),
                status=DoctorStatus.PASS if exists else DoctorStatus.FAIL,
                purpose=purpose,
                detail=(
                    f"Required project file exists: {relative_path}."
                    if exists
                    else f"Required project file is missing: {relative_path}."
                ),
                remediation=(
                    None
                    if exists
                    else "Use a complete source checkout or restore the missing project file."
                ),
                required=True,
            )
        )

    output_path = project / "artifacts"
    writable_path = output_path if output_path.is_dir() else project
    writable = writable_path.is_dir() and os.access(writable_path, os.W_OK)
    checks.append(
        DoctorCheck(
            name="project.artifacts-writable",
            status=DoctorStatus.PASS if writable else DoctorStatus.WARN,
            purpose="write local run artifacts and reports",
            detail=(
                f"Artifact output parent is writable: {writable_path}."
                if writable
                else f"Artifact output parent may not be writable: {writable_path}."
            ),
            remediation=(
                None
                if writable
                else "Choose a writable project or output directory before running workflows."
            ),
            required=False,
        )
    )
    return tuple(checks)


def _report_status(checks: tuple[DoctorCheck, ...]) -> DoctorStatus:
    required_statuses = {check.status for check in checks if check.required}
    if DoctorStatus.FAIL in required_statuses:
        return DoctorStatus.FAIL
    if DoctorStatus.WARN in required_statuses:
        return DoctorStatus.WARN
    return DoctorStatus.PASS

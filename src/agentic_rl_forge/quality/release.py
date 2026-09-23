from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import ClassVar

from agentic_rl_forge.contracts import (
    AuditFinding,
    AuditSeverity,
    ReleaseAuditReport,
)


class ReleaseAuditor:
    _REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    )
    _EXCLUDED_PARTS: ClassVar[frozenset[str]] = frozenset(
        {
            ".git",
            ".venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
        }
    )
    _TEXT_SUFFIXES: ClassVar[frozenset[str]] = frozenset(
        {
            "",
            ".md",
            ".py",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
            ".jsonl",
            ".sh",
            ".txt",
        }
    )

    def __init__(self, project: Path | str) -> None:
        self.project = Path(project).resolve()

    def run(
        self,
        *,
        strict_git: bool = True,
        run_checks: bool = False,
        build_wheel: bool = False,
    ) -> ReleaseAuditReport:
        findings = []
        findings.extend(self._required_files())
        findings.extend(self._metadata())
        findings.extend(self._documentation())
        findings.extend(self._sensitive_content())
        findings.extend(self._git(strict=strict_git))
        if run_checks:
            findings.extend(self._quality_commands())
        if build_wheel:
            findings.append(self._wheel_build())
        error_count = sum(item.severity is AuditSeverity.ERROR for item in findings)
        return ReleaseAuditReport(
            project_path=str(self.project),
            ready=error_count == 0,
            findings=tuple(findings),
            pass_count=sum(item.severity is AuditSeverity.PASS for item in findings),
            warning_count=sum(item.severity is AuditSeverity.WARNING for item in findings),
            error_count=error_count,
        )

    def _required_files(self) -> list[AuditFinding]:
        missing = [name for name in self._REQUIRED_FILES if not (self.project / name).is_file()]
        return [
            self._finding(
                "files.required",
                AuditSeverity.ERROR if missing else AuditSeverity.PASS,
                "required repository files are missing" if missing else "required files exist",
                missing,
            )
        ]

    def _metadata(self) -> list[AuditFinding]:
        path = self.project / "pyproject.toml"
        if not path.is_file():
            return []
        try:
            module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
            toml = importlib.import_module(module_name)
            payload = toml.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ImportError) as error:
            return [
                self._finding(
                    "metadata.pyproject",
                    AuditSeverity.ERROR,
                    "pyproject.toml cannot be parsed",
                    (str(error),),
                )
            ]
        project = payload.get("project", {})
        required = ("name", "version", "description", "readme", "requires-python", "license")
        missing = [field for field in required if not project.get(field)]
        findings = [
            self._finding(
                "metadata.pyproject",
                AuditSeverity.ERROR if missing else AuditSeverity.PASS,
                "project metadata is incomplete" if missing else "project metadata is complete",
                missing,
            )
        ]
        urls = project.get("urls")
        findings.append(
            self._finding(
                "metadata.urls",
                AuditSeverity.PASS if isinstance(urls, dict) and urls else AuditSeverity.WARNING,
                "project URLs are configured"
                if isinstance(urls, dict) and urls
                else "project URLs are not configured yet",
            )
        )
        package_init = self.project / "src/agentic_rl_forge/__init__.py"
        if package_init.is_file() and project.get("version"):
            match = re.search(
                r'^__version__\s*=\s*["\']([^"\']+)["\']',
                package_init.read_text(encoding="utf-8"),
                flags=re.M,
            )
            package_version = match.group(1) if match is not None else None
            findings.append(
                self._finding(
                    "metadata.version",
                    (
                        AuditSeverity.PASS
                        if package_version == project.get("version")
                        else AuditSeverity.ERROR
                    ),
                    "package and project versions match"
                    if package_version == project.get("version")
                    else "package and project versions differ",
                    tuple(
                        str(value)
                        for value in (project.get("version"), package_version)
                        if value is not None
                    ),
                )
            )
        license_text = (self.project / "LICENSE").read_text(encoding="utf-8", errors="ignore")
        apache = "Apache License" in license_text and "Version 2.0" in license_text
        findings.append(
            self._finding(
                "license.apache-2.0",
                AuditSeverity.PASS if apache else AuditSeverity.ERROR,
                "Apache-2.0 license text is present"
                if apache
                else "Apache-2.0 license text was not recognized",
            )
        )
        return findings

    def _documentation(self) -> list[AuditFinding]:
        readme = self.project / "README.md"
        if not readme.is_file():
            return []
        text = readme.read_text(encoding="utf-8")
        placeholders = [marker for marker in ("<repository-url>", "TODO", "TBD") if marker in text]
        findings = [
            self._finding(
                "docs.placeholders",
                AuditSeverity.ERROR if placeholders else AuditSeverity.PASS,
                "README contains release placeholders"
                if placeholders
                else "README has no release placeholders",
                placeholders,
            )
        ]
        missing_links = []
        documents = (readme, *sorted((self.project / "docs").glob("*.md")))
        for document in documents:
            content = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", content):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                relative = target.split("#", 1)[0]
                if relative and not (document.parent / relative).resolve().exists():
                    missing_links.append(f"{document.relative_to(self.project)} -> {target}")
        findings.append(
            self._finding(
                "docs.local-links",
                AuditSeverity.ERROR if missing_links else AuditSeverity.PASS,
                "local Markdown links are broken"
                if missing_links
                else "local Markdown links resolve",
                missing_links,
            )
        )
        claims = re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:%|x|\u00d7)\b",
            text,
            flags=re.I,
        )
        findings.append(
            self._finding(
                "docs.numeric-claims",
                AuditSeverity.WARNING if claims else AuditSeverity.PASS,
                "README contains numeric performance claims requiring evidence"
                if claims
                else "README contains no numeric performance claims",
                tuple(sorted(set(claims))),
            )
        )
        return findings

    def _sensitive_content(self) -> list[AuditFinding]:
        secret_patterns = {
            "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
            "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            "github-token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
            "openai-key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        }
        assistant_names = "(?:" + "|".join(("chat" + "gpt", "clau" + "de", "co" + "dex")) + ")"
        coauthor_marker = "co-" + "authored-by"
        automated_roles = "(?:" + "|".join(("bot", "assist" + "ant")) + ")"
        trace_pattern = re.compile(
            rf"generated by (?:ai|{assistant_names}|an? (?:ai |language-model )?"
            rf"(?:assistant|model))|(?:ai|machine)[ -]generated|"
            rf"{coauthor_marker}:\s*(?:{assistant_names}|.*\b{automated_roles}\b)|"
            + "人工"
            + "智能生成",
            flags=re.I,
        )
        secrets = []
        traces = []
        for path in self._text_files():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, 1):
                location = f"{path.relative_to(self.project)}:{line_number}"
                if any(pattern.search(line) for pattern in secret_patterns.values()):
                    secrets.append(location)
                if trace_pattern.search(line):
                    traces.append(location)
        return [
            self._finding(
                "security.secrets",
                AuditSeverity.ERROR if secrets else AuditSeverity.PASS,
                "possible committed secrets detected" if secrets else "no secret patterns found",
                secrets,
            ),
            self._finding(
                "provenance.generation-traces",
                AuditSeverity.ERROR if traces else AuditSeverity.PASS,
                "generation traces detected" if traces else "no generation traces found",
                traces,
            ),
        ]

    def _git(self, *, strict: bool) -> list[AuditFinding]:
        if not (self.project / ".git").exists():
            return [
                self._finding(
                    "git.repository",
                    AuditSeverity.ERROR if strict else AuditSeverity.WARNING,
                    "Git repository is missing",
                )
            ]
        head = self._command(("git", "rev-parse", "--verify", "HEAD"))
        status = self._command(("git", "status", "--porcelain"))
        remote = self._command(("git", "remote", "get-url", "origin"))
        findings = [
            self._finding(
                "git.initial-commit",
                AuditSeverity.PASS if head[0] == 0 else AuditSeverity.ERROR,
                "Git history contains a commit"
                if head[0] == 0
                else "Git repository has no initial commit",
            ),
            self._finding(
                "git.clean",
                AuditSeverity.PASS
                if status[0] == 0 and not status[1].strip()
                else (AuditSeverity.ERROR if strict else AuditSeverity.WARNING),
                "working tree is clean"
                if status[0] == 0 and not status[1].strip()
                else "working tree contains uncommitted changes",
                tuple(status[1].splitlines()[:20]),
            ),
            self._finding(
                "git.remote",
                (
                    AuditSeverity.PASS
                    if remote[0] == 0 and remote[1].strip()
                    else AuditSeverity.WARNING
                ),
                "origin remote is configured"
                if remote[0] == 0 and remote[1].strip()
                else "origin remote is not configured",
                (remote[1].strip(),) if remote[0] == 0 and remote[1].strip() else (),
            ),
        ]
        return findings

    def _quality_commands(self) -> list[AuditFinding]:
        commands: list[tuple[str, tuple[str, ...]]] = [
            ("quality.ruff", (sys.executable, "-m", "ruff", "check", ".")),
            (
                "quality.format",
                (
                    sys.executable,
                    "-m",
                    "ruff",
                    "format",
                    "--check",
                    "src",
                    "tests",
                    "examples",
                ),
            ),
            (
                "quality.mypy",
                (sys.executable, "-m", "mypy", "src", "examples/offline_pipeline.py"),
            ),
            ("quality.pytest", (sys.executable, "-m", "pytest", "-q")),
            ("quality.pip-check", (sys.executable, "-m", "pip", "check")),
            ("quality.pip-audit", (sys.executable, "-m", "pip_audit")),
        ]
        recipe_shell = self._find_recipe_shell()
        if recipe_shell is not None:
            commands.insert(
                4,
                (
                    "quality.recipe-syntax",
                    (recipe_shell, "-n", "recipes/verl/run_search_r1_grpo.sh"),
                ),
            )
        findings = []
        for code, command in commands:
            exit_code, output = self._command(command, timeout=300)
            findings.append(
                self._finding(
                    code,
                    AuditSeverity.PASS if exit_code == 0 else AuditSeverity.ERROR,
                    f"{' '.join(command)} passed"
                    if exit_code == 0
                    else f"{' '.join(command)} failed",
                    tuple(output.splitlines()[-20:]) if exit_code else (),
                )
            )
        if recipe_shell is None:
            findings.append(
                self._finding(
                    "quality.recipe-syntax",
                    AuditSeverity.WARNING,
                    "bash was not found; the verl recipe syntax check was skipped",
                )
            )
        return findings

    @staticmethod
    def _find_recipe_shell(platform_name: str | None = None) -> str | None:
        if (platform_name or sys.platform) != "win32":
            return shutil.which("bash")

        candidates: list[Path] = []
        git = shutil.which("git")
        if git is not None:
            candidates.append(Path(git).resolve().parent.parent / "bin" / "bash.exe")
        for variable, relative in (
            ("ProgramFiles", ("Git", "bin", "bash.exe")),
            ("ProgramFiles(x86)", ("Git", "bin", "bash.exe")),
            ("LOCALAPPDATA", ("Programs", "Git", "bin", "bash.exe")),
        ):
            root = os.environ.get(variable)
            if root:
                candidates.append(Path(root).joinpath(*relative))
        return next((str(path) for path in candidates if path.is_file()), None)

    def _wheel_build(self) -> AuditFinding:
        with tempfile.TemporaryDirectory(prefix="arf-wheel-") as directory:
            exit_code, build_output = self._command(
                (
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--wheel",
                    "--outdir",
                    str(Path(directory) / "dist"),
                    ".",
                ),
                timeout=300,
            )
            distributions = tuple(sorted((Path(directory) / "dist").glob("*")))
            wheels = tuple(path for path in distributions if path.suffix == ".whl")
            sdists = tuple(path for path in distributions if path.name.endswith(".tar.gz"))
            checks_ok = exit_code == 0 and len(wheels) == 1 and len(sdists) == 1
            evidence = [path.name for path in distributions]
            if checks_ok:
                check_code, check_output = self._command(
                    (
                        sys.executable,
                        "-m",
                        "twine",
                        "check",
                        "--strict",
                        *(str(path) for path in distributions),
                    ),
                    timeout=120,
                )
                checks_ok = check_code == 0
                if check_code != 0:
                    evidence.extend(check_output.splitlines()[-20:])
            if checks_ok:
                try:
                    with zipfile.ZipFile(wheels[0]) as wheel:
                        names = set(wheel.namelist())
                    required_members = {
                        "agentic_rl_forge/py.typed",
                    }
                    missing_members = sorted(required_members - names)
                except (OSError, zipfile.BadZipFile) as error:
                    missing_members = [str(error)]
                checks_ok = not missing_members
                evidence.extend(f"missing:{item}" for item in missing_members)
        return self._finding(
            "package.distributions",
            AuditSeverity.PASS if checks_ok else AuditSeverity.ERROR,
            "wheel and source distribution passed strict metadata checks"
            if checks_ok
            else "distribution build or metadata validation failed",
            tuple(evidence) if evidence else tuple(build_output.splitlines()[-20:]),
        )

    def _text_files(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self.project.rglob("*")
            if path.is_file()
            and not any(part in self._EXCLUDED_PARTS for part in path.parts)
            and not any(part.startswith(".tmp-") for part in path.parts)
            and path.suffix.casefold() in self._TEXT_SUFFIXES
            and path.stat().st_size <= 2 * 1024 * 1024
        )

    def _command(
        self,
        command: tuple[str, ...],
        *,
        timeout: int = 30,
    ) -> tuple[int, str]:
        environment = dict(os.environ)
        for name in tuple(environment):
            if name.startswith("COV_CORE_") or name == "COVERAGE_PROCESS_START":
                environment.pop(name)
        try:
            result = subprocess.run(
                command,
                cwd=self.project,
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return 1, str(error)
        return result.returncode, result.stdout + result.stderr

    @staticmethod
    def _finding(
        code: str,
        severity: AuditSeverity,
        message: str,
        evidence: list[str] | tuple[str, ...] = (),
    ) -> AuditFinding:
        return AuditFinding(
            code=code,
            severity=severity,
            message=message,
            evidence=tuple(evidence),
        )

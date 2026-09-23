from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import uvicorn
from click import unstyle
from typer.testing import CliRunner

import agentic_rl_forge.cli as cli_module
from agentic_rl_forge.cli import app

runner = CliRunner()


def test_top_level_version_alias() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.3.0" in result.stdout


def test_studio_command_builds_local_app_and_opens_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app_instance: Any, *, host: str, port: int) -> None:
        captured.update(app=app_instance, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(
        cli_module,
        "_start_studio_browser_opener",
        lambda url: captured.update(url=url),
    )
    data_dir = tmp_path / "my studio"

    result = runner.invoke(
        app,
        ["studio", "--data-dir", str(data_dir), "--port", "8765"],
    )

    assert result.exit_code == 0, result.output
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["url"] == "http://127.0.0.1:8765/"
    assert any(getattr(route, "path", None) == "/" for route in captured["app"].routes)
    assert "my studio" in result.stdout


def test_studio_command_requires_explicit_network_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denied = runner.invoke(
        app,
        ["studio", "--data-dir", str(tmp_path), "--host", "0.0.0.0", "--no-open"],
    )
    assert denied.exit_code != 0
    assert "--allow-network" in unstyle(denied.output)

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app_instance, *, host, port: captured.update(
            app=app_instance,
            host=host,
            port=port,
        ),
    )
    allowed = runner.invoke(
        app,
        [
            "studio",
            "--data-dir",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--allow-network",
            "--no-open",
        ],
    )

    assert allowed.exit_code == 0, allowed.output
    assert captured["host"] == "0.0.0.0"
    assert "url" not in captured
    assert "http://127.0.0.1:7860/" in allowed.stdout


@pytest.mark.parametrize(
    ("host", "expected"),
    (("localhost", True), ("127.0.0.1", True), ("::1", True), ("0.0.0.0", False)),
)
def test_loopback_host_detection(host: str, expected: bool) -> None:
    assert cli_module._is_loopback_host(host) is expected

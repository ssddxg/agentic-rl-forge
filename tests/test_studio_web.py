from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agentic_rl_forge.studio import create_studio_app


@pytest.mark.asyncio
async def test_studio_serves_packaged_spa_and_security_headers(tmp_path: Path) -> None:
    app = create_studio_app(tmp_path / "studio")
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client,
    ):
        home = await client.get("/")
        script = await client.get("/static/app.js")
        styles = await client.get("/static/styles.css")
        relative_script = await client.get("/app.js")
        relative_styles = await client.get("/styles.css")
        relative_icon = await client.get("/icon.svg")
        fallback = await client.get("/knowledge-bases/example")
        missing_api = await client.get("/api/v1/does-not-exist")

    assert home.status_code == 200
    assert home.headers["content-type"].startswith("text/html")
    assert "Agentic RL Forge Studio" in home.text
    assert home.headers["cache-control"] == "no-cache"
    assert "default-src 'self'" in home.headers["content-security-policy"]
    assert home.headers["x-content-type-options"] == "nosniff"
    assert home.headers["x-frame-options"] == "DENY"
    assert script.status_code == 200
    assert script.headers["content-type"].startswith(("text/javascript", "application/javascript"))
    assert "innerHTML" not in script.text
    assert styles.status_code == 200
    assert styles.headers["content-type"].startswith("text/css")
    assert relative_script.status_code == 200
    assert relative_script.text == script.text
    assert relative_styles.status_code == 200
    assert relative_styles.text == styles.text
    assert relative_icon.status_code == 200
    assert relative_icon.headers["content-type"].startswith("image/svg+xml")
    assert fallback.status_code == 200
    assert fallback.text == home.text
    assert missing_api.status_code == 404
    assert missing_api.headers["cache-control"] == "no-store"
    assert missing_api.json()["error"]["code"] == "not_found"

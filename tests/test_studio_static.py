from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

STATIC = Path(__file__).parents[1] / "src" / "agentic_rl_forge" / "studio" / "static"


class _StudioHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: list[str] = []
        self.stylesheets: list[str] = []
        self.language = ""
        self.csp = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(str(attributes["id"]))
        if tag == "html":
            self.language = str(attributes.get("lang", ""))
        if tag == "script" and attributes.get("src"):
            self.scripts.append(str(attributes["src"]))
        if tag == "link" and attributes.get("rel") == "stylesheet":
            self.stylesheets.append(str(attributes.get("href", "")))
        if tag == "meta" and attributes.get("http-equiv") == "Content-Security-Policy":
            self.csp = str(attributes.get("content", ""))


def _asset(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_studio_is_self_contained_and_accessible_by_default() -> None:
    parser = _StudioHTMLParser()
    html = _asset("index.html")
    parser.feed(html)

    assert parser.language == "zh-CN"
    assert parser.scripts == ["./app.js"]
    assert parser.stylesheets == ["./styles.css"]
    assert "default-src 'self'" in parser.csp
    assert "object-src 'none'" in parser.csp
    assert "unsafe-inline" not in parser.csp
    assert "unsafe-eval" not in parser.csp
    assert "https://" not in html
    assert "aria-live" in html
    assert 'role="tablist"' in html
    assert 'type="file"' in html and "multiple" in html
    assert "AI 问答" in html
    assert "原文搜索" in html
    assert "资料管理" in html
    assert "强化学习实验工作台" in html
    assert "请从本地服务打开工作台" in html


def test_javascript_uses_safe_dom_rendering_and_csrf_requests() -> None:
    javascript = _asset("app.js")

    forbidden_sinks = (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    )
    assert all(sink not in javascript for sink in forbidden_sinks)
    assert ".textContent" in javascript
    assert 'headers.set("X-CSRF-Token"' in javascript
    assert 'credentials: "same-origin"' in javascript
    assert 'cache: "no-store"' in javascript
    assert 'body.append("file"' in javascript

    expected_api_paths = (
        "/bootstrap",
        "/knowledge-bases",
        "/sources",
        "/reindex",
        "/jobs?knowledgeBaseId=",
        "/search",
        "/answer",
        "/model-settings",
        "/model-settings/test",
        "/rl/overview",
        "/rl/trajectory-demo",
        "/rl/offline-runs",
        "/rl/datasets",
    )
    assert all(path in javascript for path in expected_api_paths)


def test_javascript_only_references_existing_static_ids() -> None:
    parser = _StudioHTMLParser()
    parser.feed(_asset("index.html"))
    javascript = _asset("app.js")
    match = re.search(r"const ids = \[(.*?)\];", javascript, flags=re.DOTALL)
    assert match is not None
    referenced = set(re.findall(r'"([a-z][a-z0-9-]+)"', match.group(1)))

    assert referenced
    assert referenced <= parser.ids


def test_styles_include_themes_responsive_layout_and_motion_preference() -> None:
    styles = _asset("styles.css")

    assert ':root[data-theme="dark"]' in styles
    assert "@media (max-width: 760px)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert ".sidebar.is-open" in styles
    assert ".skeleton-result" in styles
    assert ".toast-region" in styles
    assert len(_asset("icon.svg")) > 100

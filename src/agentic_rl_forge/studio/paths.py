from __future__ import annotations

import os
import re
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _default_data_root() -> Path:
    override = os.environ.get("ARF_STUDIO_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        parent = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return parent / "AgenticRLForge" / "Studio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AgenticRLForge" / "Studio"
    parent = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return parent / "agentic-rl-forge" / "studio"


def validate_storage_id(value: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError("storage id contains unsafe characters")
    return value


@dataclass(frozen=True, slots=True)
class StudioPaths:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    @classmethod
    def default(cls) -> StudioPaths:
        return cls(_default_data_root())

    @property
    def database(self) -> Path:
        return self.root / "studio.sqlite3"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def indexes(self) -> Path:
        return self.root / "indexes"

    def source_directory(self, knowledge_base_id: str) -> Path:
        return self.uploads / validate_storage_id(knowledge_base_id)

    def source_path(self, knowledge_base_id: str, stored_name: str) -> Path:
        validate_storage_id(knowledge_base_id)
        if (
            Path(stored_name).name != stored_name
            or "/" in stored_name
            or "\\" in stored_name
            or not stored_name
        ):
            raise ValueError("stored filename is unsafe")
        path = self.source_directory(knowledge_base_id) / stored_name
        if path.parent != self.source_directory(knowledge_base_id):
            raise ValueError("stored filename escapes its source directory")
        return path

    def index_directory(self, knowledge_base_id: str) -> Path:
        return self.indexes / validate_storage_id(knowledge_base_id)

    def ensure(self) -> StudioPaths:
        for path in (self.root, self.uploads, self.indexes):
            path.mkdir(parents=True, exist_ok=True)
            with suppress(OSError):
                path.chmod(0o700)
        return self

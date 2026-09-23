#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
ARF="$PROJECT_ROOT/.venv/bin/arf"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

studio_is_ready() {
  [[ -x "$ARF" && -x "$PYTHON" ]] || return 1
  "$PYTHON" - <<'PY' >/dev/null 2>&1
import sys
from importlib.resources import files

import docx
import fastapi
import multipart
import pypdf

assets = files("agentic_rl_forge.studio").joinpath("static")
required = ("index.html", "styles.css", "app.js", "icon.svg")
supported = (3, 10) <= sys.version_info[:2] <= (3, 12)
raise SystemExit(0 if supported and all(assets.joinpath(name).is_file() for name in required) else 1)
PY
}

if ! studio_is_ready; then
  printf 'First run: installing AgenticRLForge...\n'
  "$SCRIPT_DIR/setup.sh" --runtime-only
fi

exec "$ARF" studio "$@"

#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_NAME="$(basename -- "$0")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
VENV_DIR="$PROJECT_ROOT/.venv"

runtime_only=false
install_all=false
recreate=false
skip_validation=false

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Create a project-local Python environment, install AgenticRLForge Studio, and
run the built-in installation checks.

Options:
  --runtime-only    Install the local Studio (kept for script compatibility).
  --all             Install every project extra, including data/object-store.
  --recreate        Rebuild .venv even when it is already usable.
  --skip-validation Install only; do not run arf doctor and arf demo.
  -h, --help        Show this help message.

The default installation includes the local Studio and document readers.
Supported Python versions are 3.10, 3.11, and 3.12.
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --runtime-only)
      runtime_only=true
      ;;
    --all)
      install_all=true
      ;;
    --recreate)
      recreate=true
      ;;
    --skip-validation)
      skip_validation=true
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "Unknown option: $1"
      ;;
  esac
  shift
done

if [[ "$runtime_only" == true && "$install_all" == true ]]; then
  fail "--runtime-only and --all cannot be used together."
fi

[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || \
  fail "pyproject.toml was not found at $PROJECT_ROOT. Run this script from a complete checkout."

if [[ -L "$VENV_DIR" ]]; then
  fail "$VENV_DIR is a symbolic link. Refusing to clear or replace a linked environment."
fi
if [[ -e "$VENV_DIR" && ! -d "$VENV_DIR" ]]; then
  fail "$VENV_DIR exists but is not a directory. Move it aside and try again."
fi

python_is_supported() {
  "$@" -c \
    'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' \
    >/dev/null 2>&1
}

venv_is_supported() {
  local python_path="$1"
  "$python_path" -c \
    'import os, sys
expected = os.path.normcase(os.path.realpath(sys.argv[1]))
actual = os.path.normcase(os.path.realpath(sys.prefix))
supported = (3, 10) <= sys.version_info[:2] <= (3, 12)
raise SystemExit(0 if supported and sys.prefix != sys.base_prefix and actual == expected else 1)' \
    "$VENV_DIR" >/dev/null 2>&1
}

python_version() {
  "$@" -c 'import platform; print(platform.python_version())'
}

PYTHON_CMD=()

try_python_command() {
  local command_name="$1"
  shift
  local command_path

  command_path="$(command -v "$command_name" 2>/dev/null || true)"
  [[ -n "$command_path" ]] || return 1
  if python_is_supported "$command_path" "$@"; then
    PYTHON_CMD=("$command_path" "$@")
    return 0
  fi
  return 1
}

find_supported_python() {
  local name selector

  for name in python3.12 python3.11 python3.10; do
    if try_python_command "$name"; then
      return 0
    fi
  done

  if command -v py >/dev/null 2>&1; then
    for selector in -3.12 -3.11 -3.10; do
      if try_python_command py "$selector"; then
        return 0
      fi
    done
  fi

  for name in python3 python; do
    if try_python_command "$name"; then
      return 0
    fi
  done
  return 1
}

VENV_PYTHON=""
VENV_ARF=""

find_venv_commands() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
    VENV_ARF="$VENV_DIR/bin/arf"
    return 0
  fi
  if [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
    VENV_ARF="$VENV_DIR/Scripts/arf.exe"
    return 0
  fi
  return 1
}

reuse_venv=false
if [[ "$recreate" == false ]] && find_venv_commands && venv_is_supported "$VENV_PYTHON"; then
  reuse_venv=true
  log "Reusing .venv with Python $(python_version "$VENV_PYTHON")"
fi

if [[ "$reuse_venv" == false ]]; then
  if [[ -d "$VENV_DIR" ]]; then
    warn "The existing .venv is incomplete, unsupported, or was explicitly marked for recreation."
  fi
  if ! find_supported_python; then
    fail "Python 3.10-3.12 was not found. Install Python 3.12, make it available as python3.12 (or py -3.12 on Windows), and rerun this script."
  fi

  log "Creating .venv with Python $(python_version "${PYTHON_CMD[@]}")"
  "${PYTHON_CMD[@]}" -m venv --clear "$VENV_DIR" || \
    fail "Could not create $VENV_DIR. On Debian/Ubuntu, install the matching python3-venv package."

  find_venv_commands || \
    fail "The environment was created, but its Python executable could not be found."
  venv_is_supported "$VENV_PYTHON" || \
    fail "The new environment is not a valid project-local Python 3.10-3.12 virtual environment."
fi

if [[ "$install_all" == true ]]; then
  extras="dev,studio,research,data,object-store,signing"
else
  # runtime_only is a backwards-compatible alias. The normal install is the end-user Studio.
  extras="studio"
fi
install_spec="${PROJECT_ROOT}[${extras}]"

log "Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip || fail "pip could not be upgraded. Check network and proxy settings."

log "Installing AgenticRLForge with extras: $extras"
"$VENV_PYTHON" -m pip install --editable "$install_spec" || \
  fail "Installation failed. Check the dependency error above and verify access to the Python package index."

log "Checking installed dependency consistency"
"$VENV_PYTHON" -m pip check || fail "Installed packages have incompatible dependencies."

if [[ "$skip_validation" == false ]]; then
  [[ -x "$VENV_ARF" ]] || fail "The arf command was not installed at $VENV_ARF."
  log "Running arf doctor"
  "$VENV_ARF" doctor --profile server --project "$PROJECT_ROOT" --strict || fail "arf doctor failed."

  log "Validating the local Studio installation"
  "$VENV_PYTHON" - <<'PY' || fail "Studio dependencies or packaged web assets are missing."
from importlib.resources import files

import docx
import fastapi
import multipart
import pypdf

assets = files("agentic_rl_forge.studio").joinpath("static")
required = ("index.html", "styles.css", "app.js", "icon.svg")
missing = [name for name in required if not assets.joinpath(name).is_file()]
raise SystemExit(f"Missing Studio assets: {', '.join(missing)}" if missing else 0)
PY

  log "Running the offline arf demo"
  "$VENV_ARF" demo || fail "arf demo failed."
fi

printf '\nSetup complete.\n'
if [[ "$VENV_PYTHON" == "$VENV_DIR/bin/python" ]]; then
  printf 'Activate it with: source %q\n' "$VENV_DIR/bin/activate"
else
  printf 'Activate it with the script under: %s\n' "$VENV_DIR/Scripts"
fi
printf 'Open the local app with: %q\n' "$SCRIPT_DIR/start-studio.sh"
printf 'Run all local quality checks with: %q\n' "$SCRIPT_DIR/check.sh"

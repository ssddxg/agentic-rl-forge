#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_NAME="$(basename -- "$0")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
VENV_DIR="$PROJECT_ROOT/.venv"

run_security=false
skip_package=false
DIST_DIR=""
DIST_PREFIX=""

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Run the local checks expected before pushing AgenticRLForge to GitHub:
dependency validation, lint, formatting, strict typing, tests, shell syntax,
and package metadata validation.

Options:
  --security       Also run pip-audit (requires vulnerability database access).
  --skip-package   Skip wheel/sdist build and twine metadata checks.
  -h, --help       Show this help message.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

run_step() {
  local label="$1"
  shift
  printf '\n==> %s\n' "$label"
  if ! "$@"; then
    fail "$label failed. Review the command output above."
  fi
}

cleanup() {
  if [[ -n "$DIST_DIR" && -d "$DIST_DIR" ]]; then
    case "$DIST_DIR" in
      "$DIST_PREFIX"*)
        rm -rf -- "$DIST_DIR"
        ;;
      *)
        printf 'WARNING: Refusing to remove unexpected temporary path: %s\n' "$DIST_DIR" >&2
        ;;
    esac
  fi
}
trap cleanup EXIT

while (($# > 0)); do
  case "$1" in
    --security)
      run_security=true
      ;;
    --skip-package)
      skip_package=true
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

[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || \
  fail "pyproject.toml was not found at $PROJECT_ROOT. Run this script from a complete checkout."
[[ ! -L "$VENV_DIR" ]] || fail "$VENV_DIR is a symbolic link; refusing to use it."

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON="$VENV_DIR/bin/python"
elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
  PYTHON="$VENV_DIR/Scripts/python.exe"
else
  fail "No project environment was found. Run $SCRIPT_DIR/setup.sh first."
fi

if ! "$PYTHON" -c \
  'import os, sys
expected = os.path.normcase(os.path.realpath(sys.argv[1]))
actual = os.path.normcase(os.path.realpath(sys.prefix))
supported = (3, 10) <= sys.version_info[:2] <= (3, 12)
raise SystemExit(0 if supported and sys.prefix != sys.base_prefix and actual == expected else 1)' \
  "$VENV_DIR" >/dev/null 2>&1; then
  fail ".venv is not a valid project-local Python 3.10-3.12 environment. Run $SCRIPT_DIR/setup.sh --recreate."
fi

required_modules=(ruff mypy pytest build twine)
if ! "$PYTHON" - "${required_modules[@]}" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print("Missing development modules: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
then
  fail "Development dependencies are incomplete. Rerun $SCRIPT_DIR/setup.sh (without --runtime-only)."
fi

cd -- "$PROJECT_ROOT"
started_at=$SECONDS

run_step "Checking dependency consistency" "$PYTHON" -m pip check
run_step "Linting Python sources" "$PYTHON" -m ruff check .
run_step "Checking Python formatting" "$PYTHON" -m ruff format --check src tests examples
run_step "Running strict type checks" "$PYTHON" -m mypy src examples/offline_pipeline.py
run_step "Running tests with coverage" \
  "$PYTHON" -m pytest --cov=agentic_rl_forge --cov-report=term-missing
run_step "Checking the verl shell recipe syntax" \
  bash -n recipes/verl/run_search_r1_grpo.sh

if [[ "$skip_package" == false ]]; then
  temp_base="${TMPDIR:-/tmp}"
  DIST_PREFIX="${temp_base%/}/arf-quality-dist."
  DIST_DIR="$(mktemp -d "${DIST_PREFIX}XXXXXX")" || \
    fail "Could not create a temporary package directory."
  run_step "Building wheel and source distribution" \
    "$PYTHON" -m build --outdir "$DIST_DIR"

  shopt -s nullglob
  package_files=("$DIST_DIR"/*)
  shopt -u nullglob
  ((${#package_files[@]} > 0)) || fail "Package build produced no files in $DIST_DIR."
  run_step "Validating package metadata" \
    "$PYTHON" -m twine check --strict "${package_files[@]}"
fi

if [[ "$run_security" == true ]]; then
  if ! "$PYTHON" -c 'import pip_audit' >/dev/null 2>&1; then
    fail "pip-audit is not installed. Rerun $SCRIPT_DIR/setup.sh (without --runtime-only)."
  fi
  run_step "Auditing installed dependencies" "$PYTHON" -m pip_audit
fi

printf '\nAll requested quality checks passed in %s seconds.\n' "$((SECONDS - started_at))"

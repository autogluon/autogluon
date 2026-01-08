#!/usr/bin/env bash
set -euo pipefail

# Get the directory of the script and always change to it
script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

PY="${PYTHON:-python}"

# Check if we're in Colab
IN_COLAB="$("$PY" - <<'PYCODE'
try:
    import google.colab  # type: ignore
    print("true")
except ImportError:
    print("false")
PYCODE
)"

# Set installation type based on environment
if [ "$IN_COLAB" == "true" ]; then
    EDITABLE="false"
    echo "Colab detected - forcing non-editable install"
else
    EDITABLE="true"
fi

# Handle user override of editable setting
while test $# -gt 0
do
    case "$1" in
        --non-editable) EDITABLE="false";;
        *) echo "Error: Unused argument: $1" >&2
           exit 1;;
    esac
    shift
done

# --- UV discovery (standalone or pip-installed) ---
# Priority: 1) $UV env var  2) uv on PATH  3) python -m uv
UV_BIN="${UV:-}"

# Helper: set UV_LAUNCH to how we should invoke uv.
_detect_uv() {
  # 1) Explicit path via $UV
  if [[ -n "$UV_BIN" && -x "$UV_BIN" ]]; then
    UV_LAUNCH=("$UV_BIN")
    return 0
  fi

  # 2) Standalone uv on PATH
  if command -v uv >/dev/null 2>&1; then
    UV_LAUNCH=("$(command -v uv)")
    return 0
  fi

  # 3) Pip/conda-installed module importable by Python
  if "$PY" - <<'PYCODE' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("uv") else 1)
PYCODE
  then
    UV_LAUNCH=("$PY" -m uv)
    return 0
  fi

  return 1
}

# Try to detect uv first
if ! _detect_uv; then
  # 4) If pip exists, install uv automatically (like the original script), then retry detection
  if command -v pip >/dev/null 2>&1 || "$PY" -m pip --version >/dev/null 2>&1; then
    echo "[INFO] 'uv' not found. Installing via pip..."
    "$PY" -m pip install "uv"
    # Retry detection after install
    if ! _detect_uv; then
      echo "[ERROR] 'uv' still not found after pip install. Check your Python environment." >&2
      exit 1
    fi
  else
    echo "[ERROR] 'uv' not found and pip is unavailable." >&2
    echo "Install standalone uv:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "OR install via pip once pip is available:  $PY -m pip install uv" >&2
    echo "OR set UV=/full/path/to/uv and retry." >&2
    exit 1
  fi
fi

# Convenience wrapper
uvpip() { "${UV_LAUNCH[@]}" pip "$@"; }

# Use uv to install packages
# TODO: We should simplify this by having a single setup.py at project root, and let user call `pip install -e .`
if [ "$EDITABLE" == "true" ]; then
  # Editable install (used outside Colab)
  uvpip install --refresh -e "common/[tests]"
  uvpip install -e "features/" -e "core/[all,tests]" -e "tabular/[all,tests]" -e "multimodal/[tests]" -e "timeseries/[all,tests]" -e "eda/" -e "autogluon/"
else
  # Non-editable install (forced in Colab)
  uvpip install --refresh "common/[tests]"
  uvpip install "features/" "core/[all,tests]" "tabular/[all,tests]" "multimodal/[tests]" "timeseries/[all,tests]" "eda/" "autogluon/"
fi

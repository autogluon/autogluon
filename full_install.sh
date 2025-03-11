#!/usr/bin/env bash
set -euo pipefail

# Get the directory of the script
script_dir=$(dirname "$0")

# Check if we're in Colab
IN_COLAB=$(python -c "
try:
    import google.colab
    print('true')
except ImportError:
    print('false')
")

# Set installation type based on environment
if [ "$IN_COLAB" == "true" ]; then
    EDITABLE="false"
    echo "Colab detected - forcing non-editable install"
else
    EDITABLE="true"
    cd "$script_dir"
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

# Check if uv is installed
if ! python -m pip show uv &> /dev/null; then
    echo "uv could not be found. Installing uv..."
    python -m pip install uv
fi

# Use uv to install packages
# TODO: We should simplify this by having a single setup.py at project root, and let user call `pip install -e .`
if [ "$EDITABLE" == "true" ]; then
  # Editable install (never used in Colab)
  python -m uv pip install --refresh -e common/[tests]
  python -m uv pip install -e core/[all,tests] -e features/ -e tabular/[all,tests] -e multimodal/[tests] -e timeseries/[all,tests] -e eda/ -e autogluon/
else
  # Non-editable install (always used in Colab)
  python -m uv pip install --refresh common/[tests]
  python -m uv pip install core/[all,tests] features/ tabular/[all,tests] multimodal/[tests] timeseries/[all,tests] eda/ autogluon/
fi

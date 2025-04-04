#!/usr/bin/env bash
set -euo pipefail

EDITABLE="true"

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
  # install common first to avoid bugs with parallelization
  python -m uv pip install --refresh -e common/[tests]

  # install the rest
  python -m uv pip install -e core/[all,tests] -e features/ -e tabular/[all,tests] -e multimodal/[tests] -e timeseries/[all,tests] -e eda/ -e autogluon/

else
  # install common first to avoid bugs with parallelization
  python -m uv pip install --refresh common/[tests]

  # install the rest
  python -m uv pip install core/[all,tests] features/ tabular/[all,tests] multimodal/[tests] timeseries/[all,tests] eda/ autogluon/
fi

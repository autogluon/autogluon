#!/usr/bin/env bash
set -euo pipefail

EDITABLE="true"
UV_FLAGS=""

while test $# -gt 0
do
    case "$1" in
        --non-editable) EDITABLE="false";;
        *) echo "Error: Unused argument: $1" >&2
           exit 1;;
    esac
    shift
done

# Check if we're in a Jupyter/Colab environment
IN_NOTEBOOK=$(python -c "
try:
    import google.colab
    print('colab')
except ImportError:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            print('jupyter')
        else:
            print('no')
    except (NameError, ImportError):
        print('no')
")

# If in Colab or Jupyter, use --system flag with uv
if [ "$IN_NOTEBOOK" != "no" ]; then
    echo "Detected $IN_NOTEBOOK environment. Using --system flag with uv."
    UV_FLAGS="--system"
fi

# Check if uv is installed
if ! python -m pip show uv &> /dev/null; then
    echo "uv could not be found. Installing uv..."
    python -m pip install uv
fi

# Use uv to install packages
# TODO: We should simplify this by having a single setup.py at project root, and let user call `pip install -e .`
if [ "$EDITABLE" == "true" ]; then
  # install common first to avoid bugs with parallelization
  python -m uv pip install --refresh $UV_FLAGS -e common/[tests]

  # install the rest
  python -m uv pip install $UV_FLAGS -e core/[all,tests] -e features/ -e tabular/[all,tests] -e multimodal/[tests] -e timeseries/[all,tests] -e eda/ -e autogluon/

else
  # install common first to avoid bugs with parallelization
  python -m uv pip install --refresh $UV_FLAGS common/[tests]

  # install the rest
  python -m uv pip install $UV_FLAGS core/[all,tests] features/ tabular/[all,tests] multimodal/[tests] timeseries/[all,tests] eda/ autogluon/
fi

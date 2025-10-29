function setup_build_env {
    python -m pip install --upgrade pip
    python -m pip install tox
    python -m pip install flake8
    python -m pip install bandit
    python -m pip install packaging
    python -m pip install ruff
}

function setup_pytorch_cuda_env {
    # Use PyTorch bundled CUDA/cuDNN libraries instead of system CUDA
    # PyTorch 2.8+ uses wheel variants with nvidia-cudnn in nvidia/cudnn/lib
    PYTORCH_LIB_PATHS=$(python -c '
import torch
import os
import sys

paths = []
torch_dir = os.path.dirname(torch.__file__)

# Add torch/lib (contains core PyTorch CUDA libraries)
torch_lib = os.path.join(torch_dir, "lib")
if os.path.exists(torch_lib):
    paths.append(torch_lib)

# Add nvidia/cudnn/lib (PyTorch 2.8+ wheel variants)
site_packages = os.path.dirname(torch_dir)
nvidia_cudnn_lib = os.path.join(site_packages, "nvidia", "cudnn", "lib")
if os.path.exists(nvidia_cudnn_lib):
    paths.append(nvidia_cudnn_lib)

print(":".join(paths))
')
    if [ -n "$PYTORCH_LIB_PATHS" ]; then
        echo "Using PyTorch bundled libraries from: $PYTORCH_LIB_PATHS"
        export LD_LIBRARY_PATH=$PYTORCH_LIB_PATHS:$LD_LIBRARY_PATH
    else
        echo "Warning: Could not find PyTorch library directories."
    fi

    # Fallback: Skip cuDNN compatibility check if libraries aren't found correctly
    # This is safe for PyTorch 2.8+ which bundles compatible cuDNN
    export PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1
}

function setup_build_contrib_env {
    python -m pip install --upgrade pip
    python -m pip install -r $(dirname "$0")/../../docs/requirements_doc.txt
    export AG_DOCS=1
    export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in MultiModalPredictor
}

function setup_benchmark_env {
    git clone https://github.com/autogluon/autogluon-bench.git
    cd autogluon-bench
    pip install -e ".[tests]"
    cd ..
    pip install pyarrow
    git clone https://github.com/autogluon/autogluon-dashboard.git
    pip install -e ./autogluon-dashboard
    pip install yq
    pip install s3fs
}

function setup_hf_model_mirror {
    pip install PyYAML
    SUB_FOLDER="$1"
    python $(dirname "$0")/setup_hf_model_mirror.py --model_list_file $(dirname "$0")/../../multimodal/tests/hf_model_list.yaml --sub_folder $SUB_FOLDER
}

function install_local_packages {
    while(($#)) ; do
        python -m pip install --upgrade -e $1
        shift
    done
}

function install_tabular {
    python -m pip install --upgrade pygraphviz
    install_local_packages "tabular/$1"
}

function install_tabular_platforms {
    # pygraphviz will be installed with conda in platform tests
    install_local_packages "tabular/$1"
}

function install_multimodal {
    source $(dirname "$0")/setup_mmcv.sh

    # launch different process for each test to make sure memory is released
    python -m pip install --upgrade pytest-xdist
    install_local_packages "multimodal/$1"
    setup_mmcv
    # python -m pip install --upgrade "mmocr<1.0"  # not compatible with mmcv 2.0
}

function install_all {
    install_local_packages "common/[tests]" "core/[all]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]" "eda/[tests]"
    install_multimodal "[tests]"
    install_local_packages "autogluon/"
}

function install_all_windows {
    install_local_packages "common/[tests]" "core/[all]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]" "eda/[tests]"
    install_multimodal "[tests]"
    install_local_packages "autogluon/"
}

function install_all_no_tests {
    install_local_packages "common/" "core/[all]" "features/" "tabular/[all]" "timeseries/[all]" "eda/"
    install_multimodal
    install_local_packages "autogluon/"
}

function build_pkg {
    pip install --upgrade setuptools wheel
    while(($#)) ; do
        cd "$1"/
        python setup.py sdist bdist_wheel
        cd ..
        shift
    done
}

function build_all {
    build_pkg "common" "core" "features" "tabular" "multimodal" "timeseries" "autogluon" "eda"
}

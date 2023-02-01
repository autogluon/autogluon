function setup_build_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install tox
    python3 -m pip install flake8
    python3 -m pip install "black>=22.3,<23.0"
    python3 -m pip install isort>=5.10
    python3 -m pip install bandit
}

function setup_build_contrib_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install -r $(dirname "$0")/../../docs/requirements_doc.txt
    python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
    export AG_DOCS=1
    export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in AutoMMPredictor
}

function setup_mxnet_gpu {
    python3 -m pip install mxnet-cu113==1.9.*
    export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
}

function setup_torch_gpu {
    # Security-patched torch.
    python3 -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
}

function setup_torch_cpu {
    # Security-patched torch
    python3 -m pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
}

function setup_torch_gpu_non_linux {
    pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
}

function setup_torch_cpu_non_linux {
    pip3 install torch==1.13.1 torchvision==0.14.1
}

function install_local_packages {
    while(($#)) ; do
        python3 -m pip install --upgrade -e $1
        shift
    done
}

function install_multimodal {
    # launch different process for each test to make sure memory is released
    python3 -m pip install --upgrade pytest-xdist
    install_local_packages "multimodal/$1"
    mim install mmcv-full --timeout 60
    python3 -m pip install --upgrade mmdet
    python3 -m pip install --upgrade mmocr
}

function install_all {
    install_local_packages "common/[tests]" "core/[all]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]" "eda/[tests]"
    install_multimodal "[tests]"
    install_local_packages "autogluon/"
}

function install_all_no_tests {
    install_local_packages "common/" "core/[all]" "features/" "tabular/[all]" "timeseries/[all]" "eda/"
    install_multimodal
    install_local_packages "autogluon/"
}

function build_all {
    for module in common core features tabular multimodal timeseries autogluon
    do
        cd "$module"/
        python setup.py sdist bdist_wheel
        cd ..
    done
}

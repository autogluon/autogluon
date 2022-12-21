function setup_build_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install tox
    python3 -m pip install flake8
    python3 -m pip install black>=22.3
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
    # Security-patched torch
    TORCH_URL=https://aws-pytorch-unified-cicd-binaries.s3.us-west-2.amazonaws.com/r1.12.1_sm/20221201-232940/bbd58c88fe74811ebc2c7225a308eeadfa42a7b9/torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl
    TORCHVISION_URL=https://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp38-cp38-linux_x86_64.whl
    python3 -m pip uninstall -y torch torchvision torchaudio torchdata
    python3 -m pip install --no-cache-dir -U ${TORCH_URL} ${TORCHVISION_URL}
}

function setup_torch_cpu {
    # Security-patched torch
    TORCH_URL=https://aws-pytorch-unified-cicd-binaries.s3.us-west-2.amazonaws.com/r1.12.1_sm/20221130-175350/98e79c6834c193ed3751a155c5309d441bf904e3/torch-1.12.1%2Bcpu-cp38-cp38-linux_x86_64.whl
    TORCHVISION_URL=https://download.pytorch.org/whl/cpu/torchvision-0.13.1%2Bcpu-cp38-cp38-linux_x86_64.whl
    python3 -m pip uninstall -y torch torchvision torchaudio torchdata
    python3 -m pip install --no-cache-dir -U ${TORCH_URL} ${TORCHVISION_URL}
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

function install_vision {
    python3 -m pip install --upgrade pytest-xdist  # launch different process for each test to avoid resource not being released by either mxnet or torch
    install_local_packages "vision/"
}

function install_all {
    install_local_packages "common/[tests]" "core/[all]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]" "eda/[tests]"
    install_multimodal "[tests]" # multimodal must be install before vision and text
    install_vision
    install_local_packages "text/" "autogluon/"
}

function install_all_no_tests {
    install_local_packages "common/" "core/[all]" "features/" "tabular/[all]" "timeseries/[all]" "eda/"
    install_multimodal # multimodal must be installed before vision and text
    install_vision
    install_local_packages "text/" "autogluon/"
}

function build_all {
    for module in common core features tabular multimodal text vision timeseries autogluon
    do
        cd "$module"/
        python setup.py sdist bdist_wheel
        cd ..
    done
}

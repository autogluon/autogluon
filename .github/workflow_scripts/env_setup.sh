function setup_build_env {
    python3 -m pip install --upgrade pip
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
    python3 -m pip install mxnet-cu112==1.9.*
    export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
}

function setup_torch_gpu {
    python3 -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
}

function install_common {
    python3 -m pip install --upgrade -e common/[tests]
}

function install_core {
    install_common
    python3 -m pip install --upgrade -e core/
}

function install_core_all {
    install_common
    python3 -m pip install --upgrade -e core/[all]
}

function install_core_all_tests {
    install_common
    python3 -m pip install --upgrade -e core/[all,tests]
}

function install_features {
    python3 -m pip install --upgrade -e features/
}

function install_eda {
    echo "python3 -m pip install --upgrade -e eda/"
}

function install_tabular {
    python3 -m pip install --upgrade -e tabular/[tests]
}

function install_tabular_all {
    python3 -m pip install --upgrade -e tabular/[all,tests]
}

function install_multimodal {
    # launch different process for each test to make sure memory is released
    python3 -m pip install --upgrade pytest-xdist
    python3 -m pip install --upgrade -e multimodal/[tests]
    mim install mmcv-full --timeout 60
    python3 -m pip install --upgrade mmdet
    python3 -m pip install --upgrade mmocr
}

function install_text {
    python3 -m pip install --upgrade -e text/
}

function install_vision {
    python3 -m pip install --upgrade pytest-xdist  # launch different process for each test to avoid resource not being released by either mxnet or torch
    python3 -m pip install --upgrade -e vision/
}

function install_timeseries {
    python3 -m pip install --upgrade -e timeseries/[all,tests]
}

function install_cloud {
    python3 -m pip install --upgrade pytest-xdist # Enable running tests in parallel for speedup
    python3 -m pip install --upgrade -e cloud/
}

function install_autogluon {
    python3 -m pip install --upgrade -e autogluon/
}

function install_all {
    install_common
    install_core_all
    install_features
    install_tabular_all
    install_multimodal
    install_text
    install_vision
    install_timeseries
    install_eda
    install_autogluon
}

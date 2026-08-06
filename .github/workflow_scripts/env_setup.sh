# Map a pushed ref name to the path it is published under on the docs bucket.
# Single source of truth shared by build_all_docs.sh and copy_docs.sh.
function docs_publish_path {
    local ref="$1"
    if [[ $ref == 'master' ]]
    then
        echo 'dev'
    elif [[ $ref == 'dev' ]]
    then
        echo 'dev-branch'
    elif [[ $ref =~ ^v([0-9]+)\.([0-9]+)\.[0-9]+$ ]]
    then
        # Release tag vX.Y.Z -> bare major.minor (/X.Y/); patch releases refresh the same page
        echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    elif [[ $ref =~ ^v[0-9] ]]
    then
        # Looks like a release tag but isn't vX.Y.Z (e.g. v1.7.0rc1). Refuse rather than
        # publish a stray prefix to the docs bucket.
        echo "Refusing to publish docs for non-release ref '$ref'" >&2
        return 1
    else
        echo "$ref"
    fi
}

function setup_build_env {
    python -m pip install --upgrade pip
    python -m pip install tox
    python -m pip install flake8
    python -m pip install bandit
    python -m pip install packaging
    # Read the Ruff version from .pre-commit-config.yaml to keep a single source of truth
    RUFF_VERSION=$(grep -A5 'astral-sh/ruff-pre-commit' .pre-commit-config.yaml | grep 'rev:' | head -n1 | sed 's/.*v//')
    python -m pip install ruff=="$RUFF_VERSION"
}

function setup_build_contrib_env {
    python -m pip install --upgrade pip
    python -m pip install -r $(dirname "$0")/../../docs/requirements_doc.txt
    export AG_DOCS=1
    export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in MultiModalPredictor
    unset LD_LIBRARY_PATH  # avoid cuDNN version conflicts with PyTorch's bundled cuDNN
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
    SCRIPT_DIR=$(dirname "$0")
    python ${SCRIPT_DIR}/setup_hf_model_mirror.py \
        --model_list_file ${SCRIPT_DIR}/../../multimodal/tests/hf_model_list.yaml \
        --dataset_list_file ${SCRIPT_DIR}/../../multimodal/tests/hf_dataset_list.yaml \
        --sub_folder $SUB_FOLDER
    # Set HF environment variables to use cached artifacts
    export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
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
    install_local_packages "common/[tests]" "features/" "core/[all]" "tabular/[all,tests]" "timeseries/[all,tests]"
    install_multimodal "[tests]"
    install_local_packages "autogluon/"
}

function install_all_windows {
    install_local_packages "common/[tests]" "features/" "core/[all]" "tabular/[all,tests]" "timeseries/[all,tests]"
    install_multimodal "[tests]"
    install_local_packages "autogluon/"
}

function install_all_no_tests {
    install_local_packages "common/" "features/" "core/[all]" "tabular/[all]" "timeseries/[all]"
    install_multimodal
    install_local_packages "autogluon/"
}

function build_pkg {
    # FIXME: https://github.com/open-mmlab/mmcv/issues/3325, remove cap once fixed
    pip install --upgrade "setuptools<82" wheel
    while(($#)) ; do
        cd "$1"/
        python setup.py sdist bdist_wheel
        cd ..
        shift
    done
}

function build_all {
    build_pkg "common" "features" "core" "tabular" "multimodal" "timeseries" "autogluon"
}

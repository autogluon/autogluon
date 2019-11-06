max_time = 120

stage("Unit Test") {
  node('linux-gpu') {
    ws('workspace/autugluon-py3') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # remove and create new env instead
        conda env update -n autogluon_py3 -f docs/build.yml
        conda activate autogluon_py3
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        # from https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version
        pip install --upgrade --force-reinstall --no-deps .
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
        export MPLBACKEND=Agg
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
        nosetests --with-timer --timer-ok 5 --timer-warning 20 -x -v tests/unittests
        """
      }
    }
  }
}

stage("Build Docs") {
  node('linux-gpu') {
    ws('workspace/autogluon-docs') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        conda env update -n autogluon_docs -f docs/build.yml
        conda activate autogluon_docs
        export PYTHONPATH=\${PWD}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
        git submodule update --init --recursive
        git clean -fx
        pip install git+https://github.com/d2l-ai/d2l-book
        cd docs && bash build_doc.sh

        if [[ ${env.BRANCH_NAME} == master ]]; then
            aws s3 mb s3://autogluon.mxnet.io/
            aws s3 sync --delete _build/html/ s3://autogluon.mxnet.io/ --acl public-read --cache-control max-age=7200
            echo "Uploaded doc to http://autogluon.mxnet.io"
        else
            aws s3 mb s3://autogluon-staging
            aws s3 sync --delete _build/html/ s3://autogluon-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
            echo "Uploaded doc to http://autogluon-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html"
        fi
        """
      }
    }
  }
}

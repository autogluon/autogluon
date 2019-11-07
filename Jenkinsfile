max_time = 120

stage("Unit Test") {
  'Python 3': {
    node('linux-gpu') {
      ws('workspace/autugluon-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # remove and create new env instead
          conda env create -n autogluon_py3 -f docs/build.yml
          conda activate autogluon_py3
          conda list
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          make clean
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
        conda env create -n autogluon_docs -f docs/build.yml
        conda activate autogluon_docs
        export PYTHONPATH=\${PWD}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
        git submodule update --init --recursive
        git clean -fx
        pip install git+https://github.com/d2l-ai/d2l-book
        cd docs && bash build_doc.sh
      }
    }
  }
}

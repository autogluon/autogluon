max_time = 120

stage("Unit Test") {
  parallel 'Python 2': {
    node('linux-gpu') {
      ws('workspace/autogluon-py2') {
        timeout(time: max_time, unit: 'MINUTES') {
            checkout scm
            VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
            sh """#!/bin/bash
            echo Done!
            """
        }
      }
    }
  },
  'Python 3': {
    node('linux-gpu') {
      ws('workspace/autogluon-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
            # old pip packages won't be cleaned: https://github.com/conda/conda/issues/5887
            # remove and create new env instead
            set -ex
            conda env remove -n gluon_cv_py2_test
            conda env create -n gluon_cv_py2_test -f tests/py2.yml
            conda env update -n gluon_cv_py2_test -f tests/py2.yml --prune
            conda activate gluon_cv_py2_test
            conda list
            export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
            make clean
            # from https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version
            pip install --upgrade --force-reinstall --no-deps .
            env
            export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
            export MPLBACKEND=Agg
            export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
          """
        }
      }
    }
  }
}

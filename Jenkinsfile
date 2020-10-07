max_time = 180

stage("Unit Test") {
  node('linux-gpu') {
    ws('workspace/autugluon-py3') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_py3 -f docs/build.yml
        conda activate autogluon_py3
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export MPLBACKEND=Agg
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

        pip uninstall -y autogluon
        pip uninstall -y autogluon.vision
        pip uninstall -y autogluon.text
        pip uninstall -y autogluon.mxnet
        pip uninstall -y autogluon.extra
        pip uninstall -y autogluon.tabular
        pip uninstall -y autogluon.core

        cd core/
        python3 -m pip install --upgrade -e .
        python3 -m pytest --junitxml=results.xml --runslow tests
        cd ..

        cd tabular/
        python3 -m pip install --upgrade -e .
        python3 -m pytest --junitxml=results.xml --runslow tests
        cd ..

        cd mxnet/
        python3 -m pip install --upgrade -e .
        python3 -m pytest --junitxml=results.xml --runslow tests
        cd ..

        cd extra/
        python3 -m pip install --upgrade -e .
        python3 -m pytest --junitxml=results.xml --runslow tests
        cd ..

        cd text/
        python3 -m pip install --upgrade -e .
        python3 -m pytest --junitxml=results.xml --runslow tests
        cd ..

        cd vision/
        python3 -m pip install --upgrade -e .
        python3 -m pytest --junitxml=results.xml --runslow tests
        cd ..

        cd autogluon/
        python3 -m pip install --upgrade -e .
        cd ..
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
        conda env update -n autogluon_docs -f docs/build_contrib.yml
        conda activate autogluon_docs
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
        python3 -m pip install --force-reinstall ipython==7.16

       pip uninstall -y autogluon
        pip uninstall -y autogluon.vision
        pip uninstall -y autogluon.text
        pip uninstall -y autogluon.mxnet
        pip uninstall -y autogluon.extra
        pip uninstall -y autogluon.tabular
        pip uninstall -y autogluon.core

        cd core/
        python3 -m pip install --upgrade -e .
        cd ..

        cd tabular/
        python3 -m pip install --upgrade -e .
        cd ..

        cd mxnet/
        python3 -m pip install --upgrade -e .
        cd ..

        cd extra/
        python3 -m pip install --upgrade -e .
        cd ..

        cd text/
        python3 -m pip install --upgrade -e .
        cd ..

        cd vision/
        python3 -m pip install --upgrade -e .
        cd ..

        cd autogluon/
        python3 -m pip install --upgrade -e .
        cd ..

        cd docs && bash build_doc.sh
        if [[ ${env.BRANCH_NAME} == master ]]; then
            aws s3 sync --delete _build/html/ s3://autogluon.mxnet.io/ --acl public-read --cache-control max-age=7200
            echo "Uploaded doc to http://autogluon.mxnet.io"
        else
            aws s3 sync --delete _build/html/ s3://autogluon-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
            echo "Uploaded doc to http://autogluon-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html"
        fi
        """

        if (env.BRANCH_NAME.startsWith("PR-")) {
          pullRequest.comment("Job ${env.BRANCH_NAME}-${env.BUILD_NUMBER} is done. \nDocs are uploaded to http://autogluon-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
        }
      }
    }
  }
}

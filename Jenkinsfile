max_time = 180

stage("Unit Test") {
  node('linux-gpu') {
    ws('workspace/autugluon-py3-v0_0_14') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_py3-v0_0_14 -f docs/build.yml
        conda activate autogluon_py3-v0_0_14
        python3 -m pip uninstall -y autogluon
        python3 -m pip uninstall -y autogluon-contrib-nlp
        python3 -m pip uninstall -y autogluon-core
        python3 -m pip uninstall -y autogluon-extra
        python3 -m pip uninstall -y autogluon-mxnet
        python3 -m pip uninstall -y autogluon-tabular
        python3 -m pip uninstall -y autogluon-text
        python3 -m pip uninstall -y autogluon-vision
        python3 -m pip uninstall -y autogluon.vision
        python3 -m pip uninstall -y autogluon.text
        python3 -m pip uninstall -y autogluon.mxnet
        python3 -m pip uninstall -y autogluon.extra
        python3 -m pip uninstall -y autogluon.tabular
        python3 -m pip uninstall -y autogluon.core
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export MPLBACKEND=Agg
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
        python3 -m pip install --upgrade --force-reinstall -e .
        python3 -m pip install pytest
        python3 -m pytest --junitxml=results.xml --runslow tests
        """
      }
    }
  }
}

stage("Build Docs") {
  node('linux-gpu') {
    ws('workspace/autogluon-docs-v0_0_14') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8

        index_update_str = ''
        if (env.BRANCH_NAME.startsWith("PR-")) {
            bucket = 'autogluon-staging'
            path = "${env.BRANCH_NAME}/${env.BUILD_NUMBER}"
            site = "${bucket}.s3-website-us-west-2.amazonaws.com/${path}"
            flags = '--delete'
            cacheControl = ''
        } else {
            isMaster = env.BRANCH_NAME == 'master'
            isDev = env.BRANCH_NAME == 'dev'
            bucket = 'autogluon.mxnet.io'
            path = isMaster ? 'dev' : isDev ? 'dev-branch' : "${env.BRANCH_NAME}"
            site = "${bucket}/${path}"
            flags = isMaster ? '' : '--delete'
            cacheControl = '--cache-control max-age=7200'
            if (isMaster) {
                index_update_str = """
                            aws s3 cp root_index.html s3://${bucket}/index.html --acl public-read ${cacheControl}
                            echo "Uploaded root_index.html s3://${bucket}/index.html"
                        """
            }
        }

        other_doc_version_text = 'Stable Version Documentation'
        other_doc_version_branch = 'stable'
        if (env.BRANCH_NAME == 'stable') {
            other_doc_version_text = 'Nightly Version Documentation'
            other_doc_version_branch = 'dev'
        }

        escaped_context_root = site.replaceAll('\\/', '\\\\/')

        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_docs-v0_0_14 -f docs/build_contrib.yml
        conda activate autogluon_docs-v0_0_14
        python3 -m pip uninstall -y autogluon
        python3 -m pip uninstall -y autogluon-contrib-nlp
        python3 -m pip uninstall -y autogluon-core
        python3 -m pip uninstall -y autogluon-extra
        python3 -m pip uninstall -y autogluon-mxnet
        python3 -m pip uninstall -y autogluon-tabular
        python3 -m pip uninstall -y autogluon-text
        python3 -m pip uninstall -y autogluon-vision
        python3 -m pip uninstall -y autogluon.vision
        python3 -m pip uninstall -y autogluon.text
        python3 -m pip uninstall -y autogluon.mxnet
        python3 -m pip uninstall -y autogluon.extra
        python3 -m pip uninstall -y autogluon.tabular
        python3 -m pip uninstall -y autogluon.core

        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
        python3 -m pip install --upgrade --force-reinstall -e .

        sed -i -e 's/###_PLACEHOLDER_WEB_CONTENT_ROOT_###/http:\\/\\/${escaped_context_root}/g' docs/config.ini
        sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_LABEL_###/${other_doc_version_text}/g' docs/config.ini
        sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_BRANCH_###/${other_doc_version_branch}/g' docs/config.ini

        cd docs && bash build_doc.sh
        aws s3 sync ${flags} _build/html/ s3://${bucket}/${path} --acl public-read ${cacheControl}
        echo "Uploaded doc to http://${site}/index.html"

        ${index_update_str}
        """

        if (env.BRANCH_NAME.startsWith("PR-")) {
          pullRequest.comment("Job ${env.BRANCH_NAME}-${env.BUILD_NUMBER} is done. \nDocs are uploaded to http://autogluon-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
        }
      }
    }
  }
}

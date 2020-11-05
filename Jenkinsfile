max_time = 180

stage("Unit Test") {
  parallel 'core': {
    node('linux-cpu') {
      ws('workspace/autogluon-core-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon_core_py3 -f docs/build.yml
          conda activate autogluon_core_py3
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
          pip uninstall -y autogluon-contrib-nlp

          cd core/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'tabular': {
    node('linux-gpu') {
      ws('workspace/autugluon-tabular-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon_tabular_py3 -f docs/build.yml
          conda activate autogluon_tabular_py3
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
          pip uninstall -y autogluon-contrib-nlp

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../tabular/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'mxnet': {
    node('linux-gpu') {
      ws('workspace/autugluon-mxnet-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon_mxnet_py3 -f docs/build.yml
          conda activate autogluon_mxnet_py3
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
          pip uninstall -y autogluon-contrib-nlp

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../extra/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'extra': {
    node('linux-gpu') {
      ws('workspace/autugluon-extra-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon_extra_py3 -f docs/build.yml
          conda activate autogluon_extra_py3
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
          pip uninstall -y autogluon-contrib-nlp

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          cd ../extra/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'text': {
    node('linux-gpu') {
      ws('workspace/autugluon-text-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon_text_py3 -f docs/build.yml
          conda activate autogluon_text_py3
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
          pip uninstall -y autogluon-contrib-nlp

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../tabular/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          cd ../text/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'vision': {
    node('linux-gpu') {
      ws('workspace/autugluon-vision-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon_vision_py3 -f docs/build.yml
          conda activate autogluon_vision_py3
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
          pip uninstall -y autogluon-contrib-nlp

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          cd ../extra/
          python3 -m pip install --upgrade -e .
          cd ../vision/
          python3 -m pip install --upgrade -e .
          python3 -m pip install https://github.com/dmlc/gluon-cv/archive/master.zip
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'install': {
    node('linux-cpu') {
      ws('workspace/autugluon-install-py3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon_install_py3 -f docs/build.yml
          conda activate autogluon_install_py3
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
          pip uninstall -y autogluon-contrib-nlp

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../tabular/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          cd ../text/
          python3 -m pip install --upgrade -e .
          cd ../extra/
          python3 -m pip install --upgrade -e .
          cd ../autogluon/
          python3 -m pip install --upgrade -e .
          cd ..
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
            other_doc_version_text = 'Dev Version Documentation'
            other_doc_version_branch = 'dev'
        }

        escaped_context_root = site.replaceAll('\\/', '\\\\/')

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
        pip uninstall -y autogluon-contrib-nlp
        pip uninstall -y autogluon-core
        pip uninstall -y autogluon-extra
        pip uninstall -y autogluon-mxnet
        pip uninstall -y autogluon-tabular
        pip uninstall -y autogluon-text
        pip uninstall -y autogluon-vision

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

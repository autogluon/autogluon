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
      ws('workspace/autogluon-tabular-py3') {
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
      ws('workspace/autogluon-mxnet-py3') {
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
      ws('workspace/autogluon-extra-py3') {
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
      ws('workspace/autogluon-text-py3') {
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
      ws('workspace/autogluon-vision-py3') {
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
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'install': {
    node('linux-cpu') {
      ws('workspace/autogluon-install-py3') {
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
          cd ../vision/
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

stage("Build Tutorials") {
  parallel 'course': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-course') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_tutorial_course -f docs/build_contrib.yml
        conda activate autogluon_tutorial_course
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/course
        shopt -s extglob
        rm -rf ./docs/tutorials/!(course)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        pwd
        stash includes: 'docs/_build/rst/tutorials/course/*', name: 'course'
      }
    }
  },
  'image_classification': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-image-classification') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_tutorial_image_classification -f docs/build_contrib.yml
        conda activate autogluon_tutorial_image_classification
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/image_classification
        shopt -s extglob
        rm -rf ./docs/tutorials/!(image_classification)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        cat docs/_build/rst/_static/d2l.js
        cat docs/_build/rst/conf.py
        tree -L 2 docs/_build/rst
        """
        stash includes: 'docs/_build/rst/tutorials/image_classification/*', name: 'image_classification'
      }
    }
  },
  'nas': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-nas') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_tutorial_nas -f docs/build_contrib.yml
        conda activate autogluon_tutorial_nas
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/nas
        shopt -s extglob
        rm -rf ./docs/tutorials/!(nas)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/nas/*', name: 'nas'
      }
    }
  },
  'object_detection': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-object-detection') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_tutorial_object_detection -f docs/build_contrib.yml
        conda activate autogluon_tutorial_object_detection
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/object_detection
        shopt -s extglob
        rm -rf ./docs/tutorials/!(object_detection)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        tree -L 2 docs/_build/rst
        pwd
        """
        echo pwd()
        stash includes: 'docs/_build/rst/tutorials/object_detection/*', name: 'object_detection'
      }
    }
  },
  'tabular_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-tabular') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_tutorial_tabular -f docs/build_contrib.yml
        conda activate autogluon_tutorial_tabular
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/tabular
        shopt -s extglob
        rm -rf ./docs/tutorials/!(tabular_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/tabular_prediction/*', name: 'tabular'
      }
    }
  },
  'text_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-text') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_tutorial_text -f docs/build_contrib.yml
        conda activate autogluon_tutorial_text
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/text
        shopt -s extglob
        rm -rf ./docs/tutorials/!(text_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/text_prediction/*', name: 'text'
      }
    }
  },
  'torch': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-torch') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon_tutorial_torch -f docs/build_contrib.yml
        conda activate autogluon_tutorial_torch
        conda list
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/torch
        shopt -s extglob
        rm -rf ./docs/tutorials/!(torch)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/torch/*', name: 'torch'
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

        unstash 'course'
        unstash 'image_classification'
        unstash 'nas'
        unstash 'object_detection'
        unstash 'tabular'
        unstash 'text'
        unstash 'torch'

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

        shopt -s extglob
        rm -rf ./docs/tutorials/!(index.rst)
        cd docs && d2lbook build rst && d2lbook build html
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

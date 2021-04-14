max_time = 180

setup_pip_venv = """
    rm -rf venv
    conda list

    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install -U pip
    python3 -m pip install -U setuptools wheel

    python3 -m pip install 'graphviz'
    python3 -m pip install 'jupyter-sphinx>=0.2.2'
    python3 -m pip install 'portalocker'
    python3 -m pip install 'nose'
    python3 -m pip install 'docutils'
    python3 -m pip install 'mu-notedown'
    python3 -m pip install 'flake8'
    python3 -m pip install 'awscli>=1.18.140'

    export MPLBACKEND=Agg
"""

setup_mxnet_gpu = """
    python3 -m pip install mxnet-cu101==1.7.0
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
    export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
"""

cleanup_venv = """
    deactivate
    rm -rf venv
"""

install_core = """
    python3 -m pip install --upgrade -e core/
"""

install_core_tests = """
    python3 -m pip install --upgrade -e core/[tests]
"""

install_features = """
    python3 -m pip install --upgrade -e features/
"""

install_mxnet = """
    python3 -m pip install --upgrade -e mxnet/
"""

install_extra = """
    python3 -m pip install --upgrade -e extra/
"""

install_tabular = """
    python3 -m pip install --upgrade -e tabular/
"""

install_tabular_all = """
    python3 -m pip install --upgrade -e tabular/[all]
"""

install_text = """
    python3 -m pip install --upgrade -e text/
"""

install_vision = """
    python3 -m pip install --upgrade -e vision/
"""

stage("Unit Test") {
  parallel 'core': {
    node('linux-cpu') {
      ws('workspace/autogluon-core-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon-core-py3-v3 -f docs/build.yml
          conda activate autogluon-core-py3-v3
          conda list
          ${setup_pip_venv}
          python3 -m pip install 'mxnet==1.7.0.*'
          env

          ${install_core_tests}
          cd core/
          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
          """
        }
      }
    }
  },
  'features': {
    node('linux-gpu') {
      ws('workspace/autogluon-features-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon-features-py3-v3 -f docs/build.yml
          conda activate autogluon-features-py3-v3
          conda list
          ${setup_pip_venv}
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          env

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../features/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
          """
        }
      }
    }
  },
  'tabular': {
    node('linux-gpu') {
      ws('workspace/autogluon-tabular-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon-tabular-py3-v3 -f docs/build.yml
          conda activate autogluon-tabular-py3-v3
          conda list
          ${setup_pip_venv}
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          env

          ${install_core}
          ${install_features}
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          ${install_tabular_all}
          ${install_mxnet}
          ${install_text}
          ${install_extra}
          ${install_vision}

          cd tabular/
          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
          """
        }
      }
    }
  },
  'mxnet': {
    node('linux-gpu') {
      ws('workspace/autogluon-mxnet-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon-mxnet-py3-v3 -f docs/build.yml
          conda activate autogluon-mxnet-py3-v3
          conda list
          ${setup_pip_venv}
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          env

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../extra/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
          """
        }
      }
    }
  },
  'extra': {
    node('linux-gpu') {
      ws('workspace/autogluon-extra-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon-extra-py3-v3 -f docs/build.yml
          conda activate autogluon-extra-py3-v3
          conda list
          ${setup_pip_venv}
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          env

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          cd ../extra/
          python3 -m pip install --upgrade -e .
          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
          """
        }
      }
    }
  },
  'text': {
    node('linux-gpu') {
      ws('workspace/autogluon-text-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon-text-py3-v3 -f docs/build.yml
          conda activate autogluon-text-py3-v3
          conda list
          ${setup_pip_venv}
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          env

          ${install_core}
          ${install_features}
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          ${install_tabular_all}
          ${install_mxnet}
          ${install_text}

          cd text/
          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
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
          conda env update -n autogluon-vision-py3 -f docs/build.yml
          conda activate autogluon-vision-py3
          conda list
          ${setup_pip_venv}
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          env

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../mxnet/
          python3 -m pip install --upgrade -e .
          cd ../extra/
          python3 -m pip install --upgrade -e .
          cd ../vision/
          python3 -m pip install --upgrade -e .

          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing

          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
          """
        }
      }
    }
  },
  'install': {
    node('linux-cpu') {
      ws('workspace/autogluon-install-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda env update -n autogluon-install-py3-v3 -f docs/build.yml
          conda activate autogluon-install-py3-v3
          conda list
          ${setup_pip_venv}
          python3 -m pip install 'mxnet==1.7.0.*'
          env

          cd core/
          python3 -m pip install --upgrade -e .
          cd ../features/
          python3 -m pip install --upgrade -e .
          cd ../tabular/
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          python3 -m pip install --upgrade -e .[all]
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
          ${cleanup_venv}
          """
        }
      }
    }
  }
}

stage("Build Tutorials") {
  parallel 'course': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-course-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon-tutorial-course-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-course-v3
        conda list
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        env

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/course
        shopt -s extglob
        rm -rf ./docs/tutorials/!(course)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        ${cleanup_venv}
        """
        pwd
        stash includes: 'docs/_build/rst/tutorials/course/*', name: 'course'
      }
    }
  },
  'image_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-image-classification-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon-tutorial-image-classification-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-image-classification-v3
        conda list
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        env

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/image_prediction
        shopt -s extglob
        rm -rf ./docs/tutorials/!(image_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        cat docs/_build/rst/_static/d2l.js
        cat docs/_build/rst/conf.py
        tree -L 2 docs/_build/rst
        ${cleanup_venv}
        """
        stash includes: 'docs/_build/rst/tutorials/image_prediction/*', name: 'image_prediction'
      }
    }
  },
  'nas': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-nas-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon-tutorial-nas-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-nas-v3
        conda list
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        env

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/nas
        shopt -s extglob
        rm -rf ./docs/tutorials/!(nas)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        ${cleanup_venv}
        """
        stash includes: 'docs/_build/rst/tutorials/nas/*', name: 'nas'
      }
    }
  },
  'object_detection': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-object-detection-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon-tutorial-object-detection-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-object-detection-v3
        conda list
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        env

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/object_detection
        shopt -s extglob
        rm -rf ./docs/tutorials/!(object_detection)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        tree -L 2 docs/_build/rst
        ${cleanup_venv}
        """
        stash includes: 'docs/_build/rst/tutorials/object_detection/*', name: 'object_detection'
      }
    }
  },
  'tabular_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-tabular-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon-tutorial-tabular-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-tabular-v3
        conda list
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        env
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
        export AG_DOCS=1
        env

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/tabular
        shopt -s extglob
        rm -rf ./docs/tutorials/!(tabular_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        ${cleanup_venv}
        """
        stash includes: 'docs/_build/rst/tutorials/tabular_prediction/*', name: 'tabular'
      }
    }
  },
  'text_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-text-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon-tutorial-text-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-text-v3
        conda list
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1

        env
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/text
        shopt -s extglob
        rm -rf ./docs/tutorials/!(text_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        ${cleanup_venv}
        """
        stash includes: 'docs/_build/rst/tutorials/text_prediction/*', name: 'text'
      }
    }
  },
  'torch': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-torch-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        conda env update -n autogluon-tutorial-torch-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-torch-v3
        conda list
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        env
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/torch
        shopt -s extglob
        rm -rf ./docs/tutorials/!(torch)
        python -c "import torchvision; print(torchvision.__file__.split('__init__.py')[0])" | xargs -I {} find {} -name "*.py" -type f -print0 | xargs -0 sed -i 's,http://yann.lecun.com/exdb/mnist,https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist,g'
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        ${cleanup_venv}
        """
        stash includes: 'docs/_build/rst/tutorials/torch/*', name: 'torch'
      }
    }
  }
}

stage("Build Docs") {
  node('linux-gpu') {
    ws('workspace/autogluon-docs-v3') {
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
        unstash 'image_prediction'
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
        ${setup_pip_venv}
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        env

        git clean -fx
        python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
        python3 -m pip install --force-reinstall ipython==7.16
        python3 -m pip install --upgrade jupyter_sphinx
        python3 -m pip install 'sphinx==3.2.0'
        python3 -m pip install 'sphinxcontrib-applehelp==1.0.2'
        python3 -m pip install 'sphinxcontrib-bibtex==1.0.0'
        python3 -m pip install 'sphinxcontrib-devhelp==1.0.2'
        python3 -m pip install 'sphinxcontrib-htmlhelp==1.0.3'
        python3 -m pip install 'sphinxcontrib-jsmath==1.0.1'
        python3 -m pip install 'sphinxcontrib-qthelp==1.0.3'
        python3 -m pip install 'sphinxcontrib-serializinghtml==1.1.4'
        python3 -m pip install 'sphinxcontrib-svg2pdfconverter==1.1.0'

        python3 -m pip install 'docutils<0.16'
        python3 -m pip list

        cd core/
        python3 -m pip install --upgrade -e .
        cd ..

        cd features/
        python3 -m pip install --upgrade -e .
        cd ..

        cd tabular/
        python3 -m pip install --upgrade -e .[all]
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
        ${cleanup_venv}
        """

        if (env.BRANCH_NAME.startsWith("PR-")) {
          pullRequest.comment("Job ${env.BRANCH_NAME}-${env.BUILD_NUMBER} is done. \nDocs are uploaded to http://autogluon-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
        }
      }
    }
  }
}

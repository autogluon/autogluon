.. include:: install-warning.rst

.. note::

  AutoGluon requires `Python <https://www.python.org/downloads/release/python-370/>`_ version 3.6, 3.7, or 3.8 (3.8 support is experimental).
  Linux and Mac are the only operating systems fully supported for now (Windows version will be available soon).
  For troubleshooting the installation process, you can check the `Installation FAQ <install.html#installation-faq>`_.


Select your preferences below and run the corresponding install commands:

.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

  .. container:: opt-group

     :title:`OS:`
     :act:`Linux`
     :opt:`Mac`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="linux">Linux.</div>
        <div class="mdl-tooltip" data-mdl-for="mac">Mac OSX.</div>

  .. container:: opt-group

     :title:`Version:`
     :act:`PIP`
     :opt:`Source`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="pip">PIP Release.</div>
        <div class="mdl-tooltip" data-mdl-for="source">Install AutoGluon from source.</div>


  .. container:: opt-group

     :title:`Backend:`
     :act:`CPU`
     :opt:`GPU`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="cpu">Build-in backend for CPU.</div>
        <div class="mdl-tooltip" data-mdl-for="gpu">Required to run on Nvidia GPUs.</div>

  .. admonition:: Command:

     .. container:: linux

        .. container:: pip

           .. container:: cpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"

                 # CPU version of pytorch has smaller footprint - see installation instructions in
                 # pytorch documentation - https://pytorch.org/get-started/locally/
                 python3 -m pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

                 python3 -m pip install --pre autogluon

           .. container:: gpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel

                 # Here we assume CUDA 10.1 is installed.  You should change the number
                 # according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0).
                 python3 -m pip install -U "mxnet_cu101<2.0.0"
                 python3 -m pip install --pre autogluon

        .. container:: source

           .. container:: cpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && ./full_install.sh

           .. container:: gpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel

                 # Here we assume CUDA 10.1 is installed.  You should change the number
                 # according to your own CUDA version (e.g. mxnet_cu102 for CUDA 10.2).
                 python3 -m pip install -U "mxnet_cu101<2.0.0"
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && ./full_install.sh

     .. container:: mac

        .. container:: pip

           .. container:: cpu

              .. note::

                 If you don't have them, please first install:
                 `XCode <https://developer.apple.com/xcode/>`_, `Homebrew <https://brew.sh>`_, `LibOMP <https://formulae.brew.sh/formula/libomp>`_.
                 Once you have Homebrew, LibOMP can be installed via:

                 .. code-block:: bash

                     brew install libomp

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"
                 python3 -m pip install --pre autogluon

           .. container:: gpu

              .. note::

                 GPU usage is not yet supported on Mac OSX, please use Linux to utilize GPUs in AutoGluon.

        .. container:: source

           .. container:: cpu

              .. note::

                 If you don't have them, please first install:
                 `XCode <https://developer.apple.com/xcode/>`_, `Homebrew <https://brew.sh>`_, `LibOMP <https://formulae.brew.sh/formula/libomp>`_.
                 Once you have Homebrew, LibOMP can be installed via:

                 .. code-block:: bash

                     brew install libomp

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && ./full_install.sh

           .. container:: gpu

              .. note::

                 GPU usage is not yet supported on Mac OSX , please use Linux to utilize GPUs in AutoGluon.


AutoGluon is modularized into `sub-modules <https://packaging.python.org/guides/packaging-namespace-packages/>`_ specialized for tabular, text, or image data. You can reduce the number of dependencies required by solely installing a specific sub-module via:  `python3 -m pip install <submodule>`, where `<submodule>` may be one of the following options:

- `autogluon.tabular` - only functionality for tabular data (TabularPredictor)
    - The default installation of `autogluon.tabular` standalone is a skeleton installation.
    - Install via `pip install autogluon.tabular[all]` to get the same installation of tabular as via `pip install autogluon`
    - Available optional dependencies: `lightgbm,catboost,xgboost,fastai`. These are included in `all`.
    - To run `autogluon.tabular` with only the optional LightGBM and CatBoost models for example, you can do: `pip install autogluon.tabular[lightgbm,catboost]`

    - Experimental optional dependency: `skex`. This will speedup KNN models by 25x in training and inference on CPU. Use `pip install autogluon.tabular[all,skex]` to enable, or `pip install "scikit-learn-intelex<2021.3"` after a standard installation of AutoGluon.
- `autogluon.vision` - only functionality for computer vision (ImagePredictor, ObjectDetector)
- `autogluon.text` - only functionality for natural language processing (TextPredictor)
- `autogluon.core` - only core functionality (Searcher/Scheduler) useful for hyperparameter tuning of arbitrary code/models.
- `autogluon.features` - only functionality for feature generation / feature preprocessing pipelines (primarily related to Tabular data).
- `autogluon.mxnet` - miscellaneous extra MXNet functionality.

To install a submodule from source, follow the instructions for installing the entire package from source but replace the line `cd autogluon && ./full_install.sh` with `cd autogluon && python3 -m pip install -e {SUBMODULE_NAME}/{OPTIONAL_DEPENDENCIES}`

- For example, to install `autogluon.tabular[lightgbm,catboost]` from source, the command would be: `cd autogluon && python3 -m pip install -e tabular/[lightgbm,catboost]`
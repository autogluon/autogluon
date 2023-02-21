.. note::

  AutoGluon requires `Python <https://www.python.org/downloads/release/python-399/>`_ version 3.8, 3.9 or 3.10.
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
     :opt:`Mac - Apple Silicon`
     :opt:`Windows`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="linux">Linux.</div>
        <div class="mdl-tooltip" data-mdl-for="mac">Mac OSX.</div>
        <div class="mdl-tooltip" data-mdl-for="mac-apple-silicon">Mac OSX - Apple Silicon.</div>
        <div class="mdl-tooltip" data-mdl-for="windows">Windows.</div>

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

        <div class="mdl-tooltip" data-mdl-for="cpu">Built-in backend for CPU.</div>
        <div class="mdl-tooltip" data-mdl-for="gpu">Required to run on Nvidia GPUs.</div>

  .. admonition:: Command:

     .. container:: linux

        .. container:: pip

           .. container:: cpu

              .. include:: install-cpu-pip.rst

           .. container:: gpu

              .. include:: install-gpu-pip.rst

        .. container:: source

           .. container:: cpu

              .. include:: install-cpu-source.rst

           .. container:: gpu

              .. include:: install-gpu-source.rst

     .. container:: mac

        .. container:: pip

           .. container:: cpu

              .. include:: install-macos-libomp.rst

              .. include:: install-cpu-pip.rst

           .. container:: gpu

              .. include:: install-macos-no-gpu.rst

        .. container:: source

           .. container:: cpu

              .. include:: install-macos-libomp.rst

              .. include:: install-cpu-source.rst

           .. container:: gpu

              .. include:: install-macos-no-gpu.rst

     .. container:: mac-apple-silicon

        .. container:: pip

           .. container:: cpu

              .. include:: install-macos-apple-silicon.rst

           .. container:: gpu

              .. include:: install-macos-no-gpu.rst

        .. container:: source

           .. container:: cpu

              .. include:: install-macos-apple-silicon.rst

           .. container:: gpu

              .. include:: install-macos-no-gpu.rst

     .. container:: windows

        .. container:: pip

           .. container:: cpu

              .. include:: install-windows-cpu.rst

              .. include:: install-cpu-pip.rst

           .. container:: gpu

              .. include:: install-windows-gpu.rst

              .. include:: install-gpu-pip-windows.rst

        .. container:: source

           .. container:: cpu

              .. include:: install-windows-cpu.rst

              .. include:: install-cpu-source.rst

           .. container:: gpu

              .. include:: install-windows-gpu.rst

              .. include:: install-gpu-source-windows.rst


AutoGluon is modularized into `sub-modules <https://packaging.python.org/guides/packaging-namespace-packages/>`_ specialized for tabular, text, image, or time series data. You can reduce the number of dependencies required by solely installing a specific sub-module via:  `python3 -m pip install <submodule>`, where `<submodule>` may be one of the following options:

- `autogluon.tabular` - functionality for tabular data (TabularPredictor)
    - The default installation of `autogluon.tabular` standalone is a skeleton installation.
    - Install via `pip install autogluon.tabular[all]` to get the same installation of tabular as via `pip install autogluon`
    - Available optional dependencies: `lightgbm,catboost,xgboost,fastai`. These are included in `all`.
    - Optional dependencies not included in `all`: `vowpalwabbit,imodels,skex`.
    - To run `autogluon.tabular` with only the optional LightGBM and CatBoost models for example, you can do: `pip install autogluon.tabular[lightgbm,catboost]`

    - Experimental optional dependency: `skex`. This will speedup KNN models by 25x in training and inference on CPU. Use `pip install autogluon.tabular[all,skex]` to enable, or `pip install "scikit-learn-intelex<2021.5"` after a standard installation of AutoGluon.
    - Optional Dependency: `vowpalwabbit`. This will install the VowpalWabbit package and allow you to fit VowpalWabbit in TabularPredictor.
    - Optional Dependency: `imodels`. This will install the imodels package and allow you to fit interpretable models in TabularPredictor.
- `autogluon.multimodal` - functionality for image, text, and multimodal problems. Focus on deep learning models.
    - To try object detection functionality using `MultiModalPredictor`, please install additional dependencies via `mim install mmcv-full`, `pip install mmdet` and `pip install pycocotools`. Note that Windows users should also install `pycocotools`` by: `pip install pycocotools-windows`, but it only supports python 3.6/3.7/3.8.
- `autogluon.timeseries` - only functionality for time series data (TimeSeriesPredictor)
- `autogluon.common` - helper functionality. Not useful standalone.
- `autogluon.core` - only core functionality (Searcher/Scheduler) useful for hyperparameter tuning of arbitrary code/models.
- `autogluon.features` - only functionality for feature generation / feature preprocessing pipelines (primarily related to Tabular data).

To install a submodule from source, follow the instructions for installing the entire package from source but replace the line `cd autogluon && ./full_install.sh` with `cd autogluon && python3 -m pip install -e {SUBMODULE_NAME}/{OPTIONAL_DEPENDENCIES}`

- For example, to install `autogluon.tabular[lightgbm,catboost]` from source, the command would be: `cd autogluon && python3 -m pip install -e tabular/[lightgbm,catboost]`

To install all AutoGluon optional dependencies:

`pip install autogluon && pip install autogluon.tabular[vowpalwabbit,imodels,skex]`

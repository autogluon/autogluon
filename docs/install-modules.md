AutoGluon is modularized into [sub-modules](https://packaging.python.org/guides/packaging-namespace-packages/) specialized for tabular, multimodal, or time series data. You can reduce the number of dependencies required by solely installing a specific sub-module via:  `pip install <submodule>`, where `<submodule>` may be one of the following options:

- `autogluon.tabular` - functionality for tabular data (TabularPredictor)
    - The default installation of `autogluon.tabular` standalone is a skeleton installation.
    - Install via `pip install autogluon.tabular[all]` to get the same installation of tabular as via `pip install autogluon`
    - Available optional dependencies: `lightgbm,catboost,xgboost,fastai`. These are included in `all`.
    - Optional dependencies not included in `all`: `vowpalwabbit,imodels,skex,skl2onnx`.
    - To run `autogluon.tabular` with only the optional LightGBM and CatBoost models for example, you can do: `pip install autogluon.tabular[lightgbm,catboost]`
    - Experimental optional dependency: `skex`. This will speedup KNN models by 25x in training and inference on CPU. Use `pip install autogluon.tabular[all,skex]` to enable, or `pip install "scikit-learn-intelex<2021.5"` after a standard installation of AutoGluon.
    - Optional Dependency: `vowpalwabbit`. This will install the VowpalWabbit package and allow you to fit VowpalWabbit in TabularPredictor.
    - Optional Dependency: `imodels`. This will install the imodels package and allow you to fit interpretable models in TabularPredictor.
    - Optional Dependency: `skl2onnx`. This will enable model compilation support via `predictor.compile_models()` on supported models.
- `autogluon.multimodal` - functionality for image, text, and multimodal problems. Focus on deep learning models.
    - To try object detection functionality using `MultiModalPredictor`, please install additional dependencies via `mim install mmcv-full`, `pip install mmdet` and `pip install pycocotools`. Note that Windows users should also install `pycocotools` by: `pip install pycocotools-windows`, but it only supports python 3.6/3.7/3.8.
- `autogluon.timeseries` - only functionality for time series data (TimeSeriesPredictor).
- `autogluon.eda` - only functionality for exploratory data analysis.
- `autogluon.common` - helper functionality. Not useful standalone.
- `autogluon.core` - only core functionality (Searcher/Scheduler) useful for hyperparameter tuning of arbitrary code/models.
- `autogluon.features` - only functionality for feature generation / feature preprocessing pipelines (primarily related to Tabular data).

To install a submodule from source, follow the instructions for installing the entire package from source but replace the line `cd autogluon && ./full_install.sh` with `cd autogluon && pip install -e {SUBMODULE_NAME}/{OPTIONAL_DEPENDENCIES}`

- For example, to install `autogluon.tabular[lightgbm,catboost]` from source, the command would be: `cd autogluon && pip install -e tabular/[lightgbm,catboost]`

To install all AutoGluon optional dependencies:

`pip install autogluon && pip install autogluon.tabular[vowpalwabbit,imodels,skex,skl2onnx]`

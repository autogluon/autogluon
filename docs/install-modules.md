AutoGluon is modularized into [sub-modules](https://packaging.python.org/guides/packaging-namespace-packages/) specialized for tabular, multimodal, or time series data. You can reduce the number of dependencies required by solely installing a specific sub-module via:  `pip install <submodule>`, where `<submodule>` may be one of the following options:

- `autogluon.tabular` - functionality for tabular data (TabularPredictor)
    - The default installation of `autogluon.tabular` standalone is a skeleton installation.
    - Install via `pip install autogluon.tabular[all]` to get the same installation of tabular as via `pip install autogluon`
    - Available optional dependencies: `lightgbm,catboost,xgboost,fastai,tabm,mitra,ray`. These are included in `all`.
    - Optional dependencies not included in `all`: `tabicl,tabpfn,tabdpt,tabpfnmix,realmlp,nori,interpret,imodels,skex,skl2onnx`.
    - To run `autogluon.tabular` with only the optional LightGBM and CatBoost models for example, you can do: `pip install autogluon.tabular[lightgbm,catboost]`
    - Install via `pip install autogluon.tabular[tabarena]` to get `all` plus every model the `extreme` preset uses: `tabdpt,tabicl,tabpfn,realmlp,nori`.
    - Optional dependency: `tabicl`. This will enable the TabICL model, used in the `extreme` preset (key=`TABICL`).
    - Optional dependency: `tabpfn`. This will enable the TabPFN models: RealTabPFN-v2 (key=`REALTABPFN-V2`), used in the `extreme` preset, along with RealTabPFN-v2.5 (key=`REALTABPFN-V2.5`), TabPFN-2.6 (key=`TABPFN-2.6`) and TabPFN-3 (key=`TABPFN-3`). The `TABPFNV2` key was renamed to `REALTABPFN-V2` in v1.5.0. RealTabPFN-v2 is free for commercial use; commercial use of RealTabPFN-v2.5, TabPFN-2.6 and TabPFN-3 requires a license from Prior Labs ([license FAQ](https://docs.priorlabs.ai/models#tabpfn-model-license)).
    - Optional dependency: `tabdpt`. This will enable the TabDPT model, used in the `extreme` preset (key=`TABDPT`), along with TabDPT-Turbo (key=`TABDPT-TURBO`).
    - Optional dependency: `tabm`. This will enable the TabM model, used in the `extreme` preset (key=`TABM`). Included in `all`.
    - Optional dependency: `mitra`. This will enable the Mitra model, used in the `extreme` preset (key=`MITRA`). Included in `all`.
    - Optional dependency: `tabpfnmix`. This will enable the TabPFNMix model (key=`TABPFNMIX`). Refer to `mitra`, which is an improved version of `tabpfnmix`.
    - Optional dependency: `realmlp`. This will enable the RealMLP model (key=`REALMLP`).
    - Optional dependency: `nori`. This will enable the Nori model, used in the `extreme` preset (key=`NORI`). Regression only.
    - Optional dependency: `skex`. This will speedup KNN models by 25x in training and inference on CPU. Use `pip install autogluon.tabular[all,skex]` to enable. Note: Not compatible with ARM processors.
    - Optional dependency: `interpret`. This will install the interpret package and allow you to fit EBM models (key=`EBM`).
    - Experimental optional dependency: `imodels`. This will install the imodels package and allow you to fit interpretable models in TabularPredictor.
    - Optional dependency: `skl2onnx`. This will enable ONNX model compilation via `predictor.compile()` on supported models.
- `autogluon.multimodal` - functionality for image, text, and multimodal problems. Focus on deep learning models.
    - To try object detection functionality using `MultiModalPredictor`, please install additional dependencies via `mim install "mmcv==2.1.0"`, `pip install "mmdet==3.2.0"` and `pip install pycocotools`. Note that Windows users should also install `pycocotools` by: `pip install pycocotools-windows`, but it only supports python 3.6/3.7/3.8.
- `autogluon.timeseries` - only functionality for time series data (TimeSeriesPredictor).
- `autogluon.common` - helper functionality. Not useful standalone.
- `autogluon.core` - only core functionality (Searcher/Scheduler) useful for hyperparameter tuning of arbitrary code/models.
- `autogluon.features` - only functionality for feature generation / feature preprocessing pipelines (primarily related to Tabular data).

To install a submodule from source, follow the instructions for installing the entire package from source but replace the final `uv sync --all-extras` line with `uv sync --package autogluon.{SUBMODULE_NAME} --extra {OPTIONAL_DEPENDENCY} ...` (sibling `autogluon.*` packages resolve automatically from the workspace). See [Installing from source](https://github.com/autogluon/autogluon/blob/master/docs/install-from-source.md) for details.

- For example, to install `autogluon.tabular[lightgbm,catboost]` from source, the command would be: `cd autogluon && uv sync --package autogluon.tabular --extra lightgbm --extra catboost`

To install all AutoGluon optional dependencies:

`pip install autogluon && pip install autogluon.tabular[all,tests]`

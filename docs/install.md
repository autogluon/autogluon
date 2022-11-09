# Installation

AutoGluon requires Python version 3.7, 3.8, or 3.9. 

::::{dropdown} Install Python by `conda`
You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [anaconda](https://www.anaconda.com/products/distribution)) to
manage your Python environments. Once installed, you can create an environment with a particular Python version and activate it:

```bash
conda create -n autogluon python=3.9 -y
conda activate autogluon
```

Then check your Python version

```bash
python --version
```
::::

The easiest way to install `AutoGluon` is through [pip](https://pip.pypa.io/en/stable/installation/). Some of AutoGluon's depended libraries require a recent version of `pip`. You can upgrade your pip by

```bash
python -m pip install --upgrade pip
```

Now install `autogluon` by

```bash
python -m pip install autogluon
```

````{tip}
You can use `pip` instead of `python -m pip`. The later guarantee to use the `pip` module installed for the python version you want to use. While for the former, you need to check it by yourself, especially when you have multiple python versions installed. 
````

````{dropdown} Install the CPU-only version
In default, AutoGluon installs with GPU supports on both Linux and Windows, which also works well for CPU. But when you only need to run on CPUs and concern about network speed and disk usage, then you can run the following command before installing `autogluon`. It installs the CPU-version pytorch, which is 10x smaller than the normal version with GPU supports.

```bash
python -m pip install "torch>=1.0,<1.12+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

````

::::{dropdown} Install `LibOMP` in MacOS.
Both AutoGluon's dependencies, LightGBM and XGBoost, need `libomp` for multi-threading. However, if you install it via `brew install libomp`, you may get segmentation faults. Here is a way to fix it:

```bash
# Uninstall libomp if it was previous installed
brew uninstall -f libomp
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew install libomp.rb
```
::::

::::{dropdown} Install nightly builds
AutoGluon offers nightly builds. You can install it by using the `--pre` argument

```bash
python -m pip install --pre autogluon
```

But note that nightly builds are less rigorously tested than stable releases.
::::


::::{dropdown} Install a particular module
In default, AutoGluon installs all its modules. If you only want to use a particular module, you can only install that module to accelerate the installation. For example, only install the `tabular` module:

```bash
python -m pip install autogluon.tabular
```

This modules depends on multiple ML libraries, you can even select some to install. For example, only install `lightgbm` and `catboost`:

```bash
python -m pip install autogluon.tabular[lightgbm,catboost]
```

Here is a list of all modules:

1. `autogluon.tabular`: functionality for tabular data (TabularPredictor)
   - Available optional dependencies: `lightgbm`,`catboost`,`xgboost`,`fastai`. These are included in the `all` option.
   - TODO the default one doesn't install above lib? we need to explicitly use `[all]`?
   - Optional dependencies not included in `all`: `vowpalwabbit` and `skex`. The later will speedup KNN models by 25x in training and inference on CPU   
1. `autogluon.multimodal` - functionality for image, text, and multimodal problems. Focus on deep learning models.
1. `autogluon.vision` [will deprecate by ??] - only functionality for computer vision (ImagePredictor, ObjectDetector)
1. `autogluon.text` [will deprecate by ??] only functionality for natural language processing (TextPredictor)
1. `autogluon.core` - only core functionality (Searcher/Scheduler) useful for hyperparameter tuning of arbitrary code/models.
1. `autogluon.features` - only functionality for feature generation / feature preprocessing pipelines (primarily related to Tabular data).

::::

::::{dropdown} Install from source

Building from source is similar to installing by `pip` for prerequisites. But instead of using `pip install`, now you clone the repo and install by the script:

```bash
git clone https://github.com/awslabs/autogluon
cd autogluon && ./full_install.sh
```

::::

````{admonition} Trouble Shots

Please ask questions at github issue.

(TODO), add some common issues about windows and macos here?
m1: https://github.com/awslabs/autogluon/issues/1242
windows: https://github.com/awslabs/autogluon/issues/164
````
# Installation

AutoGluon requires Python version 3.7, 3.8, or 3.9. 

::::{dropdown} Optional: Use `conda` to manage Python environment
You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://www.anaconda.com/products/distribution) to
manage your Python environment. Once installed, create an environment with a specific Python version and activate it:

```bash
conda create -n autogluon python=3.9 -y
conda activate autogluon
```
::::

[pip](https://pip.pypa.io/en/stable/installation/) is the primary way to install AutoGluon, and a recent version is required by AutoGluon's dependencies:

```bash
python -m pip install --upgrade pip
```

Install AutoGluon:

```bash
python -m pip install autogluon
```

````{tip}
You can use `pip` instead of `python -m pip`. The latter guarantees the use of the `pip` module from your active Python installation.
````

::::{dropdown} Install a specific AutoGluon module
By default, AutoGluon installs all modules. You can accelerate installation by installing specific modules:

```bash
python -m pip install autogluon.tabular
```

You can also specify optional dependencies:

```bash
python -m pip install autogluon.tabular[lightgbm,catboost]
```

Modules and optional dependencies:

1. `autogluon.tabular`: functionality for tabular data (TabularPredictor)
   - Optional dependencies included by default: `lightgbm`,`catboost`,`xgboost`,`fastai`.
   - Optional dependencies not included by default: `vowpalwabbit`, `skex`. The later will speedup KNN training and inference on CPU by 25x.
1. `autogluon.multimodal`: functionality for image, text, and multimodal data (MultiModalPredictor)
1. `autogluon.core`: core functionality including HPO support
1. `autogluon.features`: functionality for feature generation and preprocessing

::::

::::{dropdown} CPU-only install
By default, AutoGluon is installed with GPU support on both Linux and Windows, and this version also works well for CPU. If you only need CPU support and are concerned about network and disk usage, run the following command before installing AutoGluon, which installs the CPU-only version of `pytorch` that is 10x smaller than the default version with GPU support:

```bash
python -m pip install "torch>=1.0,<1.13+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
::::

::::{dropdown} M1 Apple Silicon
Running AutoGluon on M1 Macs requires additional setup steps. See this [github issue](https://github.com/awslabs/autogluon/issues/1242#issuecomment-1285276870) for specific instructions. M2 Macs are currently not supported, but support is being tracked [here](https://github.com/awslabs/autogluon/issues/2271).
::::

::::{dropdown} `LibOMP` on MacOS
AutoGluon dependencies LightGBM and XGBoost use `libomp` for multi-threading. If you install `libomp` via `brew install libomp`, you may get segmentation faults due to incompatible library versions. Install a compatible version using these commands:

```bash
# Uninstall libomp if it was previous installed
brew uninstall -f libomp
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew install libomp.rb
```
::::

::::{dropdown} Install nightly builds
AutoGluon offers nightly builds that can be installed using the `--pre` argument. Nightly builds have the latest features but have not been as rigurously tested as stable releases.

```bash
python -m pip install --pre autogluon
```
::::


::::{dropdown} Install from source

To build AutoGluon from source, make sure `pip` is updated, clone the source repository, and run the install script:

```bash
python -m pip install --upgrade pip
git clone https://github.com/awslabs/autogluon
cd autogluon && ./full_install.sh
```

::::

````{admonition} Trouble Shooting

If you encounter installation issues not covered here, please create a [github issue](https://github.com/awslabs/autogluon/issues).
````
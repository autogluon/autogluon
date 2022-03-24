# Training models with GPU support
:label:`sec_tabulargpu`

Training with GPU can significantly speed up base algorithms, and is a necessity for text and vision models where training without GPU is infeasibly slow. 
CUDA toolkit is required for GPU training. Please refer to the [official documentation](https://docs.nvidia.com/cuda/) for the installation instructions.

```{.python}
predictor = TabularPredictor(label=label).fit(
    train_data,
    ag_args_fit={'num_gpus': 1}
)
```

To enable GPU acceleration on only specific models, the same parameter can be passed into model `hyperparameters`:

```{.python}
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # Train with CPU
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=hyperparameters, 
)
```

## Multi-modal

In :ref:`sec_tabularprediction_multimodal` tutorial we presented how to train an ensemble which can utilize tabular, text and images. 
If available GPUs don't have enough VRAM to fit the default model, or it is needed to speedup testing, different backends can be used:

Regular configuration is retrieved like this:

```{.python .input}
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
hyperparameters
```

### Text models

Text model preset to use can be set via:

```{.python}
hyperparameters['AG_TEXT_NN'] = ['<preset>']
```

Available text model presets:


```{.python .input}
from autogluon.text.text_prediction.presets import list_text_presets
list_text_presets()
```

### Vision models

Text model preset to use can be set via:

```{.python}
hyperparameters['AG_IMAGE_NN'] = {'model': '<model>'}
```

The list of available text model presets is:

```{.python .input}
from autogluon.vision.predictor.predictor import _get_supported_models
_get_supported_models()[:10]  # there're more, we just show a few
```

## Enabling GPU for LightGBM

The default installation of LightGBM does not support GPU training, however GPU support can be enabled via a special install. If `num_gpus` is set, the following warning will be displayed:

```
Warning: GPU mode might not be installed for LightGBM, GPU training raised an exception. Falling back to CPU training...Refer to LightGBM GPU documentation: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-versionOne possible method is:	pip uninstall lightgbm -y	pip install lightgbm --install-option=--gpu
```

If the suggested commands do not work, uninstall existing lightgbm `pip uninstall -y lightgbm` and install from sources following the instructions in the [official guide](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html). The
optional [Install Python Interface](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#install-python-interface-optional) section is also required to make it work with AutoGluon.

## Troubleshooting

```
OSError: libcudnn.so.X: cannot open shared object file: No such file or directory
OSError: libcudart.so.XX.Y: cannot open shared object file: No such file or directory
```

This might happen when installed cuda is not matching MXNet library. To resolve it, get the cuda version installed in system: `nvcc --version` or `nvidia-smi`. Then install matching `mxnet-cuXXX` package (CUDA `11.0` -> `mxnet-cu110`, etc.)

```
pip install 'mxnet-cu110<2.0.0'
```

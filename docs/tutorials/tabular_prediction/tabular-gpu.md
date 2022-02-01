# Training models with GPU support
:label:`sec_tabulargpu`

Training with GPU can significantly speed up base algorithms, but it is also a necessity for text and vision models where even on small tasks and the best CPUs the training can take forever without GPU acceleration. To enable GPU training, pass `num_gpus` parameter into a `fit()` call. All models
with GPU support will automatically start using the acceleration.

```{.python}
save_path = 'agModels-predictClass'
predictor = TabularPredictor(label=label, path=save_path).fit(
    train_data,
    ag_args_fit={'num_gpus': 1}
)
```

If GPU acceleration should be enabled just on a specific models, the same parameter can be passed into model `hyperparameters`:

```{.python}
save_path = 'agModels-predictClass'  # specifies folder to store trained models
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # Train with CPU
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU
    ]
}
predictor = TabularPredictor(label=label, path=save_path).fit(
    train_data, 
    hyperparameters=hyperparameters, 
)
```

## Multi-modal

In :ref:`sec_tabularprediction_multimodal` tutorial we presented how to train an ensemble which can utilize tabular, text and images. If available GPU don't have enough RAM to handle default model sizes or it is needed to speedup testing, different backends can be used:

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

The list of available text model presets is:

```{.python .input}
from autogluon.text.text_prediction.presets import ag_text_presets
ag_text_presets
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

When installed AutoGluon, out-of-the box it is using the LightGBM from a default installation. If `num_gpus` is set, the following warning will be displayed:

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

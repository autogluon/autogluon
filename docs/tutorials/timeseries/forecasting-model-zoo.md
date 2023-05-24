# Forecasting Time Series - Model Zoo

:::{note}
This documentation is intended for advanced users and may not be comprehensive.

For a stable public API, refer to TimeSeriesPredictor.
:::

This page contains the list of time series forecasting models available in AutoGluon.
The available hyperparameters for each model are listed under **Other Parameters**.

This list is useful if you want to override the default hyperparameters ([Manually configuring models](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html#manually-configuring-models))
or define custom hyperparameter search spaces ([Hyperparameter tuning](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html#hyperparameter-tuning)), as described in the In-depth Tutorial.
For example, the following code will train a `TimeSeriesPredictor` with `DeepAR` and `ETS` models with default hyperparameters (and a weighted ensemble on top of them):

```
predictor = TimeSeriesPredictor().fit(
   train_data,
   hyperparameters={
      "DeepAR": {},
      "ETS": {},
   },
)
```

Note that we don't include the `Model` suffix when specifying the model name in `hyperparameters`
(e.g., the class {class}`~autogluon.timeseries.models.DeepARModel` corresponds to the name `"DeepAR"` in the `hyperparameters` dictionary).

Also note that some of the models' hyperparameters have names and default values that
are different from the original libraries.

## Default models

```{eval-rst}
.. automodule:: autogluon.timeseries.models
```

```{eval-rst}
.. currentmodule:: autogluon.timeseries.models
```

```{eval-rst}
.. autosummary::
   :nosignatures:

   NaiveModel
   SeasonalNaiveModel
   ARIMAModel
   ETSModel
   ThetaModel
   AutoETSModel
   AutoARIMAModel
   DynamicOptimizedThetaModel
   DirectTabularModel
   DeepARModel
   SimpleFeedForwardModel
   TemporalFusionTransformerModel

```

### {hidden}`NaiveModel`

```{eval-rst}
.. autoclass:: NaiveModel
   :members: init
```

### {hidden}`SeasonalNaiveModel`

```{eval-rst}
.. autoclass:: SeasonalNaiveModel
   :members: init

```

### {hidden}`ARIMAModel`

```{eval-rst}
.. autoclass:: ARIMAModel
   :members: init

```

### {hidden}`ETSModel`

```{eval-rst}
.. autoclass:: ETSModel
   :members: init

```

### {hidden}`ThetaModel`

```{eval-rst}
.. autoclass:: ThetaModel
   :members: init
```

### {hidden}`AutoETSModel`

```{eval-rst}
.. autoclass:: AutoETSModel
   :members: init
```

### {hidden}`AutoARIMAModel`

```{eval-rst}
.. autoclass:: AutoARIMAModel
   :members: init
```

### {hidden}`DynamicOptimizedThetaModel`

```{eval-rst}
.. autoclass:: DynamicOptimizedThetaModel
   :members: init
```

### {hidden}`DirectTabularModel`

```{eval-rst}
.. autoclass:: DirectTabularModel
   :members: init

```

### {hidden}`DeepARModel`

```{eval-rst}
.. autoclass:: DeepARModel
   :members: init

```

### {hidden}`SimpleFeedForwardModel`

```{eval-rst}
.. autoclass:: SimpleFeedForwardModel
   :members: init

```

### {hidden}`TemporalFusionTransformerModel`

```{eval-rst}
.. autoclass:: TemporalFusionTransformerModel
   :members: init


```

## MXNet Models

Following MXNet-based models from GluonTS are available in AutoGluon.

- `DeepARMXNetModel`
- `MQCNNMXNetModel`
- `MQRNNMXNetModel`
- `SimpleFeedForwardMXNetModel`
- `TemporalFusionTransformerMXNetModel`
- `TransformerMXNetModel`

Documentation and hyperparameter settings for these models can be found [here](https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/models/gluonts/mx/models.py).

Using the above models requires installing Apache MXNet v1.9. This can be done as follows:

```
python -m pip install mxnet~=1.9
```

If you want to use a GPU, install the version of MXNet that matches your CUDA version. See the
MXNet [documentation](https://mxnet.apache.org/versions/1.9.1/get_started?) for more info.

If a GPU is available and MXNet version with CUDA is installed, all the MXNet models will be trained using the GPU.
Otherwise, the models will be trained on CPU.

## Additional features

Overview of the additional features and covariates supported by different models.
Models not included in this table currently do not support any additional features.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :align: center
   :widths: 40 15 15 15 15

   * - Model
     - Static features (continuous)
     - Static features (categorical)
     - Known covariates (continuous)
     - Past covariates (continuous)
   * - :class:`~autogluon.timeseries.models.DirectTabularModel`
     - ✓
     - ✓
     - ✓
     - ✓
   * - :class:`~autogluon.timeseries.models.DeepARModel`
     - ✓
     - ✓
     - ✓
     -
   * - :class:`~autogluon.timeseries.models.TemporalFusionTransformerModel`
     - ✓
     - ✓
     - ✓
     - ✓
   * - :class:`~autogluon.timeseries.models.gluonts.mx.DeepARMXNetModel`
     - ✓
     - ✓
     - ✓
     -
   * - :class:`~autogluon.timeseries.models.gluonts.mx.MQCNNMXNetModel`
     - ✓
     - ✓
     - ✓
     - ✓
   * - :class:`~autogluon.timeseries.models.gluonts.mx.TemporalFusionTransformerMXNetModel`
     - ✓
     -
     - ✓
     -
```

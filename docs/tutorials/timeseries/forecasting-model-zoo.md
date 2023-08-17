# Forecasting Time Series - Model Zoo

:::{note}
This documentation is intended for advanced users and may not be comprehensive.

For a stable public API, refer to the [documentation for `TimeSeriesPredictor`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html).
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

The model names in the `hyperparameters` dictionary don't have to include the `"Model"` suffix
(e.g., both `"DeepAR"` and `"DeepARModel"` correspond to {class}`~autogluon.timeseries.models.DeepARModel`).

Note that some of the models' hyperparameters have names and default values that are different from the original libraries.



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
   AverageModel
   SeasonalAverageModel
   ARIMAModel
   ETSModel
   AutoARIMAModel
   AutoETSModel
   ThetaModel
   DynamicOptimizedThetaModel
   DirectTabularModel
   RecursiveTabularModel
   DeepARModel
   DLinearModel
   PatchTSTModel
   SimpleFeedForwardModel
   TemporalFusionTransformerModel

```

## Baseline models

### {hidden}`Naive`

```{eval-rst}
.. autoclass:: NaiveModel
   :members: init
```

### {hidden}`SeasonalNaive`

```{eval-rst}
.. autoclass:: SeasonalNaiveModel
   :members: init

```

### {hidden}`Average`

```{eval-rst}
.. autoclass:: AverageModel
   :members: init
```

### {hidden}`SeasonalAverage`

```{eval-rst}
.. autoclass:: SeasonalAverageModel
   :members: init

```

## Statistical models

### {hidden}`ARIMA`

```{eval-rst}
.. autoclass:: ARIMAModel
   :members: init

```

### {hidden}`ETS`

```{eval-rst}
.. autoclass:: ETSModel
   :members: init

```

### {hidden}`AutoARIMA`

```{eval-rst}
.. autoclass:: AutoARIMAModel
   :members: init
```

### {hidden}`AutoETS`

```{eval-rst}
.. autoclass:: AutoETSModel
   :members: init
```

### {hidden}`Theta`

```{eval-rst}
.. autoclass:: ThetaModel
   :members: init
```


### {hidden}`DynamicOptimizedTheta`

```{eval-rst}
.. autoclass:: DynamicOptimizedThetaModel
   :members: init
```

### {hidden}`NPTS`

```{eval-rst}
.. autoclass:: NPTSModel
   :members: init

```

## Deep learning models

### {hidden}`DeepAR`

```{eval-rst}
.. autoclass:: DeepARModel
   :members: init

```

### {hidden}`DLinear`

```{eval-rst}
.. autoclass:: DLinearModel
   :members: init

```

### {hidden}`PatchTST`

```{eval-rst}
.. autoclass:: PatchTSTModel
   :members: init

```

### {hidden}`SimpleFeedForward`

```{eval-rst}
.. autoclass:: SimpleFeedForwardModel
   :members: init

```

### {hidden}`TemporalFusionTransformer`

```{eval-rst}
.. autoclass:: TemporalFusionTransformerModel
   :members: init


```

## Tabular models

### {hidden}`DirectTabular`

```{eval-rst}
.. autoclass:: DirectTabularModel
   :members: init

```

### {hidden}`RecursiveTabular`

```{eval-rst}
.. autoclass:: RecursiveTabularModel
   :members: init

```

## MXNet Models

MXNet models from GluonTS have been deprecated because of dependency conflicts caused by MXNet.


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
   * - :class:`~autogluon.timeseries.models.RecursiveTabularModel`
     - ✓
     - ✓
     - ✓
     -
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
```

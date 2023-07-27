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

### {hidden}`AutoARIMAModel`

```{eval-rst}
.. autoclass:: AutoARIMAModel
   :members: init
```

### {hidden}`AutoETSModel`

```{eval-rst}
.. autoclass:: AutoETSModel
   :members: init
```

### {hidden}`ThetaModel`

```{eval-rst}
.. autoclass:: ThetaModel
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

### {hidden}`RecursiveTabularModel`

```{eval-rst}
.. autoclass:: RecursiveTabularModel
   :members: init

```

### {hidden}`DeepARModel`

```{eval-rst}
.. autoclass:: DeepARModel
   :members: init

```

### {hidden}`DLinearModel`

```{eval-rst}
.. autoclass:: DLinearModel
   :members: init

```

### {hidden}`PatchTSTModel`

```{eval-rst}
.. autoclass:: PatchTSTModel
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

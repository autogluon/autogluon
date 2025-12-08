(forecasting_model_zoo)=
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



## Overview

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
   ZeroModel
   ETSModel
   AutoARIMAModel
   AutoETSModel
   AutoCESModel
   ThetaModel
   ADIDAModel
   CrostonModel
   IMAPAModel
   NPTSModel
   DeepARModel
   DLinearModel
   PatchTSTModel
   SimpleFeedForwardModel
   TemporalFusionTransformerModel
   TiDEModel
   WaveNetModel
   DirectTabularModel
   PerStepTabularModel
   RecursiveTabularModel
   Chronos2Model
   ChronosModel
   TotoModel

```

## Baseline models

Baseline models are simple approaches that use minimal historical data to make predictions. They serve as benchmarks for evaluating more complex methods.

```{eval-rst}
.. autoclass:: NaiveModel
   :members: init
```


```{eval-rst}
.. autoclass:: SeasonalNaiveModel
   :members: init

```


```{eval-rst}
.. autoclass:: AverageModel
   :members: init
```


```{eval-rst}
.. autoclass:: SeasonalAverageModel
   :members: init

```


```{eval-rst}
.. autoclass:: ZeroModel
   :members: init

```

## Statistical models

Statistical models capture simple patterns in the data like trends and seasonality.


```{eval-rst}
.. autoclass:: ETSModel
   :members: init

```


```{eval-rst}
.. autoclass:: AutoARIMAModel
   :members: init
```


```{eval-rst}
.. autoclass:: AutoETSModel
   :members: init
```


```{eval-rst}
.. autoclass:: AutoCESModel
   :members: init
```


```{eval-rst}
.. autoclass:: ThetaModel
   :members: init
```


```{eval-rst}
.. autoclass:: NPTSModel
   :members: init

```


## Statistical models for sparse data

Statistical models that are built specifically for sparse and nonnegative data, especially for use
in intermittent demand forecasting.


```{eval-rst}
.. autoclass:: ADIDAModel
   :members: init
```


```{eval-rst}
.. autoclass:: CrostonModel
   :members: init
```


```{eval-rst}
.. autoclass:: IMAPAModel
   :members: init
```

## Deep learning models

Deep learning models use neural networks to capture complex patterns in the data.

```{eval-rst}
.. autoclass:: DeepARModel
   :members: init

```


```{eval-rst}
.. autoclass:: DLinearModel
   :members: init

```


```{eval-rst}
.. autoclass:: PatchTSTModel
   :members: init

```


```{eval-rst}
.. autoclass:: SimpleFeedForwardModel
   :members: init

```


```{eval-rst}
.. autoclass:: TemporalFusionTransformerModel
   :members: init


```


```{eval-rst}
.. autoclass:: TiDEModel
   :members: init


```


```{eval-rst}
.. autoclass:: WaveNetModel
   :members: init


```

## Tabular models

Tabular models convert time series forecasting into a tabular regression problem.


```{eval-rst}
.. autoclass:: DirectTabularModel
   :members: init

```


```{eval-rst}
.. autoclass:: PerStepTabularModel
   :members: init

```


```{eval-rst}
.. autoclass:: RecursiveTabularModel
   :members: init

```


## Pretrained models

Deep learning models pretrained on large time series datasets, able to perform zero-shot forecasting.


```{eval-rst}
.. autoclass:: Chronos2Model
   :members: init

```


```{eval-rst}
.. autoclass:: ChronosModel
   :members: init

```


```{eval-rst}
.. autoclass:: TotoModel
   :members: init

```


## Hyperparameters shared by all models

- **target_scaler** *({"standard", "mean_abs", "robust", "min_max", None}, default = None)* - If provided, each time
   series will be scaled as `(y - loc) / scale` before being passed to the model for training / prediction. An inverse
   transformation `y * scale + loc` will be applied to the predictions.

   Note that `loc` and `scale` are computed separately for each individual time series.

   Available options:
   - `"standard"` - standard scaler, `loc = mean(y)`, `scale = std(y)`
   - `"mean_abs"` - mean absolute scaler, `loc = 0`, `scale = mean(abs(y))`
   - `"robust"` - robust scaler, `loc = median(y)`, `scale = quantile(y, 0.75) - quantile(y, 0.25)`
   - `"min_max"` - min-max scaler that converts data into the (0, 1) range, `loc = min(y)`, `scale = max(y) - min(y)`.
   - `None` - no scaling

- **covariate_scaler** *({"global", None})* - If provided, the chosen scaling method will be applied to the covariates
   and static features before fitting the model.

   Such scaling be helpful for deep learning models that assume that the inputs are normalized.

   Available options:
   - `"global"` - `QuantileTransform` for skewed features, passthrough for boolean features, and `StandardScaler` for the rest of the features
   - `None` - do not scale the covariates

   By default, this parameter is set to `"global"` for GluonTS models, and `None` for all other models.

- **covariate_regressor** *({"LR", "GBM", "CAT", "XGB", "RF", None}, default = None)* - If provided, the chosen tabular
   regression model will be fit on the known covariates & static features to predict the target column at the same time
   step.

   The predictions of the regression model will be subtracted from the target column, and the forecasting model will
   be used to forecast the residuals.

   At prediction time, the predictions of the regression model will be added to the predictions of the forecasting model.

   If you enable the `covariate_regressor`, it is recommended to also enable the `target_scaler`. This will usually
   lead to better accuracy and faster fitting time for the regressor.

   If both a `target_scaler` and a `covariate_regressor` are provided, then scaling will be performed before the
   regressor is applied.


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
   :widths: 55 15 15 15

   * - Model
     - Static features (continuous + categorical)
     - Known covariates (continuous + categorical)
     - Past covariates (continuous + categorical)
   * - :class:`~autogluon.timeseries.models.DirectTabularModel`
     - ✅
     - ✅
     -
   * - :class:`~autogluon.timeseries.models.RecursiveTabularModel`
     - ✅
     - ✅
     -
   * - :class:`~autogluon.timeseries.models.DeepARModel`
     - ✅
     - ✅
     -
   * - :class:`~autogluon.timeseries.models.PatchTSTModel`
     -
     - ✅
     -
   * - :class:`~autogluon.timeseries.models.TemporalFusionTransformerModel`
     - ✅
     - ✅
     - ✅
   * - :class:`~autogluon.timeseries.models.TiDEModel`
     - ✅
     - ✅
     -
   * - :class:`~autogluon.timeseries.models.WaveNetModel`
     - ✅
     - ✅
     -
   * - :class:`~autogluon.timeseries.models.Chronos2Model`
     -
     - ✅
     - ✅
```

In addition to the above table, all models in AutoGluon can handle known covariates & static features if you set the [`covariate_regressor` hyperparameter](#hyperparameters-shared-by-all-models). Note that this may sometime lead to worse forecast accuracy, especially if the features are uninformative.

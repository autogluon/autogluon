# Forecasting Time Series - Evaluation Metrics

Picking the right evaluation metric is one of the most important choices when using an AutoML framework.

This page lists the forecast evaluation metrics available in AutoGluon, explains when different metrics should be used, and describes how to define custom metrics.

When using AutoGluon, you can specify the metric using the `eval_metric` argument to `TimeSeriesPredictor`, for example:
```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(eval_metric="MASE")
```
AutoGluon will use the provided metric to tune model hyperparameters, rank models, and construct the final ensemble for prediction.

If you are not sure which evaluation metric to pick, here are three questions that can help you make the right choice for your use case.

**Are you interested in a point forecast or a probabilistic forecast?**

If your goal is to generate an accurate **probabilistic** forecast, you should use `WQL` or `SQL` metrics.
These metrics are based on the [quantile loss](https://en.wikipedia.org/wiki/Quantile_regression) and measure the accuracy of the quantile forecasts.
By default, AutoGluon predicts quantile levels `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.
To predict a different set of quantiles, you can use `quantile_levels` argument:
```python
predictor = TimeSeriesPredictor(eval_metric="WQL", quantile_levels=[0.1, 0.5, 0.75, 0.9])
```

All remaining forecast metrics described on this page are **point** forecast metrics.
Note that if you select a point forecast metric in AutoGluon, then the forecast minimizing this metric will always be provided in the `"mean"` column of the predictions data frame.

**(Point forecast only) Do you want to estimate the mean or the median?**

To estimate the **median**, you need to use metrics such as `MAE`, `MASE` or `WAPE`.
If your goal is to predict the **mean** (expected value), you should use `MSE`, `RMSE` or `RMSSE` metrics.

**Do you care more about accurately predicting time series with large values?**

If the answer is "yes" (for example, if it's important to more accurately predict sales of popular products), you should use **scale-dependent** metrics like `WQL`, `MAE`, `RMSE`, or `WAPE`.
These metrics are also well-suited for dealing with sparse (intermittent) time series that have lots of zeros.

If the answer is "no" (you care equally about all time series in the dataset), consider **scaled** metrics like `SQL`, `MASE` and `RMSSE`. Alternatively, **percentage-based** metrics `MAPE` and `SMAPE` can also be used to equalize the scale across time series. However, these precentage-based metrics have some [well-documented limitations](https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/), so we don't recommend using them in practice.
Note that both scaled and percentage-based metrics are poorly suited for sparse (intermittent) data.



```{eval-rst}
.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :align: center
   :widths: 40 20 20 20

   * - Metric
     - Probabilistic?
     - Scale-dependent?
     - Predicts median or mean?
   * - :class:`~autogluon.timeseries.metrics.SQL` 
     - ✅
     -  
     -  
   * - :class:`~autogluon.timeseries.metrics.WQL` 
     - ✅
     - ✅
     - 
   * - :class:`~autogluon.timeseries.metrics.MAE`
     - 
     - ✅
     - median
   * - :class:`~autogluon.timeseries.metrics.MASE` 
     - 
     - 
     - median
   * - :class:`~autogluon.timeseries.metrics.WAPE`
     - 
     - ✅
     - median
   * - :class:`~autogluon.timeseries.metrics.MSE`
     - 
     - ✅
     - mean
   * - :class:`~autogluon.timeseries.metrics.RMSE`
     - 
     - ✅
     - mean
   * - :class:`~autogluon.timeseries.metrics.RMSSE`
     - 
     - 
     - mean
   * - :class:`~autogluon.timeseries.metrics.MAPE`
     - 
     - 
     - 
   * - :class:`~autogluon.timeseries.metrics.SMAPE` 
     - 
     - 
     - 
```

## Overview

## Probabilistic forecast metrics

```{eval-rst}
.. autoclass:: SQL
```

```{eval-rst}
.. autoclass:: WQL
```

## Point forecast metrics

```{eval-rst}
.. autoclass:: MAE
```

```{eval-rst}
.. autoclass:: MAPE
```

```{eval-rst}
.. autoclass:: MASE
```

```{eval-rst}
.. autoclass:: MSE
```

```{eval-rst}
.. autoclass:: RMSE
```

```{eval-rst}
.. autoclass:: RMSSE
```

```{eval-rst}
.. autoclass:: SMAPE
```

```{eval-rst}
.. autoclass:: WAPE
```




## Custom metrics
If none of the built-in metrics meet your requirements, you can train provide a custom evaluation metric to AutoGluon.

```python
from autogluon.timeseries.metrics import TimeSeriesScorer
```


Below is the full documentation
```{eval-rst}
.. autoclass:: TimeSeriesScorer
   :members: compute_metric, save_past_metrics, clear_past_metrics

```

<!-- ## Baseline models

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
.. autoclass:: ThetaModel
   :members: init
```


```{eval-rst}
.. autoclass:: NPTSModel
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
     -
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
   * - :class:`~autogluon.timeseries.models.WaveNetModel`
     - ✓
     - ✓
     - ✓
     - 
``` -->

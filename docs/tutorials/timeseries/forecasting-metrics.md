# Forecasting Time Series - Evaluation Metrics

Picking the right evaluation metric is one of the most important choices when using an AutoML framework.
This page lists the forecast evaluation metrics available in AutoGluon and explains when different metrics should be used. 

When using AutoGluon, you can specify the metric using the `eval_metric` argument to `TimeSeriesPredictor`, for example:
```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(eval_metric="MASE")
```
AutoGluon will use the provided metric to tune model hyperparameters, rank models, and construct the final ensemble for prediction.

:::{note}
AutoGluon always reports all metrics in a **higher-is-better** format.
For this purpose, some metrics are multiplied by -1.
For example, if we set `eval_metric="MASE"`, the predictor will actually report `-MASE` (i.e., MASE score multiplied by -1). This means the `test_score` will be between 0 (best possible forecast) and $-\infty$ (worst possible forecast).
:::


Currently, AutoGluon supports following evaluation metrics:

```{eval-rst}
.. automodule:: autogluon.timeseries.metrics
```

```{eval-rst}
.. currentmodule:: autogluon.timeseries.metrics
```


```{eval-rst}
.. autosummary::
   :nosignatures:

   SQL
   WQL
   MAE
   MAPE
   MASE
   MSE
   RMSE
   RMSSE
   SMAPE
   WAPE

``` 

## Which evaluation metric to choose?

If you are not sure which evaluation metric to pick, here are three questions that can help you make the right choice for your use case.

**1. Are you interested in a point forecast or a probabilistic forecast?**

If your goal is to generate an accurate **probabilistic** forecast, you should use `WQL` or `SQL` metrics.
These metrics are based on the [quantile loss](https://en.wikipedia.org/wiki/Quantile_regression) and measure the accuracy of the quantile forecasts.
By default, AutoGluon predicts quantile levels `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.
To predict a different set of quantiles, you can use `quantile_levels` argument:
```python
predictor = TimeSeriesPredictor(eval_metric="WQL", quantile_levels=[0.1, 0.5, 0.75, 0.9])
```

All remaining forecast metrics described on this page are **point** forecast metrics.
Note that if you select the `eval_metric` to a point forecast metric when creating the `TimeSeriesPredictor`, then the forecast minimizing this metric will always be provided in the `"mean"` column of the predictions data frame.

**2. Do you care more about accurately predicting time series with large values?**

If the answer is "yes" (for example, if it's important to more accurately predict sales of popular products), you should use **scale-dependent** metrics like `WQL`, `MAE`, `RMSE`, or `WAPE`.
These metrics are also well-suited for dealing with sparse (intermittent) time series that have lots of zeros.

If the answer is "no" (you care equally about all time series in the dataset), consider **scaled** metrics like `SQL`, `MASE` and `RMSSE`. Alternatively, **percentage-based** metrics `MAPE` and `SMAPE` can also be used to equalize the scale across time series. However, these precentage-based metrics have some [well-documented limitations](https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/), so we don't recommend using them in practice.
Note that both scaled and percentage-based metrics are poorly suited for sparse (intermittent) data.

**3. (Point forecast only) Do you want to estimate the mean or the median?**

To estimate the **median**, you need to use metrics such as `MAE`, `MASE` or `WAPE`.
If your goal is to predict the **mean** (expected value), you should use `MSE`, `RMSE` or `RMSSE` metrics.



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



## Point forecast metrics
We use the following notation in mathematical definitions of point forecast metrics:

- $y_{i,t}$ - observed value of time series $i$ at time $t$
- $f_{i,t}$ - predicted value of time series $i$ at time $t$
- $N$ - number of time series (number of items) in the dataset
- $T$ - length of the observed time series
- $H$ - length of the forecast horizon (`prediction_length`)


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


## Probabilistic forecast metrics
In addition to the notation listed above, we use following notation to define probabilistic forecast metrics:

- $f_{i,t}^q$ - predicted quantile $q$ of time series $i$ at time $t$
- $\rho_q(y, f) $ - quantile loss at level $q$ defined as

$$
      \rho_q(y, f) =    \begin{cases}
      2 \cdot (1 - q) \cdot (f^q_{i,t} - y_{i,t}), & \text{ if } y_{i,t} < f_{i,t}^q\\
      2 \cdot q \cdot (y_{i,t} - f^q_{i,t} ), & \text{ if } y_{i,t} \ge f_{i,t}^q\\
      \end{cases}
$$



```{eval-rst}
.. autoclass:: SQL
```

```{eval-rst}
.. autoclass:: WQL
```

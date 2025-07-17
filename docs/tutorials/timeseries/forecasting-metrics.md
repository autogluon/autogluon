---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---
(forecasting_metrics)=
# Forecasting Time Series - Evaluation Metrics


Picking the right evaluation metric is one of the most important choices when using an AutoML framework.
This page lists the forecast evaluation metrics available in AutoGluon, explains [when different metrics should be used](#which-evaluation-metric-to-choose), and describes how to [define custom evaluation metrics](#custom-forecast-metrics).

When using AutoGluon, you can specify the metric using the `eval_metric` argument to `TimeSeriesPredictor`, for example:
```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(eval_metric="MASE")
```
AutoGluon will use the provided metric to tune model hyperparameters, rank models, and construct the final ensemble for prediction.

:::{note}
AutoGluon always reports all metrics in a **higher-is-better** format.
For this purpose, some metrics are multiplied by -1.
For example, if we set `eval_metric="MASE"`, the predictor will actually report `-MASE` (i.e., MASE score multiplied by -1). This means the `test_score` will be between 0 (most accurate forecast) and $-\infty$ (least accurate forecast).
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
   RMSLE
   RMSSE
   SMAPE
   WAPE

```
Alternatively, you can [define a custom forecast evaluation metric](#custom-forecast-metrics).

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
Note that if you select the `eval_metric` to a point forecast metric when creating the `TimeSeriesPredictor`, then the forecast minimizing this metric will always be provided in the `"mean"` column of the predictions dataframe.

**2. Do you care more about accurately predicting time series with large values?**

If the answer is "yes" (for example, if it's important to more accurately predict sales of popular products), you should use **scale-dependent** metrics like `WQL`, `MAE`, `RMSE`, or `WAPE`.
These metrics are also well-suited for dealing with sparse (intermittent) time series that have lots of zeros.

If the answer is "no" (you care equally about all time series in the dataset), consider **scaled** metrics like `SQL`, `MASE` and `RMSSE`. Alternatively, **percentage-based** metrics `MAPE` and `SMAPE` can also be used to equalize the scale across time series. However, these percentage-based metrics have some [well-documented limitations](https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/), so we don't recommend using them in practice.
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
   * - :class:`~autogluon.timeseries.metrics.RMSLE`
     -
     -
     -
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
.. autoclass:: RMSLE
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
      \rho_q(y_{i,t}, f_{i,t}^q) =    \begin{cases}
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


## Custom forecast metrics

If none of the built-in metrics meet your requirements, you can provide a custom evaluation metric to AutoGluon.
To define a custom metric, you need to create a class that inherits from `TimeSeriesScorer` and implements the `compute_metric` method according to the following API specification:

```{eval-rst}
.. automethod:: TimeSeriesScorer.compute_metric
```

### Custom mean squared error metric

Here is an example of how you can define a custom mean squared error (MSE) metric using `TimeSeriesScorer`.

```{code-cell} ipython3
import sklearn.metrics
from autogluon.timeseries.metrics import TimeSeriesScorer

class MeanSquaredError(TimeSeriesScorer):
   greater_is_better_internal = False
   optimum = 0.0

   def compute_metric(self, data_future, predictions, target, **kwargs):
      return sklearn.metrics.mean_squared_error(y_true=data_future[target], y_pred=predictions["mean"])
```
The internal method `compute_metric` returns metric in lower-is-better format, so we need to set `greater_is_better_internal=False`.
This will tell AutoGluon that the metric value must be multiplied by `-1` to convert it to greater-is-better format.

:::{note}
Custom metrics must be defined in a separate Python file and imported so that they can be pickled (Python’s serialization protocol). If a custom metric is not picklable, AutoGluon may crash during fit if you enable hyperparameter tuning. In the above example, you would want to create a new python file such as `my_metrics.py` with class `MeanSquaredError` defined in it, and then use it via `from my_metrics import MeanSquaredError`.
:::

We can use the custom metric to measure accuracy of a forecast generated by the predictor.
```{code-cell} ipython3
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Create dummy dataset
data = TimeSeriesDataFrame.from_iterable_dataset(
   [
       {"start": pd.Period("2023-01-01", freq="D"), "target": list(range(15))},
       {"start": pd.Period("2023-01-01", freq="D"), "target": list(range(30, 45))},
    ]
)
prediction_length = 3
train_data, test_data = data.train_test_split(prediction_length=prediction_length)
predictor = TimeSeriesPredictor(prediction_length=prediction_length, verbosity=0).fit(train_data, hyperparameters={"Naive": {}})
predictions = predictor.predict(train_data)

mse = MeanSquaredError(prediction_length=predictor.prediction_length)
mse_score = mse(
  data=test_data,
  predictions=predictions,
  target=predictor.target,
)
print(f"{mse.name_with_sign} = {mse_score}")
```
Note that the metric value has been multiplied by `-1` because we set `greater_is_better_internal=False` when defining the metric.

When we call the metric, `TimeSeriesScorer` takes care of splitting `test_data` into past & future parts, validating that `predictions` have correct timestamps, and ensuring that the score is reported in greater-is-better format.

During the metric call, the method `compute_metric` that we implemented receives as input the following arguments:

- Test data corresponding to the forecast horizon

```{code-cell} ipython3
data_future = test_data.slice_by_timestep(-prediction_length, None)
data_future
```

- Predictions for the forecast horizon

```{code-cell} ipython3
predictions.round(2)
```

Note that both `data_future` and `predictions` cover the same time range.


### Custom quantile loss metric
The metric can be computed on any columns of the `predictions` DataFrame.
For example, here is how we can define the [mean quantile loss](https://otexts.com/fpp3/distaccuracy.html#quantile-scores) metric that measures the accuracy of the quantile forecast.

```{code-cell} ipython3
class MeanQuantileLoss(TimeSeriesScorer):
   needs_quantile = True
   greater_is_better_internal = False
   optimum = 0.0

   def compute_metric(self, data_future, predictions, target, **kwargs):
      quantile_columns = [col for col in predictions if col != "mean"]
      total_quantile_loss = 0.0
      for q in quantile_columns:
        total_quantile_loss += sklearn.metrics.mean_pinball_loss(y_true=data_future[target], y_pred=predictions[q], alpha=float(q))
      return total_quantile_loss / len(quantile_columns)
```
Here we set `needs_quantile=True` to tell AutoGluon that this metric is evaluated on the quantile forecasts.
In this case, models such as {py:class}`~autogluon.timeseries.models.DirectTabularModel` will train a regression model from `autogluon.tabular` with `problem_type="quantile"` under the hood.
If `needs_quantile=False`, these models will use `problem_type="regression"` instead.

### Custom mean absolute scaled error metric
Finally, here is how we can define the [mean absolute scaled error (MASE) metric](https://otexts.com/fpp3/accuracy.html#scaled-errors).
Unlike previously discussed metrics, MASE is computed using both **past** and **future** time series values.
The past values are used to compute the scale by which we normalize the error during the forecast horizon.

```{code-cell} ipython3
class MeanAbsoluteScaledError(TimeSeriesScorer):
  greater_is_better_internal = False
  optimum = 0.0
  optimized_by_median = True
  equivalent_tabular_regression_metric = "mean_absolute_error"

  def save_past_metrics(
      self, data_past: TimeSeriesDataFrame, target: str = "target", seasonal_period: int = 1, **kwargs
  ) -> None:
      seasonal_diffs = data_past[target].groupby(level="item_id").diff(seasonal_period).abs()
      self._abs_seasonal_error_per_item = seasonal_diffs.groupby(level="item_id").mean().fillna(1.0)

  def clear_past_metrics(self):
      self._abs_seasonal_error_per_item = None

  def compute_metric(
      self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
  ) -> float:
      mae_per_item = (data_future[target] - predictions["mean"]).abs().groupby(level="item_id").mean()
      return (mae_per_item / self._abs_seasonal_error_per_item).mean()
```
We compute the metrics on past data using `save_past_metrics` method.
Doing this in a separate method allows AutoGluon to avoid redundant computations when fitting the weighted ensemble, which requires thousands of metric evaluations.

Because we set `optimized_by_median=True`, AutoGluon will automatically paste the median forecast into the `"mean"` column of predictions.
This is done for consistency: if `TimeSeriesPredictor` is trained with a point forecast metric, the optimal point forecast will always be stored in the `"mean"` column.
Finally, the `equivalent_tabular_regression_metric` is used by forecasting models that fit tabular regression models from `autogluon.tabular` under the hood.


### Using custom metrics in TimeSeriesPredictor
Now that we have created several custom metrics, let’s use them for training and evaluating models.

```{code-cell} ipython3
predictor = TimeSeriesPredictor(eval_metric=MeanQuantileLoss()).fit(train_data, hyperparameters={"Naive": {}, "SeasonalNaive": {}, "Theta": {}})
```

We can also evaluate a trained predictor using these custom metrics
```{code-cell} ipython3
predictor.evaluate(test_data, metrics=[MeanAbsoluteScaledError(), MeanQuantileLoss(), MeanSquaredError()])
```

That’s all it takes to create and use custom forecasting metrics in AutoGluon!

You can have a look at the AutoGluon source code for example implementations of [point](https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/metrics/point.py) and [quantile](https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/metrics/quantile.py) forecasting metrics.

If you create a custom metric, consider [submitting a PR](https://github.com/autogluon/autogluon/pulls) so that we can officially add it to AutoGluon.

For more tutorials, refer to [Forecasting Time Series - Quick Start](forecasting-quick-start.ipynb) and [Forecasting Time Series - In Depth](forecasting-indepth.ipynb).


## Customizing the training loss for individual models
While `eval_metric` is used for model selection and weighted ensemble construction, it usually has no effect on the training loss of the individual forecasting models.

In some models such as `AutoETS` or `AutoARIMA`, the training loss is fixed and cannot be changed.
In contrast, for GluonTS-based deep learning models the training loss can be changed by modifying the `distr_output` [hyperparameter](forecasting-model-zoo.md#deep-learning-models).
By default, most GluonTS models set the `distr_output` to the heavy‑tailed `StudentTOutput` distribution for increased robustness to outliers.

You can replace the default `StudentTOutput` with any built‑in `Output` from the [`gluonts.torch.distributions`](https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.distributions.html) module.
For example, here we train two versions of PatchTST with different outputs and losses:
- `NormalOutput` - the model outputs parameters of a Gaussian distribution and trains with the negative log-likelihood loss.
- `QuantileOutput` - the model outputs a quantile forecast and trains with the quantile loss.

```python
from autogluon.timeseries import TimeSeriesPredictor
from gluonts.torch.distributions import NormalOutput, QuantileOutput

predictor = TimeSeriesPredictor(...)
predictor.fit(
    train_data,
    hyperparameters={
        "PatchTST": [
            {"distr_output": NormalOutput()},
            {"distr_output": QuantileOutput(quantiles=predictor.quantile_levels)},
        ]
    }
)
```

You can define a custom loss function for the GluonTS models by defining a subclass of [`gluonts.torch.distributions.Output`](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/distributions/output.py)
and providing it as a `distr_output` to the model.

```python
from gluonts.torch.distributions import Output

class MyCustomOutput(Output):
    # implement methods of gluonts.torch.distributions.Output
    ...

predictor.fit(train_data, hyperparameters={"PatchTST": {"distr_output": MyCustomOutput()}})
```
You can find examples of `Output` implementations in the GluonTS code base (e.g., [`QuantileOutput`](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/distributions/quantile_output.py) or [`NormalOutput`](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/distributions/distribution_output.py)).

(forecasting_ensembles)=
# Forecasting Time Series - Ensemble Models

:::{note}
This documentation is intended for advanced users and may not be comprehensive.

For a stable public API, refer to the [documentation for `TimeSeriesPredictor`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html).
:::

This page contains the list of time series ensemble models available in AutoGluon.
These models combine predictions from multiple base forecasting models to improve accuracy.

The available hyperparameters for each model are listed under **Parameters**.

The model names in the `ensemble_hyperparameters` dictionary don't have to include the `"Ensemble"` suffix
(e.g., both `"SimpleAverage"` and `"SimpleAverageEnsemble"` correspond to {class}`~autogluon.timeseries.models.ensemble.SimpleAverageEnsemble`).


## How ensembling works

Ensemble models combine predictions from multiple base forecasting models to produce a final forecast. The ensemble is trained on held-out validation data (backtest windows) to learn how to best combine the base model predictions.

By default, AutoGluon uses a single {class}`~autogluon.timeseries.models.ensemble.GreedyEnsemble` that learns optimal weights for each base model. You can configure which ensemble models to use via the `ensemble_hyperparameters` argument in [`TimeSeriesPredictor.fit()`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html).


## Multi-layer stacking

Multi-layer stacking extends the basic ensembling approach by training ensembles in multiple stages. Each layer of ensembles is trained on different backtest windows, and uses predictions from the previous layer as its inputs.

For example, with `num_val_windows=(3, 2)` and two ensemble layers:

```
Time series:  [...history...][window 1][window 2][window 3][window 4][window 5]
                             └───────── Layer 2 ──────────┘└───── Layer 3 ─────┘
```

1. Base models generate predictions for all 5 backtest windows
2. Layer 2 ensembles are trained on windows 1-3, learning to combine base model predictions
3. Layer 3 ensembles are trained on windows 4-5, using Layer 1 ensemble predictions as inputs
4. Final validation scores are computed on windows 4-5

This approach allows later ensemble layers to correct errors made by earlier layers, often improving forecast accuracy.

To enable multi-layer stacking, pass a list of dicts to `ensemble_hyperparameters` and a matching tuple to `num_val_windows`:

```{code-cell} ipython3
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

full_data = TimeSeriesDataFrame('https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_small/train.csv')
prediction_length = 48
train_data, test_data = full_data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MASE")
predictor.fit(
    train_data,
    # Base models
    hyperparameters={"SeasonalNaive": {}, "Theta": {}, "DirectTabular": {}, "ETS": {}, "RecursiveTabular": {}},
    ensemble_hyperparameters=[
        # Layer 2
        {"WeightedEnsemble": {}, "Median": {}, "LinearStacker": {"weights_per": "mt"}},
        # Layer 3
        {"WeightedEnsemble": {}},  # Layer 2
    ],
    num_val_windows=(3, 2),  # 3 windows to fit L2 models, 2 windows to fit L3 models
    refit_every_n_windows="auto",
)
predictor.leaderboard(test_data)
```

After training the predictor, you can access the validation predictions & targets used to train the ensembles using
[`backtest_targets()`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.backtest_targets.html)
and [`backtest_predictions()`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.backtest_predictions.html) methods.


## Overview

```{eval-rst}
.. automodule:: autogluon.timeseries.models.ensemble
```

```{eval-rst}
.. currentmodule:: autogluon.timeseries.models.ensemble
```

```{eval-rst}
.. autosummary::
   :nosignatures:

    GreedyEnsemble
    LinearStackerEnsemble
    MedianEnsemble
    PerItemGreedyEnsemble
    PerQuantileTabularEnsemble
    SimpleAverageEnsemble
    TabularEnsemble

```

## Simple averages

Simple ensemble models that combine predictions using mean or median aggregation.


```{eval-rst}
.. autoclass:: SimpleAverageEnsemble
   :members: init
```


```{eval-rst}
.. autoclass:: MedianEnsemble
   :members: init
```


## Linear ensembles

Linear ensemble models that combine predictions using weighted averages or linear stacking.


```{eval-rst}
.. autoclass:: GreedyEnsemble
   :members: init
```


```{eval-rst}
.. autoclass:: PerItemGreedyEnsemble
   :members: init
```


```{eval-rst}
.. autoclass:: LinearStackerEnsemble
   :members: init
```


## Nonlinear ensembles

Nonlinear ensemble models that use tabular models to combine predictions from base forecasters.


```{eval-rst}
.. autoclass:: TabularEnsemble
   :members: init
```


```{eval-rst}
.. autoclass:: PerQuantileTabularEnsemble
   :members: init
```


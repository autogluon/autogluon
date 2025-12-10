(forecasting_ensembles)=
# Forecasting Time Series - Ensemble Models

:::{note}
This documentation is intended for advanced users and may not be comprehensive.

For a stable public API, refer to the [documentation for `TimeSeriesPredictor`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html).
:::

This page contains the list of time series ensemble models available in AutoGluon.
These models combine predictions from multiple base forecasting models to improve accuracy.

The available hyperparameters for each model are listed under **Parameters**.

This list is useful if you want to override the default hyperparameters ([Manually configuring models](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html#manually-configuring-models))
or define custom hyperparameter search spaces ([Hyperparameter tuning](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html#hyperparameter-tuning)), as described in the In-depth Tutorial.

The model names in the `hyperparameters` dictionary don't have to include the `"Ensemble"` suffix
(e.g., both `"SimpleAverage"` and `"SimpleAverageEnsemble"` correspond to {class}`~autogluon.timeseries.models.ensemble.SimpleAverageEnsemble`).



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


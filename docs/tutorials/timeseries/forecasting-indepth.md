# Forecasting Time Series - In Depth
:label:`sec_forecastingadvanced`

This tutorial provides an in-depth overview of the time series forecasting capabilities in AutoGluon.

- What is probabilistic time series forecasting?
- What forecasting models are available in AutoGluon?
- How does AutoGluon evaluate performance of time series models?
- What functionality does `TimeSeriesPredictor` offer?
    - Choosing the presets
    - Manually specifying what models should be trained
    - Hyperparameter tuning
    - Forecasting irregularly-sampled time series

This tutorial assumes that you are familiar with the contents of the basic tutorial.

## What is probabilistic time series forecasting?
A time series is a sequence of measurement made at regular intervals.
The main objective of time series forecasting is to predict the future values of a time series given the past observations.

A typical example of this task is demand forecasting.
We can represent the number of daily purchases of a certain product as a time series.
The goal in this case can be to predict the demand for each of the next 14 days given the historical purchase data.

FIGURE:

More abstractly, we can represent the past observations of the time series as a vector $(y_1, y_2, ..., y_T)$ of arbitrary length $T$.
The goal of forecasting is to predict the future values of the time series $(y_{T+1}, y_{T+2}, ..., y_{T+H})$, where $H$ is the length of the forecast horizon.


<!-- In language of AutoGluon `prediction_length` defines the length of the forecast horizon.


Predicting multiple time series --- predicting demand for different product categories.

Univariate - we model each time series independently. -->


### Point forecast vs. probabilistic forecast
One way to approach forecasting is to train a model that directly predicts the value $\hat{y}_{T+t}$ of the time series at each future time step $t = 1, ..., H$ in the forecast horizon.
This is similar to multivariate regression in traditional machine learning — the past observations $(y_1, ..., y_T)$ serve as a feature vector and the future time series values $(y_{T+1}, ..., y_{T+H})$ are the unknown targets that the forecasting model predicts as $(\hat{y}_{T+1}, ..., \hat{y}_{T+H})$.

Such forecast that consists of a single predicted value $\hat{y}_{T+t}$ for each of the future time steps is known as a _point forecast_.
The name comes from the fact that the prediction for each time step can be viewed as a single point on the $y$ axis.

In many practical applications, however, a point forecast is not enough — what we really care about is the _range of possible outcomes_, not just a single prediction for each future time step.
Back to our demand forecasting example, to plan the inventory we rather need predictions such as "With probability 90%, the number of purchases 14 days into the future will lie between 100 and 150 units".
To make such predictions, we need to model the _probability distribution_ of the future time series values given the past.

All forecasting algorithms in AutoGluon model the distribution of future time series values given the past.

Let's have a look at the distribution of the value $y_{T+1}$

The expected value


A `TimeSeriesPredictor` in AutoGluon produces both point and quantile forecasts.

In AutoGluon, we can specify what quantiles we care about using the
`quantile_levels` argument to the `TimeSeriesPredictor`.

Median corresponds to the 0.5 quantile

For example, if we are looking for the 90% interval, we can specify
`quantile_levels=[0.05, 0.5, 0.95]`

histogram



Important caveat


conditional expectation of the future given the past

## What forecasting models are available in AutoGluon?
Forecasting models in AutoGluon can be divided into three broad categories: local, global, and ensemble models.

**Local models** are simple statistical models that are specifically designed to capture patterns such as trend or seasonality.
Despite their simplicity, these models often produce reasonable forecasts and serve as a strong baseline.
Available local models include:

- `ETS`
- `ARIMA`

If the dataset consists of multiple time series, we fit a separate local model to each time series — hence the name "local".
This means, if we want to make a forecast for a new time series that wasn't part of the training set, all local models will be fit from scratch for the new time series.

**Global models** are deep-learning-based algorithms that learn a single model from the entire training set consisting of multiple time series.
AutoGluon relies on [GluonTS](https://ts.gluon.ai/stable/) for the implementation of global models in PyTorch and MXNet.
Available global models include:

- `DeepAR`
- `SimpleFeedForward`
- `Transformer`
- `MQRNN`
- `MQCNN`
- `TemporalFusionTransformer`

Finally, an **ensemble** model works by combining predictions of all other models.
By default, `TimeSeriesPredictor` always fits a `WeightedEnsemble` on top of other models.
This can be disabled by setting `enable_ensemble=False` when creating the predictor.

For a list of tunable hyperparameters for each model, their default values, and other details see [Model zoo](#TODO).

## How does AutoGluon evaluate performance of time series models?
AutoGluon evaluates the performance of different models by measuring how well their forecasts align with the actually observed time series.
For example, let's look at how the `test_score` is computed when we call the `leaderboard` method of a trained predictor
```python
# Fit a predictor to training data
predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric=eval_metric)
predictor.fit(train_data=train_data)
# Evaluate a predictor on test data
predictor.leaderboard(data=test_data)
```
For each time series in the test dataset and for each trained model, the predictor does the following:

1. Hide the last `prediction_length` values of the time series.
2. Generate a forecast for the hidden part of the time series using the model.
3. Quantify how well the model's forecast matches the actually observed (hidden) values of the time series using the `eval_metric`.

Different metrics capture different properties of the forecast, and therefore depend on the application that the user has in mind.
For example, weighted quantile loss (`"mean_wQuantileLoss"`) measures how well-calibrated the quantile forecast is; mean absolute scale error (`"MASE"`) compares the mean forecast to a naive baseline.
For more details about the available metrics, see [Metrics overview](#TODO).

Note that AutoGluon always reports all metrics in a **higher-is-better** format.
For this purpose, some metrics are multiplied by -1.
For example, if we set `eval_metric="MASE"`, the predictor will actually report `-MASE` (i.e., MASE score multiplied by -1). This means the `test_score` will be between 0 (best possible forecast) and $-\infty$ (worst possible forecast).


### How are validation scores computed?
`predictor.evaluate(data)`

Internally, the predictor
1. "hides" the last `prediction_length` steps in each time series in data
2. generates a forecast for the last
3. computes how well the forecast matches the observed data

Important point - when a `TimeSeriesDataFrame` is used for validation, only the last `prediction_length` timesteps are used for computing the validation score


## What functionality does `TimeSeriesPredictor` offer?

### Choosing the presets
The simplest way to is using the `presets` argument of the `fit` method

```python
predictor = TimeSeriesPredictor()
predictor.fit(train_data=train_data, presets="medium_quality")
```

### Manually specifying what models should be trained

```python
predictor = TimeSeriesPredictor()
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "ETS": {
            "seasonal": "add",
            "seasonal_period": 7,
        },
        "ARIMA": {
            "order": (1, 2, 1),
        }
        "DeepAR": {},  # train with default hyperparameters
    }
    time_limit=60*60,
)
```

### Hyperparameter tuning

```python
import autogluon.core as ag

predictor = TimeSeriesPredictor()

predictor.fit(
    train_data=train_data,
    hyperparameters={
        "DeepAR": {
            "num_layers": ag.Categorical(2, 3, 4),
            "num_cells": ag.Int(10, 30),
        }
    }
    hyperparameter_tune_kwargs="random",
    time_limit=60*60,
)
```

### Forecasting irregularly-sampled time series

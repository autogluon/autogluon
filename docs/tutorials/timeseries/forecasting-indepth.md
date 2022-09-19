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

<!-- More abstractly, we can represent the past observations of the time series as a vector $(y_1, y_2, ..., y_T)$ of arbitrary length $T$.
The goal of forecasting is to predict the future values of the time series $(y_{T+1}, y_{T+2}, ..., y_{T+H})$, where $H$ is the length of the forecast horizon. -->

In AutoGluon, a `TimeSeriesPredictor` trains forecasting models


use a `TimeSeriesDataFrame` to store a dataset consisting a multiple related time series.

```
data = TimeSeriesDataFrame(...)
predictor = TimeSeriesPredictor(prediction_length=...)

predictor.fit(train_data=data)

forecast = predictor.predict(data)
```

### How does training work?

After we finished training, AutoGluon outputs the list of available models


When we call `predictor.predict`, it makes prediction using the model that had the best validation loss

Can also override to make predictions


A `TimeSeriesPredictor` in AutoGluon is trained to predict multiple related time series simultaneously.


(for example, demand for different item categories)


AutoGluon provides two types of forecasts:

- mean forecast
- quantile forecast represents the range of possible outcomes

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

To reason about the range of possible outcomes, we need to model the _probability distribution_ of the future time series values given the past.
For example, a distribution of the value $y_{T+1}$

There exist multiple ways to summarize this distribution.
A point forecast

To reason about the range of possible outcomes, we can look at the quantiles of the distribution.



### What predictions does it make?


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
# TODO: example leaderboard output
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
When we fit the predictor with `predictor.fit(train_data=train_data)`, under the hood AutoGluon splits the original dataset `train_data` into train and validation parts.




You can provide a custom validation dataset to the predictor using the `tuning_data` argument: `predictor.fit(..., tuning_data=tuning_data)`.


Important point - when a `TimeSeriesDataFrame` is used for validation, only the last `prediction_length` timesteps are used for computing the validation score


## What functionality does `TimeSeriesPredictor` offer?
AutoGluon offers multiple ways to configure the behavior of a `TimeSeriesPredictor` that are suitable for both beginners and expert users.

### Basic configuration with `presets` and `time_limit`
We can fit `TimeSeriesPredictor` with different pre-defined configurations using the `presets` argument of the `fit` method.

```python
predictor = TimeSeriesPredictor()
predictor.fit(train_data=train_data, presets="medium_quality")
```

Higher quality presets, in general, result in better forecasts but take longer to train.
The following presets are available:

**TODO: These will be significantly changed by 0.6.0**

- `"low_quality"`: quickly train a few toy models. This setting should only be used as a sanity check.
- `"medium_quality"`: train several selected models (`"ETS"`, `"ARIMA"`, `"DeepAR"`, `"SimpleFeedForward"`) without hyperparameter optimization. A good baseline setting.
- `"high_quality"`: same as `"medium_quality"`, but with an extended model zoo (+ `"MQRNN"`, `"Transformer"`, `"TemporalFusionTransformer"`).
- `"best_quality"`: Train all available models with hyperparameter optimization.

Another way to control the training time is using the `time_limit` argument.

```python
predictor.fit(
    train_data=train_data,
    time_limit=60 * 60,  # total training time in seconds
)
```

If no `time_limit` is provided, the predictor will train until all models have been fit.


### Manually specifying what models should be trained
Advanced users can override the presets and manually specify what models should be trained by the predictor using the `hyperparameters` argument.


```python
predictor = TimeSeriesPredictor()

predictor.fit(
    train_data=train_data,
    hyperparameters={
        "DeepAR": {},
        "ETS": {
            "seasonality": "add",
            "seasonal_period": 7,
        }
    }
)
```
The code above will only train two models:

- `DeepAR` (with default hyperparameters)
- `ETS` (with given `seasonality` and `seasonal_period`, remaining parameters set to their defaults).

For the full list of available models and the respective hyperparameters, see [Model zoo](#TODO).

### Hyperparameter tuning
Advanced users can define search spaces for model hyperparameters and let AutoGluon automatically determine the best configuration for the model.

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
    },
)
```
This code will train multiple versions of the `DeepAR` model with different hyperparameter configurations.
AutGluon will automatically select the best model configuration that achieves the highest validation score.


### Forecasting irregularly-sampled time series
By default, `TimeSeriesPredictor` expects the time series data to be regularly sampled (e.g., measurements done every day).
However, in some applications, like finance, data often comes with irregular measurements (e.g., no stock price is available for weekends or holidays).

To train on such irregularly-sampled time series, we can set the `ignore_time_index` flag in the predictor.
```python
predictor = TimeSeriesPredictor(..., ignore_time_index=True)
predictor.fit(train_data=train_data)
```
In this case, the predictor will completely ignore the timestamps in `train_data`, and the predictions made by the model will have a dummy `timestamp` index with frequency equal to 1 second.
Also, the seasonality will be disabled for models like as `ETS` and `ARIMA`.

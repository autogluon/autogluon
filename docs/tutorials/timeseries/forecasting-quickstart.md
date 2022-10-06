# Forecasting Time Series - Quick Start
:label:`sec_forecasting_quickstart`

Via a simple `fit()` call, AutoGluon can train and tune

- simple forecasting models (e.g., ARIMA, ETS),
- powerful deep learning models (e.g., DeepAR, Transformer, MQ-CNN),
- an ensemble that combines prediction of other models

to produce multi-step ahead _probabilistic_ forecasts for univariate time series data.

This tutorial demonstrates how to quickly start using AutoGluon to forecast [the number of COVID-19 cases](https://www.kaggle.com/c/covid19-global-forecasting-week-4) in different countries given historical data.
For a short summary of how to train models and make forecasts in a few lines of code with `autogluon.timeseries`, scroll to the [bottom of this page](#summary).

---
**NOTE**

`autogluon.timeseries` depends on Apache MXNet. Please install MXNet by running

```shell
python -m pip install mxnet~=1.9
```

If you want to use a GPU, install the version of MXNet that matches your CUDA version. See the
MXNet [documentation](https://mxnet.apache.org/versions/1.9.1/get_started?) for more info.

---
## Loading time series data as a `TimeSeriesDataFrame`

First, we make several imports
```{.python .input}
import pandas as pd
from matplotlib import pyplot as plt

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```
To use `autogluon.timeseries`, we will only need the following two classes:

- `TimeSeriesDataFrame` stores a dataset consisting of multiple time series.
- `TimeSeriesPredictor` takes care of fitting, tuning and selecting the best forecasting models.


In this tutorial we work with COVID case data as of April 2020.
Our goal is to forecast the cumulative number of confirmed COVID cases for each country in the dataset given the past observations.

We load the dataset from an [AWS S3 bucket](https://aws.amazon.com/s3/) as a `pandas.DataFrame`
```{.python .input}
df = pd.read_csv(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv",
    parse_dates=["Date"],  # make sure that pandas parses the dates
)
df.head()
```
Each row of the data frame contains a single observation (timestep) of a single time series represented by

- unique ID of the time series — in our case, name of the country (`"name"`)
- timestamp of the observation (`"Date"`)
- value of the time series (`"ConfirmedCases"`)

The raw dataset should always follow this format (3 columns: unique ID, timestamp, value), but the names of these columns can be arbitrary.
It is important, however, that we provide the names of the columns when constructing a `TimeSeriesDataFrame` that is used by AutoGluon
```{.python .input}
ts_dataframe = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="name",  # column that contains unique ID of each time series
    timestamp_column="Date",  # column that contains timestamps of each observation
)
ts_dataframe
```
AutoGluon will raise an exception if the data doesn't match the expected format.

We refer to each individual time series stored in a `TimeSeriesDataFrame` as an _item_.
In our case, each item corresponds to a country.
As another example, items might correspond to different products in demand forecasting.
This setting is also sometimes referred to as a "panel" of time series.
Note that this is *not* the same as multivariate forecasting — AutoGluon generates forecasts for each time series individually, without modeling interactions between different items (time series).


`TimeSeriesDataFrame` inherits from [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), so all attributes and methods of `pandas.DataFrame` are also available in a `TimeSeriesDataFrame`.

Note how `TimeSeriesDataFrame` organizes the data with a `pandas.MultiIndex`:

-  the first _level_ of the index corresponds to the item ID (here, country name);
-  the second level contains the timestamp when each observation was made.

We can use the `loc` accessor, as in pandas, to access individual country data.
```{.python .input}
ts_dataframe.loc['Afghanistan_'].head()
```
We can also plot the time series for some countries in the dataset
```{.python .input}
plt.figure(figsize=(20, 3))
for country in ["United Kingdom_", "Germany_"]:
    plt.plot(ts_dataframe.loc[country], label=country)
plt.legend()
```

## Forecasting problem formulation
Models in `autogluon.timeseries` forecast the value of each time series _multiple steps_ into the future.
We choose the length of the prediction interval (also known as forecast horizon) depending on our task.
For example, our COVID cases dataset contains daily data, so we can set `prediction_length = 7` to train models that make daily forecasts up to 7 days into the future.

Moreover, models in `autogluon.timeseries` provide a _probabilistic_ forecast.
That is, in addition to predicting the _mean_ (expected value) of the time series in the future, they also provide the _quantiles_ of the forecast distribution.

We will split our dataset into a training set (used to train & tune models) and a test set (used to evaluate the final performance).
In forecasting, this is usually done by hiding the last `prediction_length` steps of each time series during training, and only using these last steps to evaluate the forecast quality (also known as an "out of time" validation).
We perform this split using the `slice_by_timestep` method of `TimeSeriesDataFrame`.
```{.python .input}
prediction_length = 7

test_data = ts_dataframe.copy()  # the full data set

# last prediction_length time steps of each time series are excluded, akin to `x[:-7]`
train_data = ts_dataframe.slice_by_timestep(None, -prediction_length)
```
Below, we plot the training and test parts of the time series for a single country, and mark the test forecast horizon.
We will compute the test scores by measuring how well the forecast generated by a model matches the actually observed values in the forecast horizon.
```{.python .input}
plt.figure(figsize=(20, 3))
plt.plot(test_data.loc["Germany_"], label="Test")
plt.plot(train_data.loc["Germany_"], label="Train")

test_range = (
    train_data.loc["Germany_"].index.max(),
    test_data.loc["Germany_"].index.max(),
)

plt.fill_betweenx(
    y=(0, test_data.loc["Germany_"]["ConfirmedCases"].max()),
    x1=test_range[0],
    x2=test_range[1],
    alpha=0.1,
    label="Test forecast horizon",
)

plt.legend()
```


## Training time series models with `TimeSeriesPredictor.fit`

Below we instantiate a `TimeSeriesPredictor` object and instruct AutoGluon to fit models that can forecast up to
7 time-points into the future (`prediction_length`) and save them in the folder `./autogluon-covidforecast`.
We also specify that AutoGluon should rank models according to mean absolute percentage error (MAPE) and that data is stored in the column `"ConfirmedCases"` of the `TimeSeriesDataFrame`.

```{.python .input}
predictor = TimeSeriesPredictor(
    path="autogluon-covidforecast",
    target="ConfirmedCases",
    prediction_length=prediction_length,
    eval_metric="MAPE",
)

predictor.fit(
    train_data=train_data,
    presets="local_models",
)
```
Here we used the `"local_models"` presets to quickly obtain the results.
In a short amount of time AutoGluon fits three statistical models (ARIMA, ETS, Theta), and a weighted ensemble on top of these models.

In realistic scenarios, we can set `presets` to be one of `"medium_quality"` or `"best_quality"`.
These presets additionally include powerful deep learning models (such as DeepAR, Temporal Fusion Transformer).
Higher quality presets will usually produce more accurate forecasts but take longer to train and may produce less efficient models.

Note that inside `fit()` the last `prediction_length` steps of each time series in `train_data` were automatically used as a tuning (validation) set.
This validation set is used internally by the predictor to rank models and fit the ensemble.

## Evaluating the performance of different models

We can view the test performance of each model AutoGluon has trained via the `leaderboard()` method.
We provide the test data set to the leaderboard function to see how well our fitted models are doing on the held out test data.
The leaderboard also includes the validation scores computed on the internal validation dataset.

In AutoGluon leaderboards, higher scores always correspond to better predictive performance.
Therefore our MAPE scores are multiplied by `-1`, such that higher "negative MAPE"s correspond to better models.


```{.python .input}
predictor.leaderboard(test_data, silent=True)
```

## Making forecasts with `TimeSeriesPredictor.predict`

We can now use the fitted `TimeSeriesPredictor` to make forecasts.
By default, AutoGluon will make forecasts using the model that had the best validation score (as shown in the leaderboard).
Let's use the predictor to generate forecasts starting from the end of the time series in `train_data`

```{.python .input}
predictions = predictor.predict(train_data)
predictions.head()
```
Predictions are also stored as a `TimeSeriesDataFrame`. However, now the columns contain the mean and quantile predictions of each model.
The quantile forecasts give us an idea about the range of possible outcomes.
For example, if the `"0.1"` quantile is equal to `x`, it means that the model predicts a 10% chance that the target value will be below `x`.

We will now visualize the forecast and the actually observed number of COVID cases during the prediction interval for a single country.
We plot the mean forecast, as well as the 10% and 90% quantiles to show the range of potential outcomes.
```{.python .input}
plt.figure(figsize=(20, 3))

y_past = train_data.loc["Germany_"]["ConfirmedCases"]
y_pred = predictions.loc["Germany_"]
y_true = test_data.loc["Germany_"]["ConfirmedCases"][-prediction_length:]

# prepend the last value of true range to predicted range for plotting continuity
y_pred.loc[y_past.index[-1]] = [y_past[-1]] * 10
y_pred = y_pred.sort_index()

plt.plot(y_past[-30:], label="Training data")
plt.plot(y_pred["mean"], label="Mean forecast")
plt.plot(y_true, label="Observed")

plt.fill_between(
    y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1
)
plt.title("COVID case forecasts in Germany, compared to actual observations")
_ = plt.legend()
```

## Summary
We used `autogluon.timeseries` to make probabilistic multi-step forecasts of COVID cases data.
Here is a short summary of the main steps for applying AutoGluon to make forecasts for the entire dataset using a few lines of code.
```python
import pandas as pd

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Load the data into a TimeSeriesDataFrame
df = pd.read_csv(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv",
    parse_dates=["Date"],
)
ts_dataframe = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="name",  # name of the column with unique ID of each time series
    timestamp_column="Date",  # name of the column with timestamps of observations
)

# Create & fit the predictor
predictor = TimeSeriesPredictor(
    path="autogluon-covidforecast",  # models will be saved in this folder
    target="ConfirmedCases",  # name of the column with time series values
    prediction_length=7,  # number of steps into the future to predict
    eval_metric="MAPE",  # other options: "MASE", "sMAPE", "mean_wQuantileLoss"
)
predictor.fit(
    train_data=ts_dataframe,
    presets="local_models",  # other options: "medium_quality", "best_quality"
)

# Generate the forecasts
predictions = predictor.predict(ts_dataframe)
```
Check out :ref:`sec_forecasting_indepth` to learn about the advanced capabilities of AutoGluon for time series forecasting.

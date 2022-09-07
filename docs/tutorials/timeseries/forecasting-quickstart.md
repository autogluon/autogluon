# Forecasting Time Series - Quick Start
:label:`sec_forecastingquick`

Via a simple `fit()` call, AutoGluon can train and tune

- simple forecasting models (e.g., ARIMA, ETS),
- powerful deep learning models (e.g., DeepAR, Transformer, MQ-CNN),
- an ensemble that combines prediction of other models

to produce multi-step ahead _probabilistic_ forecasts for univariate time series data.

This tutorial demonstrates how to quickly start using AutoGluon to produce forecasts of COVID-19 cases in different countries given [historical data](https://www.kaggle.com/c/covid19-global-forecasting-week-4).

---
**NOTE**

`autogluon.timeseries` depends on Apache MXNet. Please install MXNet by running

```shell
python -m pip install mxnet~=1.9
```

If you want to use a GPU, install the version of MXNet that matches your CUDA version. See the
MXNet [documentation](https://mxnet.apache.org/versions/1.9.1/get_started?) for more info.

---
## Loading time series data

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
Our goal is to forecast the cumulative number of confirmed COVID cases for each country in the dataset.

We load the dataset from an [AWS S3 bucket](https://aws.amazon.com/s3/) as a `pandas.DataFrame`
```{.python .input}
df = pd.read_csv(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv",
    parse_dates=["Date"],  # make sure that pandas parses the dates
)
df
```
Each row of the data frame contains
- unique ID of each time series — in our case, name of the country (`"name"`)
- timestamp of the observation (`"Date"`)
- value of the time series (`"ConfirmedCases"`)

We now convert the original data to a `TimeSeriesDataFrame` used by AutoGluon
```{.python .input}
ts_dataframe = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="name",  # column that contains unique ID of each time series
    timestamp_column="Date",  # column that contains timestamps of each observation
)
ts_dataframe
```
We refer to each individual time series stored in a `TimeSeriesDataFrame` as an _item_.
In our case, each item corresponds to a country.
As another example, items might correspond to different products in demand forecasting.
This setting is also sometimes referred to as a "panel" of time series.

`TimeSeriesDataFrame` inherits from [`pandas.DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), so all attributes and methods of `pandas.DataFrame` are also available in a `TimeSeriesDataFrame`.

Note how `TimeSeriesDataFrame` organizes the data with a `pandas.MultiIndex` where the first _level_ of the index
corresponds to the item (here, country) and the second level contains the timestamp of the observation.

We can use the `loc` accessor, as in pandas, to access individual country data.
```{.python .input}
train_data.loc['Afghanistan_'].head()
```

```{.python .input}
plt.figure(figsize=(20, 3))
for country in ["United Kingdom_", "Germany_"]:
    plt.plot(ts_dataframe.loc[country], label=country)
plt.legend()
```


The primary use case of `autogluon.timeseries` is time series forecasting. In our example, our goal is to train models on COVID case data
that can forecast the future trajectory of cases given the past, for each country in the data set.
By default, `autogluon.timeseries` supports multi-step ahead _probabilistic_ forecasting. That is, multiple time steps in the future
can be forecasted, given that models are trained with the prerequisite number of steps (also known as the _forecast horizon_).
Moreover, when trained models are used to predict the future, the library will provide both `"mean"`
forecasts--expected values of the time series in the future, as well as _quantiles_ of the forecast distribution.

In order to train our forecasting models, we first split the data into training and test data sets.
In forecasting, this is often done via excluding the last `prediction_length` many steps of the data set during training, and
only use these steps to compute validation scores (also known as an "out of time" validation sample).
We carry out this split via the `slice_by_timestep` method provided by `TimeSeriesDataFrame` which takes python `slice` objects.


```{.python .input}
prediction_length = 5

test_data = ts_dataframe.copy()  # the full data set

# the data set with the last prediction_length time steps excluded, i.e., akin to `a[:-5]`
train_data = ts_dataframe.slice_items_by_index(start_index=None, end_index=-prediction_length)
```

Below, for a single country we plot the training and test data sets showing how they overlap and explicitly mark the forecast horizon of the
test data set. The test scores will be computed on forecasts provided for this range.


```{.python .input}
plt.figure(figsize=(20, 3))
plt.plot(test_data.loc["Germany_"], label="test")
plt.plot(train_data.loc["Germany_"], label="train")

test_range = (
    test_data.loc["Germany_"].index.max(),
    train_data.loc["Germany_"].index.max(),
)

plt.fill_betweenx(
    y=(0, test_data.loc["Germany_"]["ConfirmedCases"].max()),
    x1=test_range[0],
    x2=test_range[1],
    alpha=0.1,
    label="test forecast horizon",
)

plt.legend()
```


## Training time series models with `TimeSeriesPredictor.fit`

Below we instantiate a `TimeSeriesPredictor` object and instruct AutoGluon to fit models that can forecast up to
5 time-points into the future (`prediction_length`) and save them in the folder `./autogluon-covidforecast`.
We also specify that AutoGluon should rank models according to mean absolute percentage error (MAPE) and that
the target field to be forecasted is `"ConfirmedCases"`.


```{.python .input}
predictor = TimeSeriesPredictor(
    path="autogluon-covidforecast",
    target="ConfirmedCases",
    prediction_length=prediction_length,
    eval_metric="MAPE",
)
predictor.fit(
    train_data=train_data,
    presets="low_quality",
)
```


In a short amount of time AutoGluon fits four time series forecasting models on the training data.
These models are three neural network forecasters: DeepAR, MQCNN, a simple feedforward neural network; and a simple exponential smoothing model with
automatic parameter tuning: Auto-ETS.
AutoGluon also constructs a weighted ensemble of these models capable of quantile forecasting.

## Evaluating the performance of different models

We can view the test performance of each model AutoGluon has trained via the `leaderboard()` method.
We provide the test data set to the leaderboard function to see how well our fitted models are doing on the held out time frame.
In AutoGluon leaderboards, higher scores always correspond to better predictive performance.
Therefore our MAPE scores are presented with a "flipped" sign, such that higher "negative MAPE"s correspond to better models.


```{.python .input}
predictor.leaderboard(test_data, silent=True)
```

## Making forecasts with `TimeSeriesPredictor.predict`

We can now use the `TimeSeriesPredictor` to look at actual forecasts.
By default, AutoGluon will select the best performing model to forecast time series with.
Let's use the predictor to compute forecasts, and plot forecasts for an example country.


```{.python .input}
predictions = predictor.predict(train_data)
```

Show forecasts as a table



```{.python .input}
plt.figure(figsize=(20, 3))

ytrue = train_data.loc['France_']["ConfirmedCases"]
ypred = predictions.loc['France_']

# prepend the last value of true range to predicted range for plotting continuity
ypred.loc[ytrue.index[-1]] = [ytrue[-1]] * 10
ypred = ypred.sort_index()

ytrue_test = test_data.loc['France_']["ConfirmedCases"][-5:]

plt.plot(ytrue[-30:], label="Training Data")
plt.plot(ypred["mean"], label="Mean Forecasts")
plt.plot(ytrue_test, label="Actual")

plt.fill_between(
    ypred.index, ypred["0.1"], ypred["0.9"], color="red", alpha=0.1
)
plt.title("COVID Case Forecasts in France, compared to actual trajectory")
_ = plt.legend()
```


As we used a "toy" presets setting (`presets="low_quality"`) our forecasts may appear to not be doing very well. In realistic scenarios,
users can set `presets` to be one of: `"best_quality"`, `"high_quality"`, `"good_quality"`, `"medium_quality"`.
Higher quality presets will generally produce superior forecasting accuracy but take longer to train and may produce less efficient models.

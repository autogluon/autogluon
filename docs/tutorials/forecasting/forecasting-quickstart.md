<!-- #region -->
Via a simple fit() call, AutoGluon can produce forecasts to a time series. This tutorial demonstrates a simple example about how to use AutoGluon to produce forecasts of comfirmed cases in a country given previous data.  


To start, import AutoGluon's ForecastingPredictor and TabularDataset
<!-- #endregion -->

```python
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset
```

Load training data from a CSV file into an AutoGluon Dataset object. This object is essentially equivalent to a Pandas DataFrame and the same methods can be applied to both, see details in TabulerDataset.

```python
train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
train_data[50:60]
```

Note that we loaded data from a CSV file stored in the cloud (AWS s3 bucket), but you can you specify a local file-path instead if you have already downloaded the CSV file to your own machine (e.g., using wget). Each row in the table train_data corresponds to a single training example. The dataset you use for forecasting task should usually contain three columns: date_column(here it is "Date"), target_column(here it is "ConfirmedCases"), index_column(here it is "name").

Let's use AutoGluon to train multiple models.

```python
save_path = "AutogluonModels/user/"
predictor = ForecastingPredictor(path=save_path).fit(train_data,
                                                     prediction_length=19,
                                                     index_column="name",
                                                     target_column="ConfirmedCases",
                                                     time_column="Date",
                                                     quantiles=[0.1, 0.5, 0.9],
                                                     presets="low_quality",
                                                    )
```

Note that we use the `low_quality` presets here for quick training because this is a notebook for illustration purpose. When you use by yourself, it is usually not a good idea to use `presets=low_quality`.


We can see the performance of each individual trained model on our test data:

```python
test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
```

```python
# reload the predictor
predictor = None
predictor = ForecastingPredictor.load(save_path)
predictor.leaderboard(test_data)
```

When we call predict(), AutoGluon automatically predicts with the model that displayed the best performance on validation data (i.e. the MQCNN).

```python
predictor.predict(test_data)['Afghanistan_']
```

We can also use the predictor.fit_summary() to summarize the fit process:

```python
predictor.fit_summary()
```

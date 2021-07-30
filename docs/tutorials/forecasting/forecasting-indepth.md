# Forecasting Time-Series - In Depth Tutorial
:label:`sec_forecastingindepth`

This more advanced tutorial describes how you can exert greater control over AutoGluon's time-series modeling. As an example forecasting task, we again use the COVID-19 dataset previously described in :ref:`sec_forecastingquick`.

```{.python .input}
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset

train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")

save_path = "agModels-covidforecast"
eval_metric = "mean_wQuantileLoss"  # just for demonstration, this is already the default evaluation metric
```

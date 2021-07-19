""" Example script for predicting columns of tables, demonstrating simple use-case """

from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset

train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

predictor = ForecastingPredictor().fit(train_data,
                                       prediction_length=19,
                                       index_column="name",
                                       target_column="ConfirmedCases",
                                       time_column="Date",
                                       # presets="low_quality"
                                       time_limit=60
                                       )

print(predictor.leaderboard())
print(predictor.evaluate(test_data))

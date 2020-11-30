from autogluon.forecasting.task.forecasting.forecasting import Forecasting as task

from autogluon.forecasting.task.forecasting.dataset import TimeSeriesDataset
import autogluon.core as ag
import matplotlib.pyplot as plt

dataset = TimeSeriesDataset(
    train_path="./COV19/processed_train.csv",
    test_path="./COV19/processed_test.csv",
    prediction_length=19,
    index_column="name",
    target_column="ConfirmedCases",
    time_column="Date")


predictor = task.fit(train_data=dataset.train_data,
                     val_data=dataset.test_data,
                     freq=dataset.freq,
                     prediction_length=dataset.prediction_length,)

print(predictor.leaderboard())

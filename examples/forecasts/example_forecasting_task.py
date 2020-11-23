from forecasting.task.forecasting.forecasting import Forecasting as task

from forecasting.task.forecasting.dataset import TimeSeriesDataset
import core as ag
import matplotlib.pyplot as plt

dataset = TimeSeriesDataset(
    train_path="./COV19/processed_train.csv",
    test_path="./COV19/processed_test.csv",
    prediction_length=19,
    index_column="name",
    target_column="ConfirmedCases",
    time_column="Date")

print(dataset.train_data, dataset.test_data)
metric = "MAPE"
predictor = task.fit(train_data=dataset.train_data,
                     test_data=dataset.test_data,
                     freq=dataset.freq,
                     prediction_length=dataset.prediction_length,
                     hyperparameter_tune=True,
                     hyperparameters={"MQCNN": {'context_length': ag.Int(1, 20),
                                                'epochs': 10,
                                                "num_batches_per_epoch": 10}},
                     num_trials=10)
print(predictor.leaderboard())

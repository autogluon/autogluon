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

# change this to specify search strategy, can try bayesopt, random, or skopt
searcher_type = "bayesopt"
# change this to specify eval metric, one of ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]
eval_metric = "mean_wQuantileLoss"

predictor = task.fit(train_data=dataset.train_data,
                     val_data=dataset.test_data,
                     freq=dataset.freq,
                     prediction_length=dataset.prediction_length,
                     hyperparameter_tune=True,
                     hyperparameters={"MQCNN": {'context_length': ag.Int(1, 20),
                                                'epochs': 10,
                                                "num_batches_per_epoch": 10}},
                     search_strategy=searcher_type,
                     eval_metric=eval_metric,
                     num_trials=10)

print(predictor.leaderboard())
from autogluon.forecasting.task.forecasting.forecasting import Forecasting as task

from autogluon.forecasting.utils.dataset_utils import time_series_dataset
import autogluon.core as ag
import pandas as pd
import autogluon.core.utils.loaders.load_pd as load_pd

train_csv = load_pd.load("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_csv = load_pd.load("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
# change this to specify search strategy, can try bayesopt, random, or skopt
searcher_type = "random"
# change this to specify eval metric, one of ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]
eval_metric = "mean_wQuantileLoss"

predictor = task.fit(train_data=train_csv,
                     prediction_length=19,
                     index_column="name",
                     target_column="ConfirmedCases",
                     time_column="Date",
                     hyperparameter_tune=True,
                     hyperparameters={"MQCNN": {'context_length': ag.Int(10, 20),
                                                'epochs': 100,
                                                "num_batches_per_epoch": 10}},
                     search_strategy=searcher_type,
                     eval_metric=eval_metric,
                     num_trials=10)

test_data = time_series_dataset(test_csv,
                                index_column="name",
                                target_column="ConfirmedCases",
                                time_column="Date")

print(predictor.leaderboard())
print(predictor.evaluate(test_data))
predictions = predictor.predict(test_data, quantiles=[0.1, 0.5, 0.9])
# print(predictions)


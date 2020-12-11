from autogluon.forecasting.task.forecasting.forecasting import Forecasting as task

from autogluon.forecasting.utils.dataset_utils import create_time_series_dataset
import autogluon.core as ag
import pandas as pd

train_csv = pd.read_csv("./COV19/processed_train.csv")
test_csv = pd.read_csv("./COV19/processed_test.csv")
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
                     hyperparameters={"MQCNN": {'context_length': ag.Int(1, 20),
                                                'epochs': 10,
                                                "num_batches_per_epoch": 10}},
                     search_strategy=searcher_type,
                     eval_metric=eval_metric,
                     num_trials=10)

test_data = create_time_series_dataset(test_csv,
                                       index_column="name",
                                       target_column="ConfirmedCases",
                                       time_column="Date")

print(predictor.leaderboard())
print(predictor.evaluate(test_data))
predicted_targets = predictor.predict(test_data)
print(list(predicted_targets)[0].quantile("0.5"))

from autogluon.forecasting import Forecasting as task
import autogluon.core as ag


train_data = task.Dataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_data = task.Dataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
# change this to specify search strategy, can try bayesopt, random, or skopt
searcher_type = "random"
# change this to specify eval metric, one of ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]
eval_metric = "mean_wQuantileLoss"

predictor = task.fit(train_data=train_data,
                     prediction_length=19,
                     index_column="name",
                     target_column="ConfirmedCases",
                     time_column="Date",
                     hyperparameter_tune=True,
                     quantiles=[0.1, 0.5, 0.9],
                     hyperparameters={"DeepAR": {'context_length': ag.Int(10, 20),
                                                 "num_batches_per_epoch": 10,
                                                 "epochs": 50},
                                      "SFF": {'context_length': ag.Int(10, 20),
                                              "num_batches_per_epoch": 10,
                                              "epochs": 50},
                                      "MQCNN": {'context_length': ag.Int(10, 20),
                                                "num_batches_per_epoch": 10,
                                                "epochs": 50},
                                      },
                     search_strategy=searcher_type,
                     eval_metric=eval_metric,
                     num_trials=3, )

print(predictor.leaderboard(test_data))
print(predictor.evaluate(test_data))
predictions = predictor.predict(test_data, quantiles=[0.5])
print(predictions['Afghanistan_'])

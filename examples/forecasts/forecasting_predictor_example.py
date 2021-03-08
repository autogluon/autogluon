from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset
import autogluon.core as ag

train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
# change this to specify search strategy, can try bayesopt, random, or skopt
searcher_type = "random"
# change this to specify eval metric, one of ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]
eval_metric = "mean_wQuantileLoss"

prediction_length = 19
path = "test_new_predictor"

predictor = ForecastingPredictor(path=path, eval_metric=eval_metric).fit(train_data,
                                                                         prediction_length,
                                                                         index_column="name",
                                                                         target_column="ConfirmedCases",
                                                                         time_column="Date",
                                                                         hyperparameter_tune=True,
                                                                         quantiles=[0.1, 0.5, 0.9],
                                                                         refit_full=True,
                                                                         search_strategy=searcher_type,
                                                                         hyperparameters={
                                                                             "MQCNN": {
                                                                                 'context_length': ag.Int(70, 90,
                                                                                                          default=prediction_length * 4),
                                                                                 "num_batches_per_epoch": 10,
                                                                                 "epochs": 5},
                                                                         },
                                                                         num_trials=3

                                                                         )

# Int: [min(max(10, 2*prediction_length), 250), min(500,12*prediction_length)]
predictor = None
predictor = ForecastingPredictor.load(path)
models = predictor._trainer.get_model_names_all()
for model in models:
    print(predictor._trainer.load_model(model).get_info())
print(predictor.leaderboard(test_data))
print(predictor.evaluate(test_data))
predictions = predictor.predict(test_data, quantiles=[0.5], time_series_to_predict=['Afghanistan_', "Algeria_"])
time_series_id = 'Afghanistan_'
print(predictions[time_series_id])

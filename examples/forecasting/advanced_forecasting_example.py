""" Example script for forecasting, demonstrating more advanced usage of fit().
    Note that all settings demonstrated here are just chosen for demonstration purposes (to minimize runtime), and do not represent wise choices to use in practice.
"""
import pandas as pd
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset
import autogluon.core as ag

train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
static_features = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries"
                                 "/toy_static_features.csv")
# change this to specify eval metric, one of ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]
eval_metric = "mean_wQuantileLoss"

prediction_length = 19
path = "hpo_models"

predictor = ForecastingPredictor(path=path, eval_metric=eval_metric).fit(train_data,
                                                                         prediction_length,
                                                                         static_features=static_features,
                                                                         index_column="name",
                                                                         target_column="ConfirmedCases",
                                                                         time_column="Date",
                                                                         # hyperparameter_tune_kwargs={
                                                                         #     'scheduler': 'local',
                                                                         #     'searcher': 'random',
                                                                         #     "num_trials": 2
                                                                         # },
                                                                         hyperparameter_tune_kwargs="auto",
                                                                         quantiles=[0.1, 0.5, 0.9],
                                                                         refit_full=True,
                                                                         hyperparameters={
                                                                             "MQCNN": {
                                                                                 'context_length': ag.Int(70, 90,
                                                                                                          default=prediction_length * 4),
                                                                                 "num_batches_per_epoch": 10,
                                                                                 "epochs": 5},
                                                                             "DeepAR": {
                                                                                 'context_length': ag.Int(70, 90,
                                                                                                          default=prediction_length * 4),
                                                                                 "num_batches_per_epoch": 10,
                                                                                 "epochs": 5},
                                                                             "SFF": {
                                                                                 'context_length': ag.Int(70, 90,
                                                                                                          default=prediction_length * 4),
                                                                                 "num_batches_per_epoch": 10,
                                                                                 "epochs": 5},
                                                                         },
                                                                         time_limits=5
                                                                         )


predictor = None
predictor = ForecastingPredictor.load(path)
models = predictor._trainer.get_model_names_all()
for model in models:
    print(predictor._trainer.load_model(model).get_info())
print(predictor.leaderboard(test_data, static_features=static_features))
print(predictor.evaluate(test_data, static_features=static_features))
predictions = predictor.predict(test_data, static_features=static_features, quantiles=[0.5], time_series_to_predict=['Afghanistan_', "Algeria_"])
time_series_id = 'Afghanistan_'
print(predictions[time_series_id])
targets = pd.DataFrame({"targets": [100 for i in range(prediction_length)]})
targets.index = predictions[time_series_id].index

all_targets = [targets for i in range(len(predictions))]
print(predictions[time_series_id].index.freq)
print(ForecastingPredictor.evaluate_predictions(predictions, all_targets))

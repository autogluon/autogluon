from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset
import autogluon.core as ag


def test_forecasting():
    train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
    test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

    searcher_type = "random"
    eval_metric = "mean_wQuantileLoss"
    path = "hpo_test_models"
    prediction_length = 19
    predictor = ForecastingPredictor(path=path, eval_metric=eval_metric).fit(train_data,
                                                                             prediction_length=prediction_length,
                                                                             index_column="name",
                                                                             target_column="ConfirmedCases",
                                                                             time_column="Date",
                                                                             hyperparameter_tune=True,
                                                                             quantiles=[0.1, 0.5, 0.9],
                                                                             refit_full=True,
                                                                             search_strategy=searcher_type,
                                                                             hyperparameters={
                                                                                 "MQCNN": {
                                                                                     'context_length': ag.Int(10, 20),
                                                                                     "num_batches_per_epoch": 10,
                                                                                     "epochs": 5},
                                                                             },
                                                                             num_trials=3

                                                                             )

    predictor.predict(test_data, quantiles=[0.1, 0.5, 0.9])


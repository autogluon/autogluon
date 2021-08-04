import pandas as pd
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset
import autogluon.core as ag
import tempfile


def test_forecasting_no_hpo():
    with tempfile.TemporaryDirectory() as path:
        train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
        test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

        predictor = ForecastingPredictor(path=path).fit(train_data,
                                                        prediction_length=19,
                                                        index_column="name",
                                                        target_column="ConfirmedCases",
                                                        time_column="Date",
                                                        presets="low_quality"
                                                        )
        print(predictor.predict(test_data))


def test_forecasting_hpo():
    with tempfile.TemporaryDirectory() as path:
        train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
        test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

        predictor = ForecastingPredictor(path=path).fit(train_data,
                                                        prediction_length=19,
                                                        index_column="name",
                                                        target_column="ConfirmedCases",
                                                        time_column="Date",
                                                        presets="low_quality_hpo"
                                                        )
        print(predictor.predict(test_data))


def test_forecasting_advance():
    with tempfile.TemporaryDirectory() as path:
        train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
        test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
        eval_metric = "mean_wQuantileLoss"

        prediction_length = 19

        predictor = ForecastingPredictor(path=path, eval_metric=eval_metric).fit(train_data,
                                                                                 prediction_length,
                                                                                 index_column="name",
                                                                                 target_column="ConfirmedCases",
                                                                                 time_column="Date",
                                                                                 hyperparameter_tune_kwargs={
                                                                                     'scheduler': 'local',
                                                                                     'searcher': 'random',
                                                                                     "num_trials": 2
                                                                                 },
                                                                                 quantiles=[0.1, 0.5, 0.9],
                                                                                 refit_full=True,
                                                                                 hyperparameters={
                                                                                     "MQCNN": {
                                                                                         'context_length': ag.Int(70,
                                                                                                                  90,
                                                                                                                  default=prediction_length*4),
                                                                                         "num_batches_per_epoch": 10,
                                                                                         "epochs": 2},
                                                                                     "DeepAR": {
                                                                                         'context_length': ag.Int(70,
                                                                                                                  90,
                                                                                                                  default=prediction_length*4),
                                                                                         "num_batches_per_epoch": 10,
                                                                                         "epochs": 2},
                                                                                     "SFF": {
                                                                                         'context_length': ag.Int(70,
                                                                                                                  90,
                                                                                                                  default=prediction_length*4),
                                                                                         "num_batches_per_epoch": 10,
                                                                                         "epochs": 2},
                                                                                 },
                                                                                 time_limits=10
                                                                                 )

        predictor = None
        predictor = ForecastingPredictor.load(path)
        models = predictor._trainer.get_model_names_all()
        for model in models:
            print(predictor._trainer.load_model(model).get_info())
        print(predictor.leaderboard(test_data))
        print(predictor.evaluate(test_data))
        predictions = predictor.predict(train_data, quantiles=[0.5],
                                        time_series_to_predict=['Afghanistan_', "Algeria_"])
        time_series_id = 'Afghanistan_'
        print(predictions[time_series_id])
        print(ForecastingPredictor.evaluate_predictions(predictions, test_data, index_column="name",
                                                        target_column="ConfirmedCases", time_column="Date"))


def test_forecasting_with_static_features():
    with tempfile.TemporaryDirectory() as path:
        train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
        test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
        static_features = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries"
                                         "/toy_static_features.csv")

        predictor = ForecastingPredictor(path=path).fit(train_data,
                                                        prediction_length=19,
                                                        static_features=static_features,
                                                        index_column="name",
                                                        target_column="ConfirmedCases",
                                                        time_column="Date",
                                                        presets="low_quality"
                                                        )
        print(predictor.predict(test_data, static_features=static_features))


def test_forecasting_mqcnn():
    with tempfile.TemporaryDirectory() as path:
        train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
        test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

        predictor = ForecastingPredictor(path=path).fit(train_data,
                                                        prediction_length=19,
                                                        index_column="name",
                                                        target_column="ConfirmedCases",
                                                        time_column="Date",
                                                        hyperparameters={
                                                            "MQCNN": {
                                                                "num_batches_per_epoch": 10,
                                                                "epochs": 2},
                                                        }
                                                        )
        print(predictor.predict(test_data))


def test_forecasting_deepar():
    with tempfile.TemporaryDirectory() as path:
        train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
        test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

        predictor = ForecastingPredictor(path=path).fit(train_data,
                                                        prediction_length=19,
                                                        index_column="name",
                                                        target_column="ConfirmedCases",
                                                        time_column="Date",
                                                        hyperparameters={
                                                            "DeepAR": {
                                                                "num_batches_per_epoch": 10,
                                                                "epochs": 2},
                                                        }
                                                        )
        print(predictor.predict(test_data))


def test_forecasting_sff():
    with tempfile.TemporaryDirectory() as path:
        train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
        test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

        predictor = ForecastingPredictor(path=path).fit(train_data,
                                                        prediction_length=19,
                                                        index_column="name",
                                                        target_column="ConfirmedCases",
                                                        time_column="Date",
                                                        hyperparameters={
                                                            "SFF": {
                                                                "num_batches_per_epoch": 10,
                                                                "epochs": 2},
                                                        }
                                                        )
        print(predictor.predict(test_data))

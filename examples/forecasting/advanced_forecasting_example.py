"""Example script for forecasting, demonstrating more advanced usage of fit().
Note that settings demonstrated here are just chosen for demonstration purposes
(to minimize runtime), and do not represent good choices for use in practice.
"""
from pprint import pprint

import autogluon.core as ag
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset

train_data = TabularDataset(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv"
)
test_data = TabularDataset(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv"
)
static_features = TabularDataset(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries"
    "/toy_static_features.csv"
)

# change this to specify eval metric, one of ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]
eval_metric = "mean_wQuantileLoss"
eval_quantiles = [0.5]
prediction_length = 19
predictor_path = "forecasting_hpo_models"
context_length_range = ag.Int(70, 90, default=prediction_length * 4)

if __name__ == "__main__":
    ForecastingPredictor(path=predictor_path, eval_metric=eval_metric).fit(
        train_data,
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
                "context_length": context_length_range,
                "num_batches_per_epoch": 10,
                "epochs": 5,
            },
            "DeepAR": {
                "context_length": context_length_range,
                "num_batches_per_epoch": 10,
                "epochs": 5,
            },
            "SFF": {
                "context_length": context_length_range,
                "num_batches_per_epoch": 10,
                "epochs": 30,
            },
        },
        time_limit=60,
    )

    # Load predictor from persisted path
    predictor = ForecastingPredictor.load(predictor_path)
    print("Best Model")
    pprint(predictor._trainer.get_model_best())

    # Print fit summary
    predictor.fit_summary()

    print("Summary of Models")
    models = predictor._trainer.get_model_names_all()
    for model in models:
        pprint(predictor._trainer.load_model(model).get_info())

    leaderboard = predictor.leaderboard(test_data, static_features=static_features)
    print("Model Leaderboard")
    print(leaderboard)

    print("Evaluations on Test Data")
    evaluations = predictor.evaluate(
        test_data, static_features=static_features, quantiles=eval_quantiles
    )
    print(f"Evaluation result: {evaluations}")

    time_series_id = "Afghanistan_"
    print(f"Showing predictions for {time_series_id}")
    predictions = predictor.predict(
        train_data, static_features=static_features, quantiles=eval_quantiles
    )
    print(predictions[time_series_id])
    prediction_eval_result = ForecastingPredictor.evaluate_predictions(
        forecasts=predictions,
        targets=test_data,
        index_column=predictor.index_column,
        time_column=predictor.time_column,
        target_column=predictor.target_column,
        eval_metric=predictor.eval_metric,
    )
    print(f"Evaluation Result for {time_series_id}: {prediction_eval_result}")

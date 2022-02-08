import os
import time

from gluonts.dataset.repository.datasets import get_dataset

from autogluon.forecasting import ForecastingPredictor


dataset_list = [
    "m4_hourly",
    "m4_daily",
    "m4_weekly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
    "traffic",
    "electricity",
]

if __name__ == "__main__":
    os.makedirs("benchmark_results", exist_ok=True)
    for dataset_name in dataset_list:
        print(f"Evaluating dataset {dataset_name}...")
        dataset = get_dataset(dataset_name)
        train_data = dataset.train
        test_data = dataset.test
        prediction_length = dataset.metadata.prediction_length
        freq = dataset.metadata.freq
        predictor = ForecastingPredictor(
            path=f"autogluon_benchmark_{dataset_name}"
        ).fit(
            train_data,
            prediction_length,
            index_column="index_column",  # TODO: column names redundant
            time_column="time_column",
            target_column="target_column",
            freq=freq,
            hyperparameter_tune_kwargs={
                "scheduler": "local",
                "searcher": "random",
                "num_trials": 10,
            },
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            refit_full=True,
            set_best_to_refit_full=True,
        )
        leaderboard = predictor.leaderboard(dataset.test)
        leaderboard.to_csv(
            f"benchmark_results/results_{dataset_name}_{int(time.time())}.csv",
            index=False,
        )
        print(leaderboard)
        print(predictor.evaluate(test_data))

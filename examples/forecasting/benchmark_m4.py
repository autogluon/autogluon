from gluonts.dataset.repository.datasets import get_dataset
from autogluon.forecasting import ForecastingPredictor
import os
import time

os.makedirs("benchmark_results", exist_ok=True)
dataset = get_dataset("m4_hourly")
train_data = dataset.train
test_data = dataset.test
prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
predictor = ForecastingPredictor(path="m4_benchmark").fit(train_data,
                                                          prediction_length,
                                                          freq=freq,
                                                          hyperparameter_tune=True,
                                                          quantiles=[0.1, 0.5, 0.9],
                                                          refit_full=True,
                                                          keep_only_best=True,
                                                          set_best_to_refit_full=True,
                                                          num_trials=5,
                                                          )
leaderboard = predictor.leaderboard()
leaderboard.to_csv(f"benchmark_results/results_{int(time.time())}.csv", index=False)
print(leaderboard)
print(predictor.evaluate(test_data))

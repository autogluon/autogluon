from gluonts.dataset.repository.datasets import get_dataset
from autogluon.forecasting import ForecastingPredictor
from autogluon.core.dataset import TabularDataset
import autogluon.core as ag

dataset = get_dataset("m4_hourly")
train_data = dataset.train
test_data = dataset.test
prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
# train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
# test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
# prediction_length = 19
predictor = ForecastingPredictor(path="m4_benchmark").fit(train_data,
                                                          prediction_length,
                                                          freq=freq,
                                                          hyperparameter_tune=True,
                                                          quantiles=[0.1, 0.5, 0.9],
                                                          refit_full=True,
                                                          keep_only_best=True,
                                                          set_best_to_refit_full=True,
                                                          num_trials=10,
                                                          )
leaderboard = predictor.leaderboard()
leaderboard.to_csv("./results.csv", index=False)
print(leaderboard)
print(predictor.evaluate(test_data))

from autogluon.forecasting.task.forecasting.forecasting import Forecasting as task

from autogluon.forecasting.task.forecasting.dataset_v2 import TimeSeriesDataset
from autogluon.forecasting.utils.dataset_utils import time_series_dataset
import autogluon.core.utils.loaders.load_pd as load_pd
import autogluon.core as ag
import matplotlib.pyplot as plt

train_csv = load_pd.load("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_csv = load_pd.load("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

test_data = time_series_dataset(test_csv,
                                index_column="name",
                                target_column="ConfirmedCases",
                                time_column="Date")
# set num_trials = 1 to make sure at least one model finished training.
predictor = task.fit(train_data=train_csv,
                     prediction_length=19,
                     index_column="name",
                     target_column="ConfirmedCases",
                     time_column="Date",
                     )

print(predictor.leaderboard())
print(predictor.evaluate(test_data))
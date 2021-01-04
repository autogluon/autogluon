from autogluon.forecasting.task.forecasting.forecasting import Forecasting as task

train_data = task.Dataset(file_path="https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_data = task.Dataset(file_path="https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

predictor = task.fit(train_data=train_data,
                     prediction_length=19,
                     index_column="name",
                     target_column="ConfirmedCases",
                     time_column="Date",
                     )

print(predictor.leaderboard())
print(predictor.evaluate(test_data))
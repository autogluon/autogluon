"""Example script for predicting columns of tables, demonstrating simple use-case"""

from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset

train_data = TabularDataset(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv"
)
test_data = TabularDataset(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv"
)

if __name__ == "__main__":
    predictor = ForecastingPredictor().fit(
        train_data,
        prediction_length=19,
        index_column="name",
        target_column="ConfirmedCases",
        time_column="Date",
        time_limit=60,
    )

    print("Model Leaderboard")
    print(predictor.leaderboard())
    print(f"Evaluation result on test data: {predictor.evaluate(test_data)}")

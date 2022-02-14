"""Example script for defining and using custom GluonTS models in AutoGluon Forecasting"""

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput

from autogluon.forecasting import ForecastingPredictor, TabularDataset
from autogluon.forecasting.models.gluonts_model.abstract_gluonts.abstract_gluonts_model import (
    AbstractGluonTSModel,
)
from autogluon.forecasting.utils.dataset_utils import time_series_dataset


class CustomDeepARModel(AbstractGluonTSModel):
    def __init__(
        self,
        path: str,
        freq: str,
        prediction_length: int,
        name: str = "DeepAR_custom",
        eval_metric: str = None,
        hyperparameters=None,
        model=None,
        **kwargs,
    ):
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            hyperparameters=hyperparameters,
            name=name,
            eval_metric=eval_metric,
            model=model,
            **kwargs,
        )

    def create_model(self):
        self.model = DeepAREstimator.from_hyperparameters(**self.params)


train_data = TabularDataset(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv"
)
test_data = TabularDataset(
    "https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv"
)

gluonts_train_data = time_series_dataset(
    train_data, index_column="name", target_column="ConfirmedCases", time_column="Date"
)
gluonts_test_data = time_series_dataset(
    test_data, index_column="name", target_column="ConfirmedCases", time_column="Date"
)

if __name__ == "__main__":
    epochs = 5
    example_target = "Afghanistan_"

    # training standalone custom model
    customized_model = CustomDeepARModel(
        path="AutogluonModels/",
        freq="D",
        prediction_length=19,
        hyperparameters={
            "epochs": epochs,
            "num_batches_per_epoch": 10,
            "distr_output": NegativeBinomialOutput(),
        },
    )
    customized_model.fit(gluonts_train_data)

    print(
        f"Custom model score on test data: {customized_model.score(gluonts_test_data)}"
    )
    predictions = customized_model.predict(gluonts_test_data)
    print(
        f"Custom model predictions on {example_target}: {predictions[example_target]}"
    )

    # Training custom model alongside other models using ForecastingPredictor.fit()
    custom_hyperparameters = {
        CustomDeepARModel: {
            "epochs": epochs,
            "num_batches_per_epoch": 10,
            "distr_output": NegativeBinomialOutput(),
        },
        "MQCNN": {"context_length": 19, "epochs": epochs, "num_batches_per_epoch": 10},
    }

    predictor = ForecastingPredictor().fit(
        train_data=train_data,
        prediction_length=19,
        index_column="name",
        target_column="ConfirmedCases",
        time_column="Date",
        hyperparameters=custom_hyperparameters,
        quantiles=[0.1, 0.5, 0.9],
    )

    print("Model Leaderboard")
    print(predictor.leaderboard())
    print(
        f"Predictions on {example_target}: {predictor.predict(test_data)[example_target]}"
    )

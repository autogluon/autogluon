import random
from unittest import mock

import pytest

from ..common import get_prediction_for_df


@pytest.fixture()
def patch_models():
    rng = random.Random(42)

    def mock_predict(self, data, **kwargs):
        return get_prediction_for_df(data, prediction_length=self.prediction_length)

    def mock_greedy_fit(self, predictions_per_window, *args, **kwargs):
        model_names = list(predictions_per_window.keys())
        weights = [rng.uniform(0, 1) for _ in range(len(model_names))]
        self.model_to_weight = {model_name: weights[i] / sum(weights) for i, model_name in enumerate(model_names)}
        return self

    with (
        mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.predict", mock_predict),
        mock.patch("autogluon.timeseries.models.local.naive.SeasonalNaiveModel.predict", mock_predict),
        mock.patch("autogluon.timeseries.models.ensemble.weighted.greedy.GreedyEnsemble.fit", mock_greedy_fit),
        # nvutil cudaInit and cudaShutdown is triggered for each run of the trainer. we disable this here
        mock.patch("autogluon.common.utils.resource_utils.ResourceManager.get_gpu_count", return_value=0),
    ):
        yield

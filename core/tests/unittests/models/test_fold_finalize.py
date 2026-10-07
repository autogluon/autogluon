from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy, _ray_predict_oof


@pytest.mark.parametrize("parallel", [False, True])
def test_fold_finalization_follows_oof_scoring(parallel):
    events = []
    prediction = np.array([0.2, 0.8])
    X = pd.DataFrame({"feature": [0, 1]})
    y = pd.Series([0, 1])

    class Model:
        def predict_proba(self, data, **kwargs):
            events.append("predict")
            return prediction

        def score_with_y_pred_proba(self, y, y_pred_proba):
            events.append("score")
            assert y_pred_proba is prediction
            return 0.9

        def _post_fit_oof(self, X, y):
            assert events == ["predict", "score"]
            events.append("finalize")

        def reduce_memory_size(self, **kwargs):
            events.append("reduce_memory")

    model = Model()
    if parallel:
        result, probabilities = _ray_predict_oof(model, X, y)
    else:
        strategy = SequentialLocalFoldFittingStrategy.__new__(SequentialLocalFoldFittingStrategy)
        strategy.X, strategy.y = X, y
        strategy.time_limit = None
        strategy.bagged_ensemble_model = SimpleNamespace(save_bag_folds=True)
        result, probabilities = strategy._predict_oof(
            model,
            {
                "fold": (np.array([], dtype=int), np.array([0, 1])),
                "folds_finished": 0,
                "folds_left": 1,
                "folds_to_fit": 1,
                "is_last_fold": True,
                "model_name_suffix": "S1F1",
                "random_seed": 0,
            },
        )
    assert result is model
    assert probabilities is prediction
    assert model.val_score == 0.9
    assert events == ["predict", "score", "finalize", "reduce_memory"]

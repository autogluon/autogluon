import numpy as np
import pandas as pd
import pytest

from autogluon.core.metrics import log_loss, roc_auc
from autogluon.tabular import TabularPredictor


def _train_test(label_type: str, n: int = 600):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    signal = 2 * df["x1"] + df["x2"] + rng.normal(size=n)
    df["y"] = np.where(signal > 0, "p", "n") if label_type == "binary" else signal
    return df.iloc[:400], df.iloc[400:]


@pytest.mark.parametrize("eval_metric", [roc_auc, log_loss], ids=["roc_auc", "log_loss"])
def test_model_feature_importance_uses_the_requested_eval_metric(tmp_path, eval_metric):
    """A model scores the permuted data with the `eval_metric` it is given, not with its own.

    It used to pick `predict_proba` for the requested metric but score with its own (here accuracy), which raised.
    For a level 1 model, the result is the one the trainer computes on the same features.
    """
    train, test = _train_test("binary")
    predictor = TabularPredictor(label="y", eval_metric="accuracy", path=str(tmp_path), verbosity=0).fit(
        train, hyperparameters={"GBM": {"num_boost_round": 20}}
    )
    trainer = predictor._trainer
    X = predictor._learner.transform_features(test.drop(columns="y"))
    y = predictor._learner.label_cleaner.transform(test["y"])
    model = trainer.load_model("LightGBM")

    fi_model = model.compute_feature_importance(X=X, y=y, eval_metric=eval_metric, num_shuffle_sets=2, silent=True)
    fi_trainer = trainer._get_feature_importance_raw(
        X=X, y=y, model="LightGBM", eval_metric=eval_metric, num_shuffle_sets=2, silent=True
    )

    pd.testing.assert_series_equal(fi_model["importance"].sort_index(), fi_trainer["importance"].sort_index())


def test_model_feature_importance_scores_quantile_models(tmp_path):
    """A quantile model passes its quantile levels to the metric when computing feature importance.

    Without them `pinball_loss` raised, which broke `feature_stage="transformed_model"`, the out-of-fold importance of
    bagged models, and feature pruning (which then kept every feature).
    """
    train, test = _train_test("quantile")
    predictor = TabularPredictor(
        label="y", problem_type="quantile", quantile_levels=[0.1, 0.5, 0.9], path=str(tmp_path), verbosity=0
    ).fit(
        train,
        hyperparameters={"GBM": {"num_boost_round": 20}},
        num_bag_folds=2,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    fi_oof = predictor.feature_importance(
        model="LightGBM_BAG_L1", feature_stage="transformed_model", num_shuffle_sets=2, silent=True
    )
    fi_test = predictor.feature_importance(
        test, model="LightGBM_BAG_L1", feature_stage="transformed_model", num_shuffle_sets=2, silent=True
    )

    for fi in (fi_oof, fi_test):
        assert fi.loc["x1", "importance"] > fi.loc["x3", "importance"]

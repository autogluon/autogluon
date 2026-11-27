import numpy as np
import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor


@pytest.fixture(scope="module")
def train_test():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    df["y"] = 2 * df["x1"] + df["x2"] + rng.normal(size=n)
    return df.iloc[:400], df.iloc[400:]


def test_model_feature_importance_of_feature_groups(tmp_path, train_test):
    """`feature_stage="transformed_model"` computes the importance of (feature_name, feature_list) groups.

    Groups used to be treated as features the model does not use, which raised when building the result. For a level 1
    model the importances are the ones of `feature_stage="transformed"`. Members the model does not use are ignored,
    and a group with none gets importance 0, like a single feature the model does not use.
    """
    train, test = train_test
    predictor = TabularPredictor(label="y", path=str(tmp_path), verbosity=0).fit(
        train, hyperparameters={"GBM": {"num_boost_round": 20}}
    )
    features = ["x1", "x2", ("x1_x2", ["x1", "x2"])]
    kwargs = dict(model="LightGBM", num_shuffle_sets=2, silent=True)

    fi_model = predictor.feature_importance(test, features=features, feature_stage="transformed_model", **kwargs)
    fi_transformed = predictor.feature_importance(test, features=features, feature_stage="transformed", **kwargs)
    pd.testing.assert_series_equal(fi_model["importance"].sort_index(), fi_transformed["importance"].sort_index())

    features = ["x1", ("x1_and_unused", ["x1", "unused"]), ("only_unused", ["unused"])]
    fi = predictor.feature_importance(test, features=features, feature_stage="transformed_model", **kwargs)
    assert fi.loc["x1_and_unused", "importance"] == fi.loc["x1", "importance"]
    assert fi.loc["only_unused", "importance"] == 0


def test_bagged_out_of_fold_feature_importance_of_feature_groups(tmp_path, train_test):
    """The out-of-fold importance of a bagged model, computed child by child, also accepts feature groups."""
    train, _ = train_test
    predictor = TabularPredictor(label="y", path=str(tmp_path), verbosity=0).fit(
        train,
        hyperparameters={"GBM": {"num_boost_round": 20}},
        num_bag_folds=2,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    fi = predictor.feature_importance(
        model="LightGBM_BAG_L1",
        features=["x1", "x3", ("x1_x2", ["x1", "x2"])],
        feature_stage="transformed_model",
        num_shuffle_sets=2,
        silent=True,
    )

    assert fi.loc["x1_x2", "importance"] > fi.loc["x1", "importance"] > fi.loc["x3", "importance"]

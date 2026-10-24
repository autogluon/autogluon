import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor


def test_weighted_ensemble_feature_importance_has_the_columns_of_other_models(tmp_path):
    """The weighted ensemble's `transformed_model` importance (its model weights) has a `p_value` column like any other.

    The column used to be named `p_score`.
    """
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = np.where(df["x1"] + rng.normal(size=n) > 0, "p", "n")
    train, test = df.iloc[:400], df.iloc[400:]
    predictor = TabularPredictor(label="y", path=str(tmp_path), verbosity=0).fit(
        train, hyperparameters={"GBM": {"num_boost_round": 20}, "RF": {"n_estimators": 10}}
    )

    fi_ensemble = predictor.feature_importance(
        test, model="WeightedEnsemble_L2", feature_stage="transformed_model", silent=True
    )
    fi_model = predictor.feature_importance(test, model="LightGBM", feature_stage="transformed_model", silent=True)

    assert list(fi_ensemble.columns) == list(fi_model.columns)
    assert fi_ensemble["p_value"].isna().all()

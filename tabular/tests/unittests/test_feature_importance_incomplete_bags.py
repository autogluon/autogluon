import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor


def test_out_of_fold_feature_importance_of_a_bag_left_incomplete_by_validation_folds(tmp_path):
    """A bag that `validation_folds` left at its first fold gets its out-of-fold importance on that fold's rows.

    The out-of-fold importance only went through complete bagging sets, so such a bag got an empty result, or an
    error, although its out-of-fold predictions and validation score come from that fold.
    """
    rng = np.random.default_rng(0)
    n = 600
    train_data = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    train_data["y"] = np.where(train_data["x1"] + 0.5 * rng.normal(size=n) > 0, "p", "n")
    hyperparameters = {
        "GBM": [
            {"num_boost_round": 40, "num_leaves": 4},
            {"num_boost_round": 3, "learning_rate": 0.005, "ag_args": {"name_suffix": "Weak"}},
        ]
    }
    predictor = TabularPredictor(label="y", path=str(tmp_path), verbosity=0).fit(
        train_data,
        hyperparameters=hyperparameters,
        num_bag_folds=4,
        validation_folds=1,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    trainer = predictor._trainer
    weights = trainer.load_model(predictor.model_best)._get_model_weights()
    # the ensemble leaves the weak model out, so its bag is never completed
    incomplete = [m for m in predictor.model_names() if m.endswith("_BAG_L1") and m not in weights]
    assert incomplete
    bag = trainer.load_model(incomplete[0])
    assert len(bag.models) == 1

    fi = predictor.feature_importance(
        model=bag.name, feature_stage="transformed_model", num_shuffle_sets=2, silent=True
    )

    X = trainer.get_inputs_to_model(model=bag, X=trainer.load_X(), fit=True)
    y = trainer.load_y()
    (val_idx,) = bag.get_oof_fold_val_idx(X=X, y=y)
    expected = bag.load_child(bag.models[0]).compute_feature_importance(
        X=X.iloc[val_idx], y=y.iloc[val_idx], num_shuffle_sets=2, silent=True
    )
    pd.testing.assert_series_equal(fi["importance"].sort_index(), expected["importance"].sort_index())
    assert (fi["n"] == 2).all()

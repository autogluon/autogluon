from autogluon.tabular.models.tabicl.tabicl_model import TabICLModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_tabicl():
    model_cls = TabICLModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        verify_single_prediction_equivalent_to_multi=True,
    )


def test_checkpoint_version_is_passed_to_the_model():
    """The pinned checkpoint must reach the estimator, not just be declared on the class.

    `get_checkpoint_version` existed but was never called, so the pin had no effect and the
    documented `(classification, regression)` tuple form was forwarded to tabicl raw, which does
    not accept it. The pin is currently identical to tabicl's own default, so nothing would look
    wrong today -- it would only diverge silently once tabicl ships a new default checkpoint.
    """
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"a": np.arange(20, dtype="float32"), "b": np.arange(20, dtype="float32")})
    y = pd.Series([0, 1] * 10)

    for problem_type, expected in (
        ("binary", TabICLModel.default_classification_model),
        ("regression", TabICLModel.default_regression_model),
    ):
        model = TabICLModel(problem_type=problem_type, eval_metric="accuracy" if problem_type == "binary" else "rmse")
        with patch.object(TabICLModel, "get_model_cls") as get_model_cls:
            try:
                model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
            except Exception:  # noqa: BLE001 - the mocked estimator cannot actually fit
                pass
        kwargs = get_model_cls.return_value.call_args.kwargs
        assert kwargs["checkpoint_version"] == expected


def test_checkpoint_version_hyperparameter_forms():
    """A bare string applies to both problem types; a tuple selects by problem type."""
    for problem_type, tuple_expected in (("binary", "clf.ckpt"), ("regression", "reg.ckpt")):
        model = TabICLModel(problem_type=problem_type)
        assert model.get_checkpoint_version({}) == (
            TabICLModel.default_classification_model
            if problem_type == "binary"
            else TabICLModel.default_regression_model
        )
        assert model.get_checkpoint_version({"checkpoint_version": "custom.ckpt"}) == "custom.ckpt"
        assert model.get_checkpoint_version({"checkpoint_version": ("clf.ckpt", "reg.ckpt")}) == tuple_expected


def test_batch_size_tiers_and_row_cap():
    """Batch size steps down with table size, bottoming out at 1 for the largest tables."""
    assert TabICLModel._get_batch_size(1_000_000) == 8
    assert TabICLModel._get_batch_size(5_000_000) == 4
    assert TabICLModel._get_batch_size(500_000_000) == 2
    assert TabICLModel._get_batch_size(500_000_001) == 1
    assert TabICLModel()._get_default_auxiliary_params()["max_rows"] == 500_000

import sys
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
@pytest.mark.parametrize("explicit_options", [False, True])
def test_tabpfn_applies_preprocessing_options_to_every_scaler(monkeypatch, problem_type, explicit_options):
    scalers = ["safepower", "quantile_uni"]
    options = (
        {
            "preprocessing/global": "svd_quarter_components",
            "preprocessing/categoricals": "onehot",
            "preprocessing/append_original": False,
        }
        if explicit_options
        else {}
    )
    hyperparameters = {"preprocessing/scaling": scalers, **options}
    X, y = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}), pd.Series([0, 1, 0, 1])
    model = TabPFNModel(problem_type=problem_type, hyperparameters=hyperparameters)
    model.initialize(X=X, y=y)
    original_params = dict(model.params)
    monkeypatch.setattr(model, "_resolve_model_path", lambda **kwargs: None)
    monkeypatch.setattr(model, "_resolve_fit_device", lambda **kwargs: "cpu")

    estimator = Mock()
    estimator.fit.return_value = estimator
    estimator.forced_inference_dtype_ = None
    estimator.executor_ = None
    constructor = Mock(return_value=estimator)
    tabpfn = ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = constructor
    tabpfn.TabPFNRegressor = constructor
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)

    model._fit(X=X, y=y, num_gpus=0, verbosity=0)

    received = constructor.call_args.kwargs
    assert received["inference_config"]["PREPROCESS_TRANSFORMS"] == [
        {
            "name": scaler,
            "global_transformer_name": options.get("preprocessing/global"),
            "categorical_name": options.get("preprocessing/categoricals", "numeric"),
            "append_original": options.get("preprocessing/append_original", True),
        }
        for scaler in scalers
    ]
    assert not any(key.startswith("preprocessing/") for key in received)
    assert model.params == original_params

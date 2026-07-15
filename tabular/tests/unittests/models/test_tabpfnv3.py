import numpy as np

from autogluon.tabular.models.tabpfnv3.tabpfnv3_model import TabPFNv3Model
from autogluon.tabular.testing import FitHelper

from .test_tabpfnv2 import _make_regression_data

toy_model_params = {"n_estimators": 1}


def test_tabpfnv3():
    model_cls = TabPFNv3Model
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabPFN returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_tabpfnv3_regression_output_type_median():
    X, y = _make_regression_data()
    model = TabPFNv3Model(
        problem_type="regression",
        hyperparameters={**toy_model_params, "regression_output_type": "median"},
    )
    model.fit(X=X, y=y)

    y_pred = model.predict(X)

    X_inner = model.preprocess(X)
    y_pred_median = model.model.predict(X_inner, output_type="median")
    y_pred_mean = model.model.predict(X_inner, output_type="mean")

    assert np.allclose(y_pred, y_pred_median)
    assert not np.allclose(y_pred, y_pred_mean)

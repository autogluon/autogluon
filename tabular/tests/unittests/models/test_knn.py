import numpy as np
import pandas as pd
import pytest

from autogluon.tabular.models.knn.knn_model import KNNModel
from autogluon.tabular.testing import FitHelper


def test_knn():
    model_cls = KNNModel
    model_hyperparameters = {"n_neighbors": 2}

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


# Explicit ids: a bare "regression" id would mark the test as a `--runregression` regression test (see conftest.py)
@pytest.mark.parametrize("problem_type", ["binary", "regression"], ids=lambda p: f"problem_type={p}")
def test_knn_oof_is_aligned_when_fit_on_a_sample(problem_type, monkeypatch, tmp_path):
    """When the time limit stops the fit on a sample of the rows, the out-of-fold predictions of the rows the model was
    fit on (its leave-one-out predictions) must still belong to those rows."""
    rng = np.random.default_rng(0)
    n = 12000  # above the 10000 rows from which the fit samples rows under a time limit
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    if problem_type == "binary":
        y = pd.Series((X["a"] + X["b"] > 0).astype(int))
        eval_metric = "accuracy"
    else:
        y = X["a"] + X["b"]
        eval_metric = "root_mean_squared_error"

    # Leave no time after the first sample, so the fit stops before using all rows
    fit_with_samples = KNNModel._fit_with_samples

    def _fit_with_samples_no_time(self, X, y, model_params, time_limit, **kwargs):
        return fit_with_samples(self, X, y, model_params, time_limit=1e-6, **kwargs)

    monkeypatch.setattr(KNNModel, "_fit_with_samples", _fit_with_samples_no_time)

    model = KNNModel(
        path=str(tmp_path),
        name="KNN",
        problem_type=problem_type,
        eval_metric=eval_metric,
        hyperparameters={"ag.use_daal": False},
    )
    model.fit(X=X, y=y, time_limit=1000)
    assert model._X_unused_index  # the fit used a sample of the rows

    oof = model.predict_proba_oof(X)
    used = np.setdiff1d(np.arange(n), model._X_unused_index)
    if problem_type == "binary":
        accuracy = ((oof[used] > 0.5).astype(int) == y.to_numpy()[used]).mean()
        assert accuracy > 0.95
    else:
        correlation = np.corrcoef(oof[used], y.to_numpy()[used])[0, 1]
        assert correlation > 0.95

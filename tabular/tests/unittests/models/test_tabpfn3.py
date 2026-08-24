import os

import pytest

from autogluon.tabular.models.tabpfnv2.tabpfn3_model import TabPFN3Model
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 1}


@pytest.mark.skip(
    reason="TabPFN-3 model weights are not available publicly without accepting a license agreement; "
    "run manually on a machine with the checkpoints in the tabpfn cache."
)
def test_tabpfn3():
    model_cls = TabPFN3Model
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabPFN returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_tabpfn3_and_2_6_have_no_feature_cap():
    """TabPFN-2.6 and -3 must not cap features, while the 2.5 base still does.

    `ag.max_features` skips a fit outright, and these two models are the strongest methods on
    BeyondArena's widest tasks (up to 22k columns), so a cap would exclude them exactly where
    they win. The cap must be `None` rather than absent: `_default_auxiliary_params_extra`
    entries merge base-most class first, so an absent key would inherit the 2.5 base's cap.
    """
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_6_model import TabPFNv26Model

    assert TabPFN3Model()._get_default_auxiliary_params()["max_features"] is None
    assert TabPFNv26Model()._get_default_auxiliary_params()["max_features"] is None
    # The 2.5 base is unchanged, which is what makes the explicit None load-bearing.
    assert TabPFNModel()._get_default_auxiliary_params()["max_features"] == 2000


def test_tabpfn_preprocess_preserves_missing_categoricals():
    """Categorical columns reach TabPFN as `category` dtype, with missing values intact.

    Label-encoding them first loses the missingness: `.cat.codes` maps missing to -1, and
    TabPFN casts every column named in `categorical_features_indices` back to `category`
    (`tabpfn.preprocessing.clean.fix_dtypes`), which turns that -1 into an ordinary level. The
    model then sees no missing values in those columns. Applies to every TabPFN version, since
    they share this `_preprocess`.
    """
    import numpy as np
    import pandas as pd

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_6_model import TabPFNv26Model

    for model_cls in (TabPFN3Model, TabPFNv26Model):
        levels = np.array(["absent", "mild", "normal"], dtype=object)
        rng = np.random.default_rng(0)
        values = levels[rng.integers(0, 3, 40)]
        values[[3, 11, 27]] = None
        X = pd.DataFrame({"num": rng.normal(size=40), "cat": pd.Series(values, dtype="category")})

        model = model_cls(problem_type="binary", eval_metric=None)
        model._preprocess_set_features(X=X)
        processed = model.preprocess(X, is_train=True)

        assert str(processed["cat"].dtype) == "category", model_cls.__name__
        assert processed["cat"].isna().sum() == 3, model_cls.__name__
        assert -1 not in set(processed["cat"].cat.categories), model_cls.__name__
        # the indices TabPFN is told about still point at the categorical column: it infers
        # modality from dtype, and an integer-levelled `category` column reads as numeric, so
        # without them such a column would be treated as NUMERICAL.
        assert model._cat_indices == [X.columns.get_loc("cat")], model_cls.__name__
        # untouched numeric column
        assert processed["num"].equals(X["num"]), model_cls.__name__


def test_tabpfn_narrows_inference_context_to_the_dtype_inference_uses():
    """The cached in-context training set is stored at float32/int32, not float64/int64.

    TabPFN builds one preprocessed copy of the training data per ensemble member and
    converts it to float32 at predict time, so the wider arrays cost memory and disk
    without ever being read at full width.
    """
    import numpy as np

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(problem_type="binary", eval_metric=None)
    model.model = _stub_estimator(n_members=2, forced_inference_dtype=None)

    model._narrow_inference_context()

    for member in model.model.executor_.ensemble_members:
        assert member.X_train.dtype == np.float32
        assert member.y_train.dtype == np.int32


def test_tabpfn_keeps_inference_context_when_precision_is_forced_wider():
    """`inference_precision=torch.float64` is the case where the wider dtype is used."""
    import numpy as np
    import torch

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(problem_type="binary", eval_metric=None)
    model.model = _stub_estimator(n_members=2, forced_inference_dtype=torch.float64)

    model._narrow_inference_context()

    for member in model.model.executor_.ensemble_members:
        assert member.X_train.dtype == np.float64
        assert member.y_train.dtype == np.int64


class _StubOnDemandExecutor:
    """`InferenceEngineOnDemand` keeps the raw arrays, with no ensemble members."""

    def __init__(self, y_dtype, rng):
        import numpy as np

        self.X_train = rng.normal(size=(8, 3))
        self.y_train = rng.integers(0, 2, 8).astype(y_dtype)


class _StubOnDemandEstimator:
    def __init__(self, y_dtype, rng):
        self.executor_ = _StubOnDemandExecutor(y_dtype, rng)
        self.forced_inference_dtype_ = None


def _stub_on_demand_estimator(y_dtype):
    import numpy as np

    return _StubOnDemandEstimator(y_dtype, np.random.default_rng(0))


def _stub_estimator(n_members: int, forced_inference_dtype):
    """A stand-in for a fitted TabPFN estimator, so the test needs no checkpoint."""
    import numpy as np

    rng = np.random.default_rng(0)

    class _Member:
        def __init__(self):
            self.X_train = rng.normal(size=(8, 3))
            self.y_train = rng.integers(0, 2, 8)

    class _Executor:
        def __init__(self):
            self.ensemble_members = [_Member() for _ in range(n_members)]

    class _Estimator:
        def __init__(self):
            self.executor_ = _Executor()
            self.forced_inference_dtype_ = forced_inference_dtype

    return _Estimator()


def test_tabpfn_narrows_low_memory_features_but_not_a_float_target():
    """`fit_mode="low_memory"` keeps the raw training set and re-preprocesses per predict.

    Narrowing the features there is still lossless, but narrowing a float target is
    not: its transforms would then be computed at the narrower precision. An integer
    target (classification) is exact either way.
    """
    import numpy as np

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    for y_dtype, expected in ((np.int64, np.int32), (np.float64, np.float64)):
        model = TabPFNModel(problem_type="binary", eval_metric=None)
        model.model = _stub_on_demand_estimator(y_dtype=y_dtype)

        model._narrow_inference_context()

        assert model.model.executor_.X_train.dtype == np.float32
        assert model.model.executor_.y_train.dtype == expected
def test_tabpfn_save_keeps_foundation_weights_out_of_the_pickle(tmp_path, monkeypatch):
    """`save` writes the fitted state to a sidecar and `load` reattaches it.

    The weights are identical for every model of a TabPFN version, so pickling them
    per model writes a copy of the checkpoint each time. This covers AutoGluon's
    wiring with a stubbed tabpfn save/load pair, so it needs no checkpoint.
    """
    import pickle

    import tabpfn

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    estimator = _stub_estimator(n_members=1, forced_inference_dtype=None)
    sidecar = {}

    def _fake_save(est, path):
        sidecar["path"] = path
        sidecar["estimator"] = est
        open(path, "wb").close()

    def _fake_load(path, *, device):
        sidecar["device"] = device
        return sidecar["estimator"]

    monkeypatch.setattr(tabpfn, "save_fitted_tabpfn_model", _fake_save, raising=False)
    monkeypatch.setattr(tabpfn, "load_fitted_tabpfn_model", _fake_load, raising=False)

    model = TabPFNModel(problem_type="binary", eval_metric=None, path=str(tmp_path))
    model.model = estimator
    saved_path = model.save()

    assert sidecar["path"].endswith(TabPFNModel.tabpfn_fit_file_name)
    # The pickle no longer carries the estimator, so it cannot carry the weights.
    with open(os.path.join(saved_path, TabPFNModel.model_file_name), "rb") as f:
        assert pickle.load(f).model is None
    # ... while the live model is left fit.
    assert model.model is estimator

    loaded = TabPFNModel.load(saved_path)
    assert loaded.is_fit()
    assert loaded.model is estimator


def test_tabpfn_save_without_fit_writes_no_sidecar(tmp_path):
    """An unfit model has no fitted state to put in a sidecar."""
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(problem_type="binary", eval_metric=None, path=str(tmp_path))
    saved_path = model.save()

    assert not os.path.exists(os.path.join(saved_path, TabPFNModel.tabpfn_fit_file_name))
    assert not TabPFNModel.load(saved_path).is_fit()

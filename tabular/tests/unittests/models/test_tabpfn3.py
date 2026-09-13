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
        self.X_train = rng.normal(size=(8, 3))
        self.y_train = rng.integers(0, 2, 8).astype(y_dtype)


class _StubOnDemandEstimator:
    def __init__(self, y_dtype, rng):
        self.executor_ = _StubOnDemandExecutor(y_dtype, rng)
        self.forced_inference_dtype_ = None


def _stub_on_demand_estimator(y_dtype):
    import numpy as np

    return _StubOnDemandEstimator(y_dtype, np.random.default_rng(0))


class _StubEnsembleMember:
    """Holds the in-context arrays the real ensemble members carry."""

    def __init__(self, rng):
        self.X_train = rng.normal(size=(8, 3))
        self.y_train = rng.integers(0, 2, 8)


class _StubNetwork:
    def to(self, device):
        return self


class _StubExecutor:
    def __init__(self, n_members, rng):
        self.ensemble_members = [_StubEnsembleMember(rng) for _ in range(n_members)]
        self.model_caches = None

    def _set_models(self, models):
        self.models = list(models)


class _StubEstimator:
    """A stand-in for a fitted TabPFN estimator, so the tests need no checkpoint."""

    def __init__(self, n_members, forced_inference_dtype, rng):
        import torch

        self.executor_ = _StubExecutor(n_members, rng)
        self.forced_inference_dtype_ = forced_inference_dtype
        self.devices_ = [torch.device("cpu")]
        self.models_ = [_StubNetwork()]

    def to(self, device):
        import torch

        self.devices_ = [torch.device(device)]
        return self


def _stub_estimator(n_members: int, forced_inference_dtype):
    import numpy as np

    return _StubEstimator(n_members, forced_inference_dtype, np.random.default_rng(0))


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
    """Under `ag.save_pretrained_weights=False`, the pickle carries the fitted state without
    the network, and `load` attaches the process's shared network for the checkpoint.

    The weights are identical for every model of a TabPFN version, so pickling them per
    model writes a copy of the checkpoint each time. The shared-network registry is
    stubbed, so this needs no checkpoint.
    """
    import pickle
    from types import SimpleNamespace

    from autogluon.tabular.models.tabpfnv2 import tabpfnv2_5_model
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    estimator = _stub_estimator(n_members=1, forced_inference_dtype=None)
    network = estimator.models_[0]
    shared = _StubNetwork()
    requests = []

    def _fake_shared_model_specs(checkpoint_path, estimator_type, device):
        requests.append((checkpoint_path, estimator_type, device))
        return SimpleNamespace(model=shared)

    monkeypatch.setattr(tabpfnv2_5_model, "_shared_model_specs", _fake_shared_model_specs)

    model = TabPFNModel(
        problem_type="binary",
        eval_metric=None,
        path=str(tmp_path),
        hyperparameters={"ag.save_pretrained_weights": False},
    )
    model.initialize()
    model.model = estimator
    model.device = "cpu"  # normally set during fit; this test does not fit
    model._checkpoint_path = "checkpoint.ckpt"
    model._estimator_type = "classifier"
    saved_path = model.save()

    with open(os.path.join(saved_path, TabPFNModel.model_file_name), "rb") as f:
        pickled = pickle.load(f).model
    assert pickled.models_ is None
    assert pickled.executor_.model_caches is None
    assert pickled.executor_.ensemble_members[0].X_train.shape == (8, 3), "the fitted state is kept"
    # ... while the live model keeps its network.
    assert model.model is estimator
    assert estimator.models_ == [network]

    loaded = TabPFNModel.load(saved_path)
    assert loaded.is_fit()
    assert loaded.model.models_ == [shared]
    assert loaded.model.executor_.models == [shared]
    assert requests == [("checkpoint.ckpt", "classifier", "cpu")]


def test_tabpfn_references_pretrained_weights_by_default(tmp_path):
    """The default is to reference the weights, not to write a copy per model.

    The behaviour that follows from this default is covered by
    `test_tabpfn_save_keeps_foundation_weights_out_of_the_pickle`, which stubs tabpfn's
    save/load pair; this pins the default itself so a schema change cannot flip it silently.
    """
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(problem_type="binary", eval_metric=None, path=str(tmp_path))
    model.initialize()

    assert model.aux_params.save_pretrained_weights is False


def test_tabpfn_save_pretrained_weights_true_keeps_the_estimator_in_the_pickle(tmp_path):
    """Opting in gives a self-contained save: the estimator stays in the pickle."""
    import pickle

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(
        problem_type="binary",
        eval_metric=None,
        path=str(tmp_path),
        hyperparameters={"ag.save_pretrained_weights": True},
    )
    model.initialize()
    model.model = _stub_estimator(n_members=1, forced_inference_dtype=None)
    model.device = "cpu"  # normally set during fit; this test does not fit
    saved_path = model.save()

    with open(os.path.join(saved_path, TabPFNModel.model_file_name), "rb") as f:
        assert pickle.load(f).model.models_ is not None, "the network is in the pickle"
    assert TabPFNModel.load(saved_path).is_fit()


def test_tabpfn_save_without_fit_round_trips(tmp_path):
    """An unfit model has no network to detach and loads back unfit."""
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(
        problem_type="binary",
        eval_metric=None,
        path=str(tmp_path),
        hyperparameters={"ag.save_pretrained_weights": False},
    )
    model.initialize()
    saved_path = model.save()

    assert not TabPFNModel.load(saved_path).is_fit()


class _StubWeightedExecutor:
    """Holds the checkpoints the way the inference engine does, and swaps them like `load_state`."""

    def __init__(self, models, rng):
        self.models = models
        self.X_train = rng.normal(size=(64, 8))

    def _set_models(self, models):
        self.models = models


class _StubWeightedEstimator:
    """A fitted estimator whose checkpoints are small real modules, shared with its engine."""

    def __init__(self, n_features, rng):
        import torch

        self.models_ = [torch.nn.Linear(n_features, n_features), torch.nn.Linear(n_features, n_features)]
        self.executor_ = _StubWeightedExecutor(self.models_, rng)


def test_tabpfn_memory_size_counts_the_weights_without_pickling_them(monkeypatch):
    """`get_memory_size` matches a full pickle while never serialising the checkpoints."""
    import numpy as np
    import torch

    from autogluon.core.models import AbstractModel
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(problem_type="binary", eval_metric=None)
    model.model = _StubWeightedEstimator(n_features=512, rng=np.random.default_rng(0))
    full_pickle_size = AbstractModel._get_memory_size(model)
    weights = 2 * (512 * 512 + 512) * 4

    pickled_modules = []
    reduce = torch.nn.Module.__reduce_ex__

    def spy_reduce(self, protocol):
        pickled_modules.append(self)
        return reduce(self, protocol)

    monkeypatch.setattr(torch.nn.Module, "__reduce_ex__", spy_reduce)
    memory_size = model.get_memory_size()

    assert pickled_modules == []
    assert weights < memory_size
    assert abs(memory_size - full_pickle_size) < 0.01 * full_pickle_size
    # The live model is left as it was.
    assert model.model.models_ and model.model.executor_.models is model.model.models_


def test_tabpfn_memory_size_of_an_unfit_model_is_the_pickle():
    from autogluon.core.models import AbstractModel
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    model = TabPFNModel(problem_type="binary", eval_metric=None)
    assert model.get_memory_size() == AbstractModel._get_memory_size(model)


def test_tabpfn_auto_max_batch_size_resolution():
    """ "auto" chunking starts only once the prediction set exceeds the training set by the slack."""
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

    assert TabPFNModel._resolve_auto_max_batch_size(n_train=500) == 1_000, "floor for TabPFN-2.5"
    assert TabPFNModel._resolve_auto_max_batch_size(n_train=20_000) == 20_000, "no slack for TabPFN-2.5"
    assert TabPFN3Model._resolve_auto_max_batch_size(n_train=500) == 100_500
    assert TabPFN3Model._resolve_auto_max_batch_size(n_train=100_000) == 200_000
    assert TabPFN3Model._resolve_auto_max_batch_size(n_train=500_000) == 600_000
    assert TabPFN3Model._resolve_auto_max_batch_size(n_train=950_000) == 1_000_000, "capped at 1M"
    # the memory-estimate proxy stays bounded by the training size
    assert TabPFN3Model._n_test_for_memory_estimate(n_train=500_000, hyperparameters=None) == 500_000
    assert (
        TabPFN3Model._n_test_for_memory_estimate(n_train=500_000, hyperparameters={"ag.max_batch_size": 20_000})
        == 20_000
    )

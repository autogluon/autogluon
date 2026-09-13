import pytest
import torch

from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import RealTabPFNv2Model
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 1}


def test_tabpfnv2():
    model_cls = RealTabPFNv2Model
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabPFN returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="needs 2+ GPUs: with one device `models_` IS the engine's only cached copy, so it "
    "moves with `.to()` and the bug cannot reproduce",
)
def test_tabpfn_set_device_moves_inner_checkpoints():
    """`set_device("cpu")` must leave nothing on CUDA, or the saved artifact cannot load on CPU.

    `tabpfn.base.estimator_to_device` (which backs `estimator.to()`) moves the inference engine's
    per-device model caches but not the loaded checkpoints in `models_`, and those are pickled with
    the model. Without `TabPFNModel._set_device` moving them too, a GPU fit produces an artifact
    that raises "Attempting to deserialize object on a CUDA device" on a CPU-only machine.

    Only reproduces with several devices: with one, `models_[i]` *is* the engine's only cached
    copy and moves with it. CI runs on a single GPU (`NVIDIA_VISIBLE_DEVICES` holds one UUID), so
    this is skipped there -- nothing in CI exercises the fix, which is also why the bug survived
    until now.

    Only RealTabPFN-v2 is exercised: its checkpoint is the one CI can download, while TabPFN-2.6
    and -3 need a one-time license acceptance (which is also why `test_tabpfn3` is skipped). They
    inherit this `_set_device` unchanged, so they are covered by construction -- verified manually
    against locally cached checkpoints for both.
    """
    import numpy as np
    import pandas as pd

    from autogluon.tabular import TabularPredictor

    model_cls = RealTabPFNv2Model

    rng = np.random.RandomState(0)
    data = pd.DataFrame({"a": rng.rand(60), "b": rng.rand(60)})
    data["label"] = (data["a"] > 0.5).astype(int)

    predictor = TabularPredictor(label="label", verbosity=0).fit(
        data,
        hyperparameters={model_cls: toy_model_params},
        fit_weighted_ensemble=False,
    )
    model = predictor._trainer.load_model(predictor.model_names()[0])
    assert model.get_device() == "cuda", "expected a GPU fit"

    model.set_device("cpu")

    on_cuda = [
        name
        for inner_model in getattr(model.model, "models_", None) or []
        for name, parameter in inner_model.named_parameters()
        if parameter.is_cuda
    ]
    assert not on_cuda, f"parameters left on CUDA after set_device('cpu'): {on_cuda}"

    # The checkpoints must be the engine's own copies, not separate ones, or the weights are
    # stored twice: once via `models_` and once via the engine cache.
    cached = {id(cache.get(device)) for cache in model.model.executor_.model_caches for device in cache.get_devices()}
    assert cached == {id(inner_model) for inner_model in model.model.models_}, (
        "models_ must reference the inference engine's cached checkpoints, not duplicates"
    )


def test_tabpfn_models_share_one_network_and_pickle_without_it(tmp_path):
    """Two fits from one checkpoint use the same network object, the saved model holds no
    weights, and a load with an empty registry rebuilds the network and predicts the same.
    """
    import os
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    from autogluon.tabular.models.tabpfnv2 import tabpfnv2_5_model

    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    y = pd.Series(y)

    def fit(name):
        model = RealTabPFNv2Model(
            path=str(tmp_path / name),
            name=name,
            problem_type="binary",
            eval_metric="log_loss",
            hyperparameters=dict(toy_model_params),
        )
        model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
        return model

    first, second = fit("first"), fit("second")
    assert first.model.models_[0] is second.model.models_[0], "one network per checkpoint and process"

    saved_path = first.save()
    with open(os.path.join(saved_path, RealTabPFNv2Model.model_file_name), "rb") as f:
        pickled = pickle.load(f)
    assert pickled.model.models_ is None
    n_params = sum(p.numel() for p in first.model.models_[0].parameters())
    assert os.path.getsize(os.path.join(saved_path, RealTabPFNv2Model.model_file_name)) < n_params

    expected = first.predict_proba(X)
    tabpfnv2_5_model._MODEL_SPECS.clear()
    loaded = RealTabPFNv2Model.load(saved_path)
    assert loaded.model.models_[0] is not first.model.models_[0], "rebuilt from the checkpoint"
    np.testing.assert_allclose(loaded.predict_proba(X), expected, rtol=1e-5, atol=1e-6)
    assert loaded.model.models_[0] is fit("third").model.models_[0], "and shared again"


def test_tabpfn_shared_network_capacity_bounds_the_registry(tmp_path, monkeypatch):
    """The registry keeps the `shared_network_capacity` most recently used networks; an
    evicted network lives on in the estimators that hold it, and capacity 0 shares nothing.
    """
    import pandas as pd
    from sklearn.datasets import make_classification, make_regression

    from autogluon.tabular.models.tabpfnv2 import tabpfnv2_5_model

    Xc, yc = make_classification(n_samples=60, n_features=4, random_state=0)
    Xr, yr = make_regression(n_samples=60, n_features=4, random_state=0)
    data = {
        "binary": (pd.DataFrame(Xc, columns=list("abcd")), pd.Series(yc), "log_loss"),
        "regression": (pd.DataFrame(Xr, columns=list("abcd")), pd.Series(yr), "rmse"),
    }

    def fit(name, problem_type):
        X, y, metric = data[problem_type]
        model = RealTabPFNv2Model(
            path=str(tmp_path / name),
            name=name,
            problem_type=problem_type,
            eval_metric=metric,
            hyperparameters=dict(toy_model_params),
        )
        model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
        return model

    tabpfnv2_5_model.release_shared_networks()
    monkeypatch.setattr(RealTabPFNv2Model, "shared_network_capacity", 1)
    classifier = fit("classifier", "binary")
    regressor = fit("regressor", "regression")
    assert len(tabpfnv2_5_model._MODEL_SPECS) == 1, "the regressor's network evicted the classifier's"
    assert classifier.model.models_[0] is not None, "the evicted network stays with its estimator"
    classifier.predict_proba(data["binary"][0])
    rebuilt = fit("classifier_again", "binary")
    assert rebuilt.model.models_[0] is not classifier.model.models_[0], "rebuilt after eviction"
    assert regressor.model.models_[0] is not rebuilt.model.models_[0]

    monkeypatch.setattr(RealTabPFNv2Model, "shared_network_capacity", 2)
    regressor_again = fit("regressor_again", "regression")
    assert regressor_again.model.models_[0] is not regressor.model.models_[0], "the rebuild evicted it"
    assert fit("classifier_third", "binary").model.models_[0] is rebuilt.model.models_[0], "still registered"
    assert len(tabpfnv2_5_model._MODEL_SPECS) == 2

    monkeypatch.setattr(RealTabPFNv2Model, "shared_network_capacity", 0)
    tabpfnv2_5_model.release_shared_networks()
    unshared = fit("unshared", "binary")
    assert unshared.model.models_[0] is not fit("unshared_too", "binary").model.models_[0]
    assert not tabpfnv2_5_model._MODEL_SPECS

import pytest
import torch

from autogluon.core.models.abstract import _object_graph as graph
from autogluon.core.models.abstract import _shared_weights_registry as registry
from autogluon.core.models.abstract import shared_weights as sw
from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import RealTabPFNv2Model, TabPFNModel
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


def _fit_tabpfn(tmp_path, name, X, y, *, problem_type="binary", eval_metric="log_loss", hyperparameters=None):
    model = RealTabPFNv2Model(
        path=str(tmp_path / name),
        name=name,
        problem_type=problem_type,
        eval_metric=eval_metric,
        hyperparameters={**toy_model_params, **(hyperparameters or {})},
    )
    model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
    return model


def _classification_frame():
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), pd.Series(y)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Every test starts and ends with an empty shared-weights registry at the default capacity."""
    registry.release()
    registry.reset_stats()
    registry.set_capacity(registry.DEFAULT_CAPACITY)
    yield
    registry.release()
    registry.reset_stats()
    registry.set_capacity(registry.DEFAULT_CAPACITY)


def test_tabpfn_models_share_one_network_and_pickle_without_it(tmp_path):
    """Two fits from one checkpoint use the same network object, the saved model holds no
    weights, and a load with an empty registry rebuilds the network and predicts the same.
    """
    import os
    import pickle

    import numpy as np

    X, y = _classification_frame()
    first, second = _fit_tabpfn(tmp_path, "first", X, y), _fit_tabpfn(tmp_path, "second", X, y)
    assert first.model.models_[0] is second.model.models_[0], "one network per checkpoint and process"
    assert first._shared_state.key == second._shared_state.key
    assert registry.peek(first._shared_state.key).hits == 1
    info = first.get_info()["shared_weights"]
    assert info["library"] == "tabpfn" and info["variant"] == "TabPFNClassifier._initialize_model_variables"
    assert info["device"] == "cpu" and info["loaded_by"] == "fit" and info["present_before_fit"] is False
    assert second.get_info()["shared_weights"]["present_before_fit"] is True
    assert info["checkpoint"] == RealTabPFNv2Model.default_classification_model
    assert isinstance(first.model.model_path, str), "the estimator keeps tabpfn's own model_path argument"

    payload = registry.peek(first._shared_state.key).value
    network = payload.attributes["models_"][0]
    assert network is first.model.models_[0], "the registry holds what tabpfn's loader built"
    assert first.model.models_ is not second.model.models_, "the list is per estimator, its module shared"

    saved_path = first.save()
    with open(os.path.join(saved_path, RealTabPFNv2Model.model_file_name), "rb") as f:
        pickled = pickle.load(f)
    assert not sw.attached(pickled)
    assert not graph.find_shared(pickled.model, payload.components()), "the pickle references the network"
    n_params = sum(p.numel() for p in first.model.models_[0].parameters())
    assert os.path.getsize(os.path.join(saved_path, RealTabPFNv2Model.model_file_name)) < n_params
    assert first.model.models_[0] is network, "the live model keeps its network"

    expected = first.predict_proba(X)
    registry.release()
    loaded = RealTabPFNv2Model.load(saved_path)
    assert loaded.model.models_[0] is not first.model.models_[0], "rebuilt from the checkpoint"
    assert loaded.get_info()["shared_weights"]["loaded_by"] == "load"
    np.testing.assert_allclose(loaded.predict_proba(X), expected, rtol=1e-5, atol=1e-6)
    assert loaded.model.models_[0] is _fit_tabpfn(tmp_path, "third", X, y).model.models_[0], "and shared again"


def test_tabpfn_fits_with_other_inference_settings_do_not_share_an_entry(tmp_path):
    """The loader resolves the inference configuration into the estimator, so a fit with other inference
    settings takes an entry of its own instead of the configuration an earlier fit resolved.
    """
    import numpy as np

    X, y = _classification_frame()
    default = _fit_tabpfn(tmp_path, "default", X, y)
    sharp = _fit_tabpfn(tmp_path, "sharp", X, y, hyperparameters={"softmax_temperature": 0.25})
    again = _fit_tabpfn(tmp_path, "again", X, y, hyperparameters={"softmax_temperature": 0.25})
    assert sharp._shared_state.key != default._shared_state.key
    assert again._shared_state.key == sharp._shared_state.key, "fits with the same settings still share"
    assert sharp.model.softmax_temperature_ == 0.25 != default.model.softmax_temperature_
    assert not np.allclose(sharp.predict_proba(X), default.predict_proba(X))


def test_tabpfn_registry_capacity_bounds_the_shared_networks(tmp_path):
    """The registry keeps its `capacity` most recently used networks; an evicted network lives on
    in the estimators that hold it, and capacity 0 shares nothing between fits.
    """
    import pandas as pd
    from sklearn.datasets import make_regression

    Xc, yc = _classification_frame()
    Xr, yr = make_regression(n_samples=60, n_features=4, random_state=0)
    Xr, yr = pd.DataFrame(Xr, columns=list("abcd")), pd.Series(yr)

    def fit(name, problem_type):
        if problem_type == "binary":
            return _fit_tabpfn(tmp_path, name, Xc, yc)
        return _fit_tabpfn(tmp_path, name, Xr, yr, problem_type="regression", eval_metric="rmse")

    registry.set_capacity(1)
    classifier = fit("classifier", "binary")
    regressor = fit("regressor", "regression")
    assert len(registry.report()["entries"]) == 1, "the regressor's network evicted the classifier's"
    assert not registry.contains(classifier._shared_state.key) and registry.contains(regressor._shared_state.key)
    assert classifier.model.models_[0] is not None, "the evicted network stays with its estimator"
    classifier.predict_proba(Xc)
    rebuilt = fit("classifier_again", "binary")
    assert rebuilt.model.models_[0] is not classifier.model.models_[0], "rebuilt after eviction"

    registry.set_capacity(2)
    regressor_again = fit("regressor_again", "regression")
    assert regressor_again.model.models_[0] is not regressor.model.models_[0], "the rebuild evicted it"
    assert fit("classifier_third", "binary").model.models_[0] is rebuilt.model.models_[0], "still registered"
    assert len(registry.report()["entries"]) == 2

    registry.set_capacity(0)
    registry.release()
    unshared = fit("unshared", "binary")
    assert unshared.model.models_[0] is not fit("unshared_too", "binary").model.models_[0]
    assert not registry.report()["entries"]


def test_tabpfn_share_weights_off_loads_and_pickles_its_own_network(tmp_path, monkeypatch):
    """`model_class_settings={"REALTABPFN-V2": {"share_weights": False}}` restores tabpfn's own load
    for that version only: no registry entry, the network in the pickle, a CPU round trip on save.
    """
    import os
    import pickle

    import numpy as np

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import RealTabPFNv25Model

    X, y = _classification_frame()
    monkeypatch.setattr(RealTabPFNv2Model, "_class_settings", None, raising=False)
    monkeypatch.setattr(RealTabPFNv2Model, "_class_settings_set", False, raising=False)
    RealTabPFNv2Model.set_class_settings(share_weights=False)
    assert RealTabPFNv25Model.get_class_settings().share_weights is True, "owned per registered version"
    assert TabPFNModel.get_class_settings().share_weights is True

    owned = _fit_tabpfn(tmp_path, "owned", X, y)
    assert owned._shared_state is None and owned.get_info()["shared_weights"] is None
    assert not registry.report()["entries"]
    assert isinstance(owned.model.model_path, str), "tabpfn's own load ran from the checkpoint path"
    expected = owned.predict_proba(X)
    saved_path = owned.save()
    with open(os.path.join(saved_path, RealTabPFNv2Model.model_file_name), "rb") as f:
        assert pickle.load(f).model.models_, "the network is in the pickle"
    loaded = RealTabPFNv2Model.load(saved_path)
    assert loaded.model.models_[0] is not owned.model.models_[0]
    np.testing.assert_allclose(loaded.predict_proba(X), expected, rtol=1e-5, atol=1e-6)
    assert not registry.report()["entries"]


@pytest.mark.parametrize(
    ("inputs", "mutates"),
    [
        ({}, False),
        ({"fit_mode": "fit_preprocessors"}, False),
        ({"fit_mode": "low_memory"}, False),
        ({"fit_mode": "fit_with_cache"}, True),
        ({"fit_mode": "batched"}, True),
        ({"inference_precision": torch.float64}, True),
        ({"inference_precision": "auto"}, False),
    ],
)
def test_tabpfn_mutates_network(inputs, mutates):
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import _mutates_network

    assert _mutates_network(inputs) is mutates


def test_tabpfn_low_memory_fits_share_one_network(tmp_path, monkeypatch):
    """`fit_mode="low_memory"` fits share the network, a weightless save reloads to the same predictions,
    and they equal an unshared low-memory fit's.
    """
    import numpy as np

    X, y = _classification_frame()
    low_memory = {"fit_mode": "low_memory"}
    first = _fit_tabpfn(tmp_path, "first", X, y, hyperparameters=low_memory)
    second = _fit_tabpfn(tmp_path, "second", X, y, hyperparameters=low_memory)
    assert first.model.fit_mode == "low_memory"
    assert first._shared_state is not None
    assert first.model.models_[0] is second.model.models_[0], "one network per checkpoint and process"
    expected = first.predict_proba(X)

    saved_path = first.save()
    registry.release()
    loaded = RealTabPFNv2Model.load(saved_path)
    assert loaded.get_info()["shared_weights"]["loaded_by"] == "load"
    np.testing.assert_allclose(loaded.predict_proba(X), expected, rtol=1e-5, atol=1e-6)

    monkeypatch.setattr(RealTabPFNv2Model, "_class_settings", None, raising=False)
    monkeypatch.setattr(RealTabPFNv2Model, "_class_settings_set", False, raising=False)
    RealTabPFNv2Model.set_class_settings(share_weights=False)
    owned = _fit_tabpfn(tmp_path, "owned", X, y, hyperparameters=low_memory)
    assert owned._shared_state is None
    np.testing.assert_allclose(owned.predict_proba(X), expected, rtol=1e-5, atol=1e-6)


def _cached_default_checkpoint(model_cls, problem_type):
    """The default checkpoint of `model_cls` for `problem_type` when tabpfn's cache already holds it, else None."""
    from pathlib import Path

    model = model_cls(problem_type=problem_type, eval_metric=None)
    path = model._resolve_model_path(hps={}, is_classification=problem_type in ("binary", "multiclass"))
    return path if path is not None and Path(path).is_file() else None


def _tensors_in(obj, depth: int = 10):
    """Every tensor reachable from ``obj`` through dicts, sequences, modules and object attributes."""
    seen = set()

    def walk(node, level):
        if level < 0 or node is None or id(node) in seen:
            return
        if isinstance(node, torch.Tensor):
            yield node
            return
        if isinstance(node, (str, bytes, int, float, bool, type)):
            return
        seen.add(id(node))
        if isinstance(node, torch.nn.Module):
            yield from node.parameters()
            yield from node.buffers()
        elif isinstance(node, dict):
            for value in node.values():
                yield from walk(value, level - 1)
        elif isinstance(node, (list, tuple, set)):
            for value in node:
                yield from walk(value, level - 1)
        elif hasattr(node, "__dict__"):
            for value in vars(node).values():
                yield from walk(value, level - 1)

    yield from walk(obj, depth)


# The ids avoid "regression" in the test name: the conftest reserves that word for `--runregression` tests.
@pytest.mark.parametrize("problem_type", ["binary", "regression"], ids=["classifier", "regressor"])
def test_tabpfn_shared_and_unshared_fits_predict_the_same(problem_type, tmp_path, monkeypatch):
    """A fit on the registry's specs equals a fit on the checkpoint path, for TabPFN-2.5 on the CPU.

    The shared child's pickle holds no network tensor and a reloaded child references the registry
    module again. Runs only against checkpoints already in tabpfn's cache and never downloads one
    (`ag.fetch_pretrained_weights=False`), so the CI job without the 2.5 checkpoints skips it.
    """
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_regression

    from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import RealTabPFNv25Model

    if _cached_default_checkpoint(RealTabPFNv25Model, problem_type) is None:
        pytest.skip(f"the TabPFN-2.5 {problem_type} checkpoint is not in tabpfn's cache; not downloading it here")
    if problem_type == "binary":
        X, y = _classification_frame()
        eval_metric = "log_loss"
    else:
        X, y = make_regression(n_samples=60, n_features=4, random_state=0)
        X, y = pd.DataFrame(X, columns=list("abcd")), pd.Series(y)
        eval_metric = "rmse"
    X_test = X.iloc[:20]
    hyperparameters = {**toy_model_params, "ag.fetch_pretrained_weights": False}
    monkeypatch.setattr(RealTabPFNv25Model, "_class_settings", None, raising=False)
    monkeypatch.setattr(RealTabPFNv25Model, "_class_settings_set", False, raising=False)

    def fit(name, share):
        RealTabPFNv25Model.set_class_settings(share_weights=share)
        model = RealTabPFNv25Model(
            path=str(tmp_path / name),
            name=name,
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=dict(hyperparameters),
        )
        model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
        return model

    shared = fit("shared", share=True)
    plain = fit("plain", share=False)
    assert shared._shared_state is not None and plain._shared_state is None
    np.testing.assert_allclose(shared.predict_proba(X_test), plain.predict_proba(X_test), rtol=1e-5, atol=1e-6)
    payload = registry.peek(shared._shared_state.key).value
    network = payload.attributes["models_"][0]
    assert shared.model.models_[0] is network
    assert shared.get_info()["shared_weights"]["variant"] == (
        "TabPFNClassifier._initialize_model_variables"
        if problem_type == "binary"
        else "TabPFNRegressor._initialize_model_variables"
    )

    modules = payload.components()[1:]
    payload_tensors = {id(t) for module in modules for t in (*module.parameters(), *module.buffers())}
    state = shared.__getstate__()
    assert state["_shared_state"].paths, "the pickle records where the network sat"
    for tensor in _tensors_in(state):
        assert tensor.device.type == "cpu"
        assert id(tensor) not in payload_tensors, "a network tensor is in the pickle"
    assert shared.model.models_[0] is network, "the live estimator keeps its network"

    reloaded = pickle.loads(pickle.dumps(shared))
    assert not sw.attached(reloaded)
    sw.ensure_network(reloaded)
    assert graph.find_shared(reloaded.model, modules), "the reloaded child does not reference the registry module"
    assert reloaded.model.models_[0] is network
    np.testing.assert_allclose(reloaded.predict_proba(X_test), shared.predict_proba(X_test))

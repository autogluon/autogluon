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
    this is skipped there and `test_tabpfn_set_device_repoints_models_at_engine_cache` is what
    actually guards the mechanism in CI.

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


class _StubModule:
    """Stands in for a loaded checkpoint: records whether it was moved."""

    def __init__(self, name: str):
        self.name = name
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _StubCache:
    """`_PerDeviceModelCache`: holds one copy per device and hands them out after a move."""

    def __init__(self, model, device="cpu"):
        self._models = {device: model}

    def get(self, device):
        return self._models[device]

    def get_devices(self):
        return list(self._models)


class _StubEstimator:
    def __init__(self, models_, executor_=None):
        self.models_ = models_
        self.moved_to = None
        if executor_ is not None:
            self.executor_ = executor_

    def to(self, device):
        self.moved_to = device


def _stub_tabpfn_model(estimator):
    model = RealTabPFNv2Model.__new__(RealTabPFNv2Model)
    model.model = estimator
    return model


def test_tabpfn_set_device_repoints_models_at_engine_cache():
    """The multi-device divergence, reproduced without needing two GPUs.

    `estimator.to()` moves the engine's per-device caches and leaves `models_` pointing at whatever
    it pointed at before. With several devices the surviving cache copy is a *different* object from
    `models_[i]`, so the artifact ends up holding a checkpoint on the old device and a second copy
    of the same weights. Stubbing the cache reproduces exactly that, so this guards the mechanism on
    any machine -- including the single-GPU CI runner, where the end-to-end test cannot bite.
    """
    stale, fresh = _StubModule("stale"), _StubModule("fresh")
    estimator = _StubEstimator(models_=[stale], executor_=_StubEstimator([], None))
    estimator.executor_.model_caches = [_StubCache(fresh)]

    _stub_tabpfn_model(estimator).set_device("cpu")

    assert estimator.moved_to == "cpu", "the estimator itself must still be moved"
    assert estimator.models_ == [fresh], "models_ must be re-pointed at the engine's copy"
    assert stale.moved_to is None, "the stale copy is dropped, not moved and kept alongside"


def test_tabpfn_set_device_falls_back_when_the_engine_shape_is_unfamiliar():
    """If tabpfn stops exposing the caches as expected, move `models_` directly rather than break.

    Keeps the artifact CPU-loadable (the correctness half) even if the de-duplication half stops
    applying, instead of failing or silently leaving CUDA tensors behind.
    """
    inner = _StubModule("inner")
    estimator = _StubEstimator(models_=[inner])  # no executor_ at all

    _stub_tabpfn_model(estimator).set_device("cpu")

    assert estimator.moved_to == "cpu"
    assert inner.moved_to == "cpu", "fell back to moving the checkpoint itself"

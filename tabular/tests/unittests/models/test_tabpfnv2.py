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

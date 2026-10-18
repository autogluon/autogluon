from autogluon.tabular.models.tabdpt.tabdpt_turbo_model import TabDPTTurboModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_tabdpt_turbo():
    model_cls = TabDPTTurboModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabDPT returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_no_feature_cap():
    """TabDPT-Turbo pads or projects every input to a fixed width, so a feature cap has no basis.

    `tabdpt` sets `max_features = model.num_features` (128 for the pinned checkpoint) — an
    architectural bound, since the encoder is `nn.Linear(num_features, ninp)`. Wider inputs are
    PCA-projected down to it, narrower ones padded up. Measured peak VRAM is 1.23 GB from 200
    features through 5000, and the GPU estimate has no feature term for the same reason.
    """
    aux = TabDPTTurboModel()._get_default_auxiliary_params()
    assert aux["max_features"] is None
    # The base TabDPT keeps its cap: only the Turbo checkpoint's width was measured.
    from autogluon.tabular.models.tabdpt.tabdpt_model import TabDPTModel

    assert TabDPTModel()._get_default_auxiliary_params()["max_features"] == 2500

    # The estimate is feature-independent, matching the fixed-width encoding.
    import numpy as np
    import pandas as pd

    narrow = pd.DataFrame(np.zeros((1000, 10), dtype="float32"))
    wide = pd.DataFrame(np.zeros((1000, 5_000), dtype="float32"))
    assert TabDPTTurboModel._estimate_gpu_memory_usage_static(
        X=narrow
    ) == TabDPTTurboModel._estimate_gpu_memory_usage_static(X=wide)

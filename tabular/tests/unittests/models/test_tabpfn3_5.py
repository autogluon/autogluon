import pytest

from autogluon.tabular.models.tabpfnv2.tabpfn3_5_model import TabPFN35Model
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 1}


@pytest.mark.skip(
    reason="TabPFN-3.5 model weights are not available publicly without accepting a license agreement; "
    "run manually on a machine with the checkpoint in the tabpfn cache."
)
def test_tabpfn3_5():
    FitHelper.verify_model(
        model_cls=TabPFN35Model,
        model_hyperparameters=toy_model_params,
        verify_load_wo_cuda=True,
        # TabPFN returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression", "quantile"])
def test_tabpfn3_5_uses_the_multitask_checkpoint_for_every_problem_type(problem_type):
    model = TabPFN35Model(problem_type=problem_type, eval_metric=None)
    assert model._default_model_map()[problem_type] == "tabpfn-v3.5-20260909.safetensors"


def test_tabpfn3_5_checkpoint_name_resolves_to_the_v3_5_model_version():
    """tabpfn picks the download source from the checkpoint's file name."""
    pytest.importorskip("tabpfn")
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import resolve_model_version

    assert resolve_model_version(TabPFN35Model.default_classification_model) == ModelVersion.V3_5
    assert resolve_model_version(TabPFN35Model.default_regression_model) == ModelVersion.V3_5


def test_tabpfn3_5_limits():
    aux = TabPFN35Model()._get_default_auxiliary_params()
    assert aux["max_rows"] == 1_000_000
    assert aux["max_classes"] == 160
    # None rather than absent: an absent key would inherit the 2.5 base's cap of 2000.
    assert aux["max_features"] is None


def test_tabpfn3_5_auto_max_batch_size_resolution():
    assert TabPFN35Model._resolve_auto_max_batch_size(n_train=500) == 100_500
    assert TabPFN35Model._resolve_auto_max_batch_size(n_train=950_000) == 1_000_000, "capped at 1M"


def test_tabpfn3_5_gpu_memory_estimate_shape():
    import numpy as np
    import pandas as pd

    X = pd.DataFrame(np.zeros((100_000, 20)))
    classification = TabPFN35Model._estimate_gpu_memory_usage_static(X=X, problem_type="binary")
    regression = TabPFN35Model._estimate_gpu_memory_usage_static(X=X, problem_type="regression")
    # Regression's distributional output makes each prediction row far more expensive.
    assert regression > classification
    # A smaller prediction batch lowers the estimate.
    chunked = TabPFN35Model._estimate_gpu_memory_usage_static(
        X=X, problem_type="regression", hyperparameters={"ag.max_batch_size": 10_000}
    )
    assert chunked < regression
    # The feature cost saturates where internal subsampling caps it.
    wide = TabPFN35Model._estimate_gpu_memory_usage_static(
        X=pd.DataFrame(np.zeros((100, 1000))), problem_type="binary"
    )
    wider = TabPFN35Model._estimate_gpu_memory_usage_static(
        X=pd.DataFrame(np.zeros((100, 5000))), problem_type="binary"
    )
    assert wide == wider

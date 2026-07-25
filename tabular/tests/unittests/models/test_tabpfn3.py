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

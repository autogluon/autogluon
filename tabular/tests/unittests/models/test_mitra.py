import pytest

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.mitra.mitra_model import MitraModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"fine_tune_steps": 3}


@pytest.mark.gpu
def test_mitra():
    if ResourceManager.get_gpu_count_torch() == 0:
        # Skip test if no GPU available
        pytest.skip("Skip, no GPU available.")

    model_cls = MitraModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)

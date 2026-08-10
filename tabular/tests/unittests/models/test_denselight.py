from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.tabular.models.denselight.denselight_model import DenseLightModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_epochs": 2, "patience": 2, "batch_size": 32}


def test_denselight():
    model_cls = DenseLightModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
    )


def test_denselight_auxiliary_params_match_custom_model_tutorial():
    """valid_raw_types should match the custom-model tutorial style (int/float/category/bool)."""
    model = DenseLightModel(problem_type="binary", eval_metric=None)
    aux = model._get_default_auxiliary_params()
    assert set(aux["valid_raw_types"]) == {R_BOOL, R_INT, R_FLOAT, R_CATEGORY}

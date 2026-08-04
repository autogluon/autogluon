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

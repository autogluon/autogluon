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
    )

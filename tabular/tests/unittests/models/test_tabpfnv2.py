from autogluon.tabular.models.tabpfnv2.tabpfnv2_model import TabPFNV2Model
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_tabpfnv2():
    model_cls = TabPFNV2Model
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from autogluon.tabular.models.mitra.mitra_model import MitraModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_mitra():
    model_cls = MitraModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
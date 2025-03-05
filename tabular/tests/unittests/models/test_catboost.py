import sys

import pytest

from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"iterations": 10}


@pytest.mark.skipif(sys.version_info >= (3, 11) and sys.platform == "darwin", reason="catboost has no wheel for py311 darwin")
def test_catboost():
    model_cls = CatBoostModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


from autogluon.tabular.models.tabprep.prep_lgb_model import PrepLGBModel
from autogluon.tabular.testing import FitHelper


def test_lightgbm():
    model_cls = PrepLGBModel
    model_hyperparameters = {
        'ag.prep_params': [
            [
                ['ArithmeticFeatureGenerator', {}],
                [
                    ['CategoricalInteractionFeatureGenerator', {"passthrough": True}],
                    ['OOFTargetEncodingFeatureGenerator', {}],
                ],
            ],
        ],
        'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]},
    }
    """Additionally tests that all metrics work"""
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters, extra_metrics=True)

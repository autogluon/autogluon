import pytest

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabprep.prep_lgb_model import PrepLGBModel
from autogluon.tabular.testing import FitHelper
from autogluon.tabular.trainer import abstract_trainer as abstract_trainer_module


PREP_HYPERPARAMETERS = {
    "ag.prep_params": [
        [
            ["ArithmeticFeatureGenerator", {}],
            [
                ["CategoricalInteractionFeatureGenerator", {"passthrough": True}],
                ["OOFTargetEncodingFeatureGenerator", {}],
            ],
        ],
    ],
    "ag.prep_params.passthrough_types": {"invalid_raw_types": ["category", "object"]},
}


def test_lightgbm():
    model_cls = PrepLGBModel
    model_hyperparameters = PREP_HYPERPARAMETERS.copy()
    """Additionally tests that all metrics work"""
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters, extra_metrics=True)


def test_feature_importance_temporarily_blocked_when_pipeline_uses_lightgbm_prep(
    tmp_path,
    monkeypatch,
):
    train_data, test_data, dataset_info = FitHelper.load_dataset("toy_binary_10")

    predictor = TabularPredictor(
        label=dataset_info["label"],
        problem_type=dataset_info["problem_type"],
        path=str(tmp_path / "AutogluonOutput"),
    ).fit(
        train_data,
        hyperparameters={
            "GBM_PREP": [{**PREP_HYPERPARAMETERS, "num_boost_round": 10}],
            "GBM": [{"num_boost_round": 10}],
        },
        num_bag_folds=2,
        num_stack_levels=1,
        fit_weighted_ensemble=True,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    warning_messages = []
    monkeypatch.setattr(
        abstract_trainer_module.logger,
        "warning",
        lambda message: warning_messages.append(message),
    )

    stacked_model = next(
        model_name for model_name in predictor.model_names() if predictor._trainer.get_model_level(model_name) > 1
    )
    with pytest.raises(NotImplementedError, match="Temporary safeguard"):
        predictor.feature_importance(data=test_data, model=stacked_model, num_shuffle_sets=1)
    assert len(warning_messages) == 1
    assert "excluded_model_types=['LGBMPrep']" in warning_messages[0]
    assert "remove `GBM_PREP` from the requested hyperparameters" in warning_messages[0]

    prep_model = next(model_name for model_name in predictor.model_names() if model_name.startswith("LightGBMPrep"))
    warning_messages.clear()
    with pytest.raises(NotImplementedError, match=prep_model):
        predictor.feature_importance(data=test_data, model=prep_model, num_shuffle_sets=1)
    assert len(warning_messages) == 1
    assert prep_model in warning_messages[0]


def test_feature_importance_allowed_when_lightgbm_prep_has_no_prep_params(tmp_path):
    train_data, test_data, dataset_info = FitHelper.load_dataset("toy_binary_10")

    predictor = TabularPredictor(
        label=dataset_info["label"],
        problem_type=dataset_info["problem_type"],
        path=str(tmp_path / "AutogluonOutput"),
    ).fit(
        train_data,
        hyperparameters={"GBM_PREP": [{"ag.prep_params": [], "num_boost_round": 10}]},
        fit_weighted_ensemble=False,
    )

    prep_model = next(model_name for model_name in predictor.model_names() if model_name.startswith("LightGBMPrep"))
    fi_df = predictor.feature_importance(data=test_data, model=prep_model, num_shuffle_sets=1, subsample_size=10)

    assert "importance" in fi_df.columns

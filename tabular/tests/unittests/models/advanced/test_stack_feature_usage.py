"""
Unit tests to ensure correctness of internal stacking logic.
"""

import shutil

from autogluon.common.features.types import S_STACK
from autogluon.core.models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from autogluon.tabular.predictor import TabularPredictor
from autogluon.tabular.testing import FitHelper


def test_stack_feature_usage_binary():
    """Tests that stacker models use base model predictions as features correctly for binary"""
    dataset_name = "adult"
    expected_ancestors = {"LightGBM_BAG_L1", "Dummy_BAG_L1"}
    _fit_predictor_stack_feature_usage(
        dataset_name=dataset_name,
        max_base_models_per_type=1,
        max_base_models=2,
        sample_size=100,
        expected_ancestors=expected_ancestors,
    )


def test_stack_feature_usage_multiclass():
    """Tests that stacker models use base model predictions as features correctly for multiclass"""
    dataset_name = "covertype_small"
    expected_ancestors = {"LightGBM_BAG_L1", "KNeighbors_BAG_L1"}
    _fit_predictor_stack_feature_usage(
        dataset_name=dataset_name,
        max_base_models_per_type=1,
        max_base_models=2,
        sample_size=100,
        expected_ancestors=expected_ancestors,
    )


def test_stack_feature_usage_regression():
    """Tests that stacker models use base model predictions as features correctly for regression"""
    dataset_name = "ames"
    expected_ancestors = {"LightGBM_BAG_L1", "KNeighbors_BAG_L1"}
    _fit_predictor_stack_feature_usage(
        dataset_name=dataset_name,
        max_base_models_per_type=1,
        max_base_models=2,
        sample_size=100,
        expected_ancestors=expected_ancestors,
    )


def test_stack_feature_usage_binary_all():
    """Tests that stacker models use base model predictions as features correctly for binary"""
    dataset_name = "adult"
    expected_ancestors = {"LightGBM_BAG_L1", "LightGBM_2_BAG_L1", "KNeighbors_BAG_L1", "Dummy_BAG_L1"}
    _fit_predictor_stack_feature_usage(
        dataset_name=dataset_name,
        max_base_models_per_type=0,  # uncapped
        max_base_models=0,  # uncapped
        sample_size=100,
        expected_ancestors=expected_ancestors,
    )


def test_stack_feature_usage_binary_only_max_models():
    """Tests that stacker models use base model predictions as features correctly for binary"""
    dataset_name = "adult"
    expected_ancestors = {"LightGBM_BAG_L1", "LightGBM_2_BAG_L1"}
    _fit_predictor_stack_feature_usage(
        dataset_name=dataset_name,
        max_base_models_per_type=0,  # uncapped
        max_base_models=2,
        sample_size=100,
        expected_ancestors=expected_ancestors,
    )


def _fit_predictor_stack_feature_usage(
    dataset_name: str,
    max_base_models_per_type: int,
    max_base_models: int,
    sample_size: int,
    expected_ancestors: set,
):
    """Tests that stacker models use base model predictions as features correctly"""
    fit_args = dict(
        hyperparameters={
            "GBM": [
                {"num_iterations": 200},
                {"num_iterations": 50},
            ],
            "KNN": {},
            "DUMMY": {},
        },
        ag_args_ensemble={
            "max_base_models_per_type": max_base_models_per_type,
            "max_base_models": max_base_models,
            "fold_fitting_strategy": "sequential_local",
        },
        num_bag_folds=2,
        num_stack_levels=1,
        fit_weighted_ensemble=False,
    )
    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        refit_full=False,
        sample_size=sample_size,
        expected_model_count=7,
        delete_directory=False,
    )
    _assert_stack_features(predictor=predictor, expected_ancestors=expected_ancestors)
    shutil.rmtree(predictor.path, ignore_errors=True)


def _assert_stack_features(predictor: TabularPredictor, expected_ancestors: set):
    """
    Verifies that stack features are correctly set and used
    """
    leaderboard = predictor.leaderboard(extra_info=True)
    leaderboard_l2 = leaderboard[leaderboard["stack_level"] == 2]
    leaderboard_l2 = leaderboard_l2.set_index("model")
    for m in leaderboard_l2.index:
        ancestors = set(leaderboard_l2.loc[m, "ancestors"])
        assert expected_ancestors == ancestors
        model = predictor._trainer.load_model(model_name=m)
        assert isinstance(model, StackerEnsembleModel)
        assert expected_ancestors == set(model.base_model_names)
        assert expected_ancestors == set(predictor._trainer.get_minimum_model_set(model=m, include_self=False))
        stack_columns = model.stack_columns
        stack_columns_fm = model.feature_metadata.get_features(required_special_types=[S_STACK])
        assert set(stack_columns) == set(stack_columns_fm)
        features = model.feature_metadata.get_features()
        assert set(features) == set(model.features)
        # Asserts that child models use the same features as parent models
        for child_model_name in model.models:
            child_model = model.load_child(model=child_model_name)
            child_stack_columns_fm = child_model.feature_metadata.get_features(required_special_types=[S_STACK])
            assert set(stack_columns) == set(child_stack_columns_fm)
            assert set(features) == set(child_model.features)
            assert model.feature_metadata == child_model.feature_metadata

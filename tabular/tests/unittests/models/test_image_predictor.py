import pytest
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.tabular import TabularPredictor


@pytest.mark.gpu
def test_image_predictor_multiclass(fit_helper):
    # Test ImagePredictor with Torch
    from autogluon.vision import ImageDataset
    train_data, _, test_data = ImageDataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    feature_metadata = FeatureMetadata.from_df(train_data).add_special_types({'image': ['image_path']})
    predictor = TabularPredictor(label='label').fit(
        train_data=train_data,
        hyperparameters={'AG_IMAGE_NN': {'epochs': 2, 'model': 'resnet18'}},
        feature_metadata=feature_metadata
    )
    leaderboard = predictor.leaderboard(test_data)
    assert len(leaderboard) > 0


@pytest.mark.gpu
def test_image_predictor_regression(fit_helper):
    # Test ImagePredictor with Torch
    from autogluon.vision import ImageDataset
    train_data, _, test_data = ImageDataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    feature_metadata = FeatureMetadata.from_df(train_data).add_special_types({'image': ['image_path']})
    predictor = TabularPredictor(label='label', problem_type='regression').fit(
        train_data=train_data,
        hyperparameters={'AG_IMAGE_NN': {'epochs': 2, 'model': 'resnet18'}},
        feature_metadata=feature_metadata
    )
    leaderboard = predictor.leaderboard(test_data)
    assert len(leaderboard) > 0

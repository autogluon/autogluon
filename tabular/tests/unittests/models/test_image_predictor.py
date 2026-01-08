import pytest

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular import TabularPredictor


@pytest.mark.gpu
def test_image_predictor_multiclass():
    if ResourceManager.get_gpu_count_torch() == 0:
        # Skip test if no GPU available
        pytest.skip("Skip, no GPU available.")
    from autogluon.multimodal.utils.misc import shopee_dataset

    download_dir = "./automm_shopee_data_multiclass"
    train_data, test_data = shopee_dataset(download_dir)
    train_data = train_data.sample(100, random_state=0)  # Subsample for faster test
    feature_metadata = FeatureMetadata.from_df(train_data).add_special_types({"image": ["image_path"]})
    predictor = TabularPredictor(label="label").fit(
        train_data=train_data,
        hyperparameters={"AG_IMAGE_NN": {"optim.max_epochs": 1}},
        feature_metadata=feature_metadata,
    )
    leaderboard = predictor.leaderboard(test_data)
    assert len(leaderboard) > 0


@pytest.mark.gpu
def test_image_predictor_regression():
    if ResourceManager.get_gpu_count_torch() == 0:
        # Skip test if no GPU available
        pytest.skip("Skip, no GPU available.")
    from autogluon.multimodal.utils.misc import shopee_dataset

    download_dir = "./automm_shopee_data_regression"
    train_data, test_data = shopee_dataset(download_dir)
    train_data = train_data.sample(100, random_state=0)  # Subsample for faster test
    feature_metadata = FeatureMetadata.from_df(train_data).add_special_types({"image": ["image_path"]})
    predictor = TabularPredictor(label="label", problem_type="regression").fit(
        train_data=train_data,
        hyperparameters={"AG_IMAGE_NN": {"optim.max_epochs": 1}},
        feature_metadata=feature_metadata,
    )
    leaderboard = predictor.leaderboard(test_data)
    assert len(leaderboard) > 0

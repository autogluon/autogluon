import pytest
import tempfile
import zipfile

from autogluon.cloud import MultiModalCloudPredictor

from utils import (
    _prepare_data,
    _test_functionality
)


@pytest.mark.cloud
def test_multimodal_text_only():
    train_data = 'text_train.csv'
    tune_data = 'text_tune.csv'
    test_data = 'text_test.csv'
    with tempfile.TemporaryDirectory() as root:
        _prepare_data(train_data, tune_data, test_data)
        time_limit = 60

        predictor_init_args = dict(
            label='label',
            eval_metric='acc'
        )
        predictor_fit_args = dict(
            train_data=train_data,
            tuning_data=tune_data,
            time_limit=time_limit
        )
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-text',
            local_output_path='test_multimodal_text_cloud_predictor'
        )
        cloud_predictor_no_train = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-text-no-train',
            local_output_path='test_multimodal_text_cloud_predictor_no_train'
        )
        _test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_instance_type='ml.g4dn.2xlarge'
        )


@pytest.mark.cloud
def test_multimodal_image_only():
    train_data = 'image_train_relative.csv'
    train_image = 'shopee-iet.zip'
    test_data = 'test_images/BabyPants_1035.jpg'
    with tempfile.TemporaryDirectory() as root:
        _prepare_data(train_data, train_image, test_data)
        test_data = 'BabyPants_1035.jpg'
        time_limit = 60

        predictor_init_args = dict(
            label='label',
            eval_metric='acc'
        )
        predictor_fit_args = dict(
            train_data=train_data,
            time_limit=time_limit
        )
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-image',
            local_output_path='test_multimodal_image_cloud_predictor'
        )
        cloud_predictor_no_train = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-image-no-train',
            local_output_path='test_multimodal_image_cloud_predictor_no_train'
        )
        _test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            image_path='shopee-iet.zip',
            fit_instance_type='ml.g4dn.2xlarge',
            predict_kwargs=dict(image_modality_only=True)
        )


@pytest.mark.cloud
def test_multimodal_tabular_text():
    train_data = 'tabular_text_train.csv'
    test_data = 'tabular_text_test.csv'
    with tempfile.TemporaryDirectory() as root:
        _prepare_data(train_data, test_data)
        time_limit = 60

        predictor_init_args = dict(
            label='Sentiment',
        )
        predictor_fit_args = dict(
            train_data=train_data,
            time_limit=time_limit
        )
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-tabular-text',
            local_output_path='test_multimodal_tabular_text_cloud_predictor'
        )
        cloud_predictor_no_train = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-tabular-text-no-train',
            local_output_path='test_multimodal_tabular_text_cloud_predictor_no_train'
        )
        _test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_instance_type='ml.g4dn.2xlarge',
        )


@pytest.mark.cloud
def test_multimodal_tabular_text_image():
    train_data = 'tabular_text_image_train.csv'
    test_data = 'tabular_text_image_test.csv'
    images = 'tabular_text_image_images.zip'
    with tempfile.TemporaryDirectory() as root:
        _prepare_data(train_data, test_data, images)
        with zipfile.ZipFile(images, 'r') as zip_ref:
            zip_ref.extractall('.')
        time_limit = 120

        predictor_init_args = dict(
            label='AdoptionSpeed',
        )
        predictor_fit_args = dict(
            train_data=train_data,
            time_limit=time_limit
        )
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-tabular-text-image',
            local_output_path='test_multimodal_tabular_text_image_cloud_predictor'
        )
        cloud_predictor_no_train = MultiModalCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-multimodal-tabular-text-image-no-train',
            local_output_path='test_multimodal_tabular_text_image_cloud_predictor_no_train'
        )
        _test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            image_path='tabular_text_image_images.zip',
            fit_instance_type='ml.g4dn.2xlarge',
            predict_real_time_kwargs=dict(test_data_image_column='Images'),
            predict_kwargs=dict(test_data_image_column='Images')
        )

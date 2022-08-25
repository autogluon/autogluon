import tempfile

from autogluon.cloud import ImageCloudPredictor

from ..utils import (
    _prepare_data,
    _test_functionality
)


def test_image():
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
        cloud_predictor = ImageCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-image',
            local_output_path='test_image_cloud_predictor'
        )
        cloud_predictor_no_train = ImageCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-image-no-train',
            local_output_path='test_image_cloud_predictor_no_train'
        )
        _test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            image_path='shopee-iet.zip',
            fit_instance_type='ml.g4dn.2xlarge'
        )

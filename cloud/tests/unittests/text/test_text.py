import tempfile

from autogluon.cloud import TextCloudPredictor
from autogluon.cloud import MultiModalCloudPredictor


def test_text(test_helper):
    train_data = 'text_train.csv'
    tune_data = 'text_tune.csv'
    test_data = 'text_test.csv'
    with tempfile.TemporaryDirectory() as root:
        test_helper.prepare_data(train_data, tune_data, test_data)
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
        cloud_predictor = TextCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-text',
            local_output_path='test_text_cloud_predictor'
        )
        cloud_predictor_no_train = TextCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-text-no-train',
            local_output_path='test_text_cloud_predictor_no_train'
        )
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_instance_type='ml.g4dn.2xlarge',
            fit_kwargs=dict(custom_image_uri=test_helper.gpu_training_image),
            deploy_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
            predict_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
        )


def test_multimodal_text_only(test_helper):
    train_data = 'text_train.csv'
    tune_data = 'text_tune.csv'
    test_data = 'text_test.csv'
    with tempfile.TemporaryDirectory() as root:
        test_helper.prepare_data(train_data, tune_data, test_data)
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
            cloud_output_path='s3://autogluon-cloud-ci/test-multimodal-text',
            local_output_path='test_multimodal_text_cloud_predictor'
        )
        cloud_predictor_no_train = MultiModalCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-multimodal-text-no-train',
            local_output_path='test_multimodal_text_cloud_predictor_no_train'
        )
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_instance_type='ml.g4dn.2xlarge',
            fit_kwargs=dict(custom_image_uri=test_helper.gpu_training_image),
            deploy_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
            predict_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
        )

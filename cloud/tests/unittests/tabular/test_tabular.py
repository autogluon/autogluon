import tempfile
import zipfile

from autogluon.cloud import TabularCloudPredictor


def test_tabular(test_helper):
    train_data = 'tabular_train.csv'
    tune_data = 'tabular_tune.csv'
    test_data = 'tabular_test.csv'
    with tempfile.TemporaryDirectory() as root:
        test_helper.prepare_data(train_data, tune_data, test_data)
        time_limit = 60

        predictor_init_args = dict(
            label='class',
            eval_metric='roc_auc'
        )
        predictor_fit_args = dict(
            train_data=train_data,
            tuning_data=tune_data,
            time_limit=time_limit,
        )
        cloud_predictor = TabularCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-tabular',
            local_output_path='test_tabular_cloud_predictor'
        )
        cloud_predictor_no_train = TabularCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-tabular-no-train',
            local_output_path='test_tabular_cloud_predictor_no_train'
        )
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data
        )


def test_tabular_tabular_text_image(test_helper):
    train_data = 'tabular_text_image_train.csv'
    test_data = 'tabular_text_image_test.csv'
    images = 'tabular_text_image_images.zip'
    with tempfile.TemporaryDirectory() as root:
        test_helper.prepare_data(train_data, test_data, images)
        with zipfile.ZipFile(images, 'r') as zip_ref:
            zip_ref.extractall('.')
        time_limit = 600

        predictor_init_args = dict(
            label='AdoptionSpeed',
        )
        predictor_fit_args = dict(
            train_data=train_data,
            time_limit=time_limit,
            hyperparameters={
                'XGB': {},
                'AG_TEXT_NN': {'presets': 'medium_quality_faster_train'},
                'AG_IMAGE_NN': {},
            }
        )
        cloud_predictor = TabularCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-tabular-tabular-text-image',
            local_output_path='test_tabular_tabular_text_image_cloud_predictor'
        )
        cloud_predictor_no_train = TabularCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-tabular-tabular-text-image-no-train',
            local_output_path='test_tabular_tabular_text_image_cloud_predictor_no_train'
        )
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            image_path='tabular_text_image_images.zip',
            fit_instance_type='ml.g4dn.2xlarge',
            fit_kwargs=dict(image_column='Images'),
            deploy_kwargs=dict(
                instance_type='ml.g4dn.2xlarge',
            ),
            predict_real_time_kwargs=dict(
                test_data_image_column='Images',
            ),
            predict_kwargs=dict(
                instance_type='ml.g4dn.2xlarge',
                test_data_image_column='Images',
            ),
            skip_predict=True  # TODO: remove this after autogluon 0.6 release. Currently, some issues cause some module's batch inference to fail
        )


def test_tabular_tabular_text(test_helper):
    train_data = 'tabular_text_train.csv'
    test_data = 'tabular_text_test.csv'
    with tempfile.TemporaryDirectory() as root:
        test_helper.prepare_data(train_data, test_data)
        time_limit = 120

        predictor_init_args = dict(
            label='Sentiment',
        )
        predictor_fit_args = dict(
            train_data=train_data,
            time_limit=time_limit,
            hyperparameters={
                'XGB': {},
                'AG_TEXT_NN': {'presets': 'medium_quality_faster_train'},
            }
        )
        cloud_predictor = TabularCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-tabular-tabular-text',
            local_output_path='test_tabular_tabular_text_cloud_predictor'
        )
        cloud_predictor_no_train = TabularCloudPredictor(
            cloud_output_path='s3://autogluon-cloud-ci/test-tabular-tabular-text-no-train',
            local_output_path='test_tabular_tabular_text_cloud_predictor_no_train'
        )
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_instance_type='ml.g4dn.2xlarge',
            deploy_kwargs=dict(
                instance_type='ml.g4dn.2xlarge',
            ),
            predict_kwargs=dict(
                instance_type='ml.g4dn.2xlarge',
            ),
            skip_predict=True  # TODO: remove this after autogluon 0.6 release. Currently, some issues cause some module's batch inference to fail
        )

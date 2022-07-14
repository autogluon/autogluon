import boto3
import pytest
import tempfile

from autogluon.cloud import TabularCloudPredictor, TextCloudPredictor


def _prepare_data(train_data, tune_data, test_data):
    s3 = boto3.client('s3')
    s3.download_file('autogluon-cloud', train_data, train_data)
    s3.download_file('autogluon-cloud', tune_data, tune_data)
    s3.download_file('autogluon-cloud', test_data, test_data)


def _test_endpoint(cloud_predictor, test_data):
    try:
        cloud_predictor.predict_real_time(test_data)
    except Exception as e:
        cloud_predictor.cleanup_deployment()  # cleanup endpoint if test failed
        raise e


def _test_functionality(cloud_predictor, predictor_init_args, predictor_fit_args, cloud_predictor_no_train, test_data, fit_instance_type=None):
    if not fit_instance_type:
        fit_instance_type = 'ml.m5.2xlarge'
    cloud_predictor.fit(
        predictor_init_args,
        predictor_fit_args,
        instance_type=fit_instance_type
    )
    info = cloud_predictor.info()
    assert info['local_output_path'] is not None
    assert info['cloud_output_path'] is not None
    assert info['fit_job']['name'] is not None
    assert info['fit_job']['status'] == 'Completed'

    cloud_predictor.deploy()
    _test_endpoint(cloud_predictor, test_data)
    detached_endpoint = cloud_predictor.detach_endpoint()
    cloud_predictor.attach_endpoint(detached_endpoint)
    _test_endpoint(cloud_predictor, test_data)
    cloud_predictor.save()
    cloud_predictor = cloud_predictor.__class__.load(cloud_predictor.local_output_path)
    _test_endpoint(cloud_predictor, test_data)
    cloud_predictor.cleanup_deployment()

    info = cloud_predictor.info()
    assert info['local_output_path'] is not None
    assert info['cloud_output_path'] is not None
    assert info['fit_job']['name'] is not None
    assert info['fit_job']['status'] == 'Completed'

    cloud_predictor.predict(test_data)
    info = cloud_predictor.info()
    assert info['recent_transform_job']['status'] == 'Completed'

    # Test deploy with already trained predictor
    trained_predictor_path = cloud_predictor._fit_job.get_output_path()
    cloud_predictor_no_train.deploy(predictor_path=trained_predictor_path)
    _test_endpoint(cloud_predictor_no_train, test_data)
    cloud_predictor_no_train.cleanup_deployment()
    cloud_predictor_no_train.predict(test_data, predictor_path=trained_predictor_path)
    info = cloud_predictor_no_train.info()
    assert info['recent_transform_job']['status'] == 'Completed'


@pytest.mark.cloud
def test_tabular():
    train_data = 'tabular_train.csv'
    tune_data = 'tabular_tune.csv'
    test_data = 'tabular_test.csv'
    with tempfile.TemporaryDirectory() as root:
        _prepare_data(train_data, tune_data, test_data)
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
        cloud_predictor = TabularCloudPredictor(cloud_output_path='s3://ag-cloud-predictor/test-tabular', local_output_path='test_tabular_cloud_predictor')
        cloud_predictor_no_train = TabularCloudPredictor(cloud_output_path='s3://ag-cloud-predictor/test-tabular-no-train', local_output_path='test_tabular_cloud_predictor_no_train')
        _test_functionality(cloud_predictor, predictor_init_args, predictor_fit_args, cloud_predictor_no_train, test_data)


@pytest.mark.cloud
def test_text():
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
        cloud_predictor = TextCloudPredictor(cloud_output_path='s3://ag-cloud-predictor/test-text', local_output_path='test_text_cloud_predictor')
        cloud_predictor_no_train = TextCloudPredictor(cloud_output_path='s3://ag-cloud-predictor/test-text-no-train', local_output_path='test_text_cloud_predictor_no_train')
        _test_functionality(cloud_predictor, predictor_init_args, predictor_fit_args, cloud_predictor_no_train, test_data, fit_instance_type='ml.g4dn.2xlarge')

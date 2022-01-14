import os
import pytest
from autogluon.cloud import TabularCloudPredictor, TextCloudPredictor


here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')


def _test_functionality(cloud_predictor, predictor_init_args, predictor_fit_args, test_data, fit_instance_type=None):
    if not fit_instance_type:
        fit_instance_type = 'ml.m5.2xlarge'
    cloud_predictor.fit(
        predictor_init_args,
        predictor_fit_args,
        instance_type=fit_instance_type
    )
    assert cloud_predictor.info['local_output_path'] is not None
    assert cloud_predictor.info['cloud_output_path'] is not None
    assert cloud_predictor.info['fit_job']['name'] is not None
    assert cloud_predictor.info['fit_job']['status'] == 'Completed'

    cloud_predictor.deploy()
    cloud_predictor.predict_real_time(test_data)
    detached_endpoint = cloud_predictor.detach_endpoint()
    cloud_predictor.attach_endpoint(detached_endpoint)
    cloud_predictor.predict_real_time(test_data)
    cloud_predictor.cleanup_deployment()

    cloud_predictor.predict(test_data)
    assert cloud_predictor.info['recent_transform_job']['status'] == 'Completed'


@pytest.mark.cloud
def test_tabular():
    train_data = os.path.join(res, 'train.csv')
    tune_data = os.path.join(res, 'tune.csv')
    test_data = os.path.join(res, 'sample_adult.csv')
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
    cloud_predictor = TabularCloudPredictor()
    _test_functionality(cloud_predictor, predictor_init_args, predictor_fit_args, test_data)


@pytest.mark.cloud
def test_text():
    train_data = os.path.join(res, 'text_train.csv')
    tune_data = os.path.join(res, 'text_tune.csv')
    test_data = os.path.join(res, 'text_test.csv')
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
    cloud_predictor = TextCloudPredictor()
    _test_functionality(cloud_predictor, predictor_init_args, predictor_fit_args, test_data, fit_instance_type='ml.g4dn.2xlarge')

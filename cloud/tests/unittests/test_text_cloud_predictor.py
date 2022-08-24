import pytest
import tempfile

from autogluon.cloud import TextCloudPredictor

from utils import (
    _prepare_data,
    _test_functionality
)


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
        cloud_predictor = TextCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-text',
            local_output_path='test_text_cloud_predictor'
        )
        cloud_predictor_no_train = TextCloudPredictor(
            cloud_output_path='s3://ag-cloud-predictor/test-text-no-train',
            local_output_path='test_text_cloud_predictor_no_train'
        )
        _test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_instance_type='ml.g4dn.2xlarge'
        )

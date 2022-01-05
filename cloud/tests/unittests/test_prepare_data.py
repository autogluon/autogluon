import os
import pandas as pd
from autogluon.cloud import CloudPredictor

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')


def _prepare_predictor():
    return CloudPredictor('tabular', role_arn='dummy')


def _prepare_path(input_type, output_type):
    cp = _prepare_predictor()
    file = os.path.join(res, f'sample.{input_type}')
    file_name = 'unittest'
    path = cp._prepare_data(file, file_name, output_type)
    return path


def test_prepare_data():
    for input_type in ['csv', 'parquet']:
        for output_type in ['csv', 'parquet']:
            sample_file = os.path.join(res, f'sample.{input_type}')
            path = _prepare_path(input_type, output_type)
            assert os.path.exists(path)
            sample_df = pd.read_csv(sample_file)
            temp_df = pd.read_csv(path)
            assert temp_df == sample_df

import os
import tempfile

import boto3
import pandas as pd

from autogluon.cloud import TabularCloudPredictor


def _prepare_predictor():
    return TabularCloudPredictor(cloud_output_path="dummy")


def _prepare_path(input_file, output_type):
    cp = _prepare_predictor()
    file_name = "unittest"
    path = cp._prepare_data(input_file, file_name, output_type)
    return path


def test_prepare_data():
    with tempfile.TemporaryDirectory() as _:
        file_types = ["csv", "parquet"]
        s3 = boto3.client("s3")
        for file_type in file_types:
            s3.download_file("autogluon-cloud", f"sample.{file_type}", f"sample.{file_type}")
        for input_type in file_types:
            for output_type in file_types:
                sample_file = f"sample.{input_type}"
                path = _prepare_path(sample_file, output_type)
                assert os.path.exists(path)
                if input_type == "csv":
                    sample_df = pd.read_csv(sample_file)
                else:
                    sample_df = pd.read_parquet(sample_file)
                if output_type == "csv":
                    temp_df = pd.read_csv(path)
                else:
                    temp_df = pd.read_parquet(path)
                assert temp_df.equals(sample_df)

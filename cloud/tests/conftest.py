import os
import zipfile

import boto3
import pandas as pd
import pytest

from datetime import datetime, timezone


class CloudTestHelper:

    cpu_training_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-training:cpu-latest"
    gpu_training_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-training:gpu-latest"
    cpu_inference_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-inference:cpu-latest"
    gpu_inference_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-inference:gpu-latest"

    @staticmethod
    def prepare_data(*args):
        # TODO: make this handle more general structured directory format
        """
        Download files specified by args from cloud CI s3 bucket

        args: str
            names of files to download
        """
        s3 = boto3.client("s3")
        for arg in args:
            s3.download_file("autogluon-cloud", arg, os.path.basename(arg))

    @staticmethod
    def get_utc_timestamp_now():
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    @staticmethod
    def extract_images(image_zip_file):
        with zipfile.ZipFile(image_zip_file, "r") as zip_ref:
            zip_ref.extractall(".")

    @staticmethod
    def replace_image_abspath(data, image_column):
        data = pd.read_csv(data)
        data[image_column] = [os.path.abspath(path) for path in data[image_column]]
        return data

    @staticmethod
    def test_endpoint(cloud_predictor, test_data, **predict_real_time_kwargs):
        try:
            cloud_predictor.predict_real_time(test_data, **predict_real_time_kwargs)
        except Exception as e:
            cloud_predictor.cleanup_deployment()  # cleanup endpoint if test failed
            raise e

    @staticmethod
    def test_functionality(
        cloud_predictor,
        predictor_init_args,
        predictor_fit_args,
        cloud_predictor_no_train,
        test_data,
        fit_kwargs=None,
        deploy_kwargs=None,
        predict_real_time_kwargs=None,
        predict_kwargs=None
    ):
        if fit_kwargs is None:
            fit_kwargs = dict(instance_type="ml.m5.2xlarge")
        cloud_predictor.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            **fit_kwargs,
        )
        info = cloud_predictor.info()
        assert info["local_output_path"] is not None
        assert info["cloud_output_path"] is not None
        assert info["fit_job"]["name"] is not None
        assert info["fit_job"]["status"] == "Completed"

        if deploy_kwargs is None:
            deploy_kwargs = dict()
        if predict_real_time_kwargs is None:
            predict_real_time_kwargs = dict()
        cloud_predictor.deploy(**deploy_kwargs)
        CloudTestHelper.test_endpoint(cloud_predictor, test_data, **predict_real_time_kwargs)
        detached_endpoint = cloud_predictor.detach_endpoint()
        cloud_predictor.attach_endpoint(detached_endpoint)
        CloudTestHelper.test_endpoint(cloud_predictor, test_data, **predict_real_time_kwargs)
        cloud_predictor.save()
        cloud_predictor = cloud_predictor.__class__.load(cloud_predictor.local_output_path)
        CloudTestHelper.test_endpoint(cloud_predictor, test_data, **predict_real_time_kwargs)
        cloud_predictor.cleanup_deployment()

        info = cloud_predictor.info()
        assert info["local_output_path"] is not None
        assert info["cloud_output_path"] is not None
        assert info["fit_job"]["name"] is not None
        assert info["fit_job"]["status"] == "Completed"

        if predict_kwargs is None:
            predict_kwargs = dict()
        cloud_predictor.predict(test_data, **predict_kwargs)
        info = cloud_predictor.info()
        assert info["recent_transform_job"]["status"] == "Completed"

        # Test deploy with already trained predictor
        trained_predictor_path = cloud_predictor._fit_job.get_output_path()
        cloud_predictor_no_train.deploy(predictor_path=trained_predictor_path, **deploy_kwargs)
        CloudTestHelper.test_endpoint(cloud_predictor_no_train, test_data, **predict_real_time_kwargs)
        cloud_predictor_no_train.cleanup_deployment()

        cloud_predictor_no_train.predict(test_data, predictor_path=trained_predictor_path, **predict_kwargs)
        info = cloud_predictor_no_train.info()
        assert info["recent_transform_job"]["status"] == "Completed"


@pytest.fixture
def test_helper():
    return CloudTestHelper

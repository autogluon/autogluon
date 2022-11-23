import copy

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.utils.s3_utils import is_s3_url

from ..utils.ag_sagemaker import AutoGluonImageRealtimePredictor
from ..utils.constants import VALID_ACCEPT
from ..utils.utils import is_image_file, read_image_bytes_and_encode
from .cloud_predictor import CloudPredictor


class ImageCloudPredictor(CloudPredictor):

    predictor_file_name = "ImageCloudPredictor.pkl"

    @property
    def predictor_type(self):
        return "image"

    @property
    def _realtime_predictor_cls(self):
        return AutoGluonImageRealtimePredictor

    def _get_local_predictor_cls(self):
        from autogluon.vision import ImagePredictor

        predictor_cls = ImagePredictor
        return predictor_cls

    def fit(self, *, predictor_init_args, predictor_fit_args, image_path, **kwargs):
        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            image_path=image_path,
            **kwargs,
        )

    def predict_real_time(self, test_data, test_data_image_column=None, accept="application/x-parquet"):
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be a pandas.DataFrame, a numpy.ndarray, a local path or a s3 path to a csv file containing paths of test images.
            Or a local path to a single image file.
            Or a list of local paths to image files.
        test_data_image_column: Optional(str)
            If provided a pandas.DataFrame as the test_data, you must specify the column name corresponding to image paths.
            Images have to live in the same directory specified by the column.
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.DataFrame
        Predict results in DataFrame
        """
        assert self.endpoint, "Please call `deploy()` to deploy an endpoint first."
        assert accept in VALID_ACCEPT, f"Invalid accept type. Options are {VALID_ACCEPT}."

        import numpy as np

        if isinstance(test_data, str):
            if is_s3_url(test_data):
                test_data = load_pd.load(test_data)
            else:
                if is_image_file(test_data):
                    test_data = np.array([read_image_bytes_and_encode(test_data)], dtype="object")
                else:
                    test_data = load_pd.load(test_data)
        if isinstance(test_data, list):
            images = []
            test_data = np.array([read_image_bytes_and_encode(image) for image in images], dtype="object")
        if isinstance(test_data, pd.DataFrame):
            assert test_data_image_column is not None, "Please specify an image column name"
            assert test_data_image_column in test_data, "Please specify a valid image column name"

            # Convert test data to be numpy array for network transfer
            test_data = np.asarray(
                [read_image_bytes_and_encode(image_path) for image_path in test_data[test_data_image_column]]
            )
        assert isinstance(test_data, np.ndarray), f"Invalid test data format {type(test_data)}"

        return self._predict_real_time(test_data=test_data, accept=accept)

    def predict(
        self,
        test_data,
        **kwargs,
    ):
        """
        test_data: str
            The test data to be inferenced. Can be a local path or a s3 path to a directory containing the images.
        kwargs:
            Refer to `CloudPredictor.predict()`
        """
        split_type = None
        content_type = "application/x-image"
        kwargs = copy.deepcopy(kwargs)
        transformer_kwargs = kwargs.pop("transformer_kwargs", dict())
        transformer_kwargs["strategy"] = "SingleRecord"
        super().predict(
            test_data,
            split_type=split_type,
            content_type=content_type,
            transformer_kwargs=transformer_kwargs,
            **kwargs,
        )

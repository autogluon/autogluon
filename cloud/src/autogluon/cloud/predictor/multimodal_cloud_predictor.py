import copy
import logging
import os

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix

from ..utils.ag_sagemaker import AutoGluonMultiModalRealtimePredictor
from ..utils.constants import VALID_ACCEPT
from ..utils.s3_utils import is_s3_folder
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe, is_image_file, read_image_bytes_and_encode
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class MultiModalCloudPredictor(CloudPredictor):

    predictor_file_name = "MultiModalCloudPredictor.pkl"

    @property
    def predictor_type(self):
        return "multimodal"

    @property
    def _realtime_predictor_cls(self):
        return AutoGluonMultiModalRealtimePredictor

    def _get_local_predictor_cls(self):
        from distutils.version import LooseVersion

        import autogluon.text

        if LooseVersion(autogluon.text.__version__) < LooseVersion("0.5"):
            from autogluon.text.automm import AutoMMPredictor

            multimodal_predictor_cls = AutoMMPredictor
        else:
            from autogluon.multimodal import MultiModalPredictor

            multimodal_predictor_cls = MultiModalPredictor

        predictor_cls = multimodal_predictor_cls
        return predictor_cls

    def predict_real_time(
        self,
        test_data,
        test_data_image_path=None,
        test_data_image_column=None,
        accept="application/x-parquet"
    ):
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be a pandas.DataFrame, a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_path` and `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a pandas.DataFrame, a local path to a csv file.
                    Similarly, you need to specify `test_data_image_path` and `test_data_image_column`, and make sure the image column contains relative path to the image.
                Or a local path to a single image file.
                Or a list of local paths to image files.
        test_data_image_path: str, default = None
            A local path to the images. This parameter is REQUIRED if you want to inference with multimodality involving image modality.
            If you provided this parameter, the image path inside your train/tune data MUST be relative.
            Path needs to be a folder containing the images.

            Example:
            If your images live under a root directory `example_images/`, then you would provide `example_images` as the `image_path`.
            And you want to make sure in your test file, the column corresponding to the images is a relative path prefix with the root directory.
            For example, `example_images/test/image1.png`. An absolute path will NOT work.
        test_data_image_column: default = None
            If provided a pandas.DataFrame as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
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
            if is_image_file(test_data):
                test_data = [test_data]
            else:
                test_data = load_pd.load(test_data)
        if isinstance(test_data, list):
            test_data = np.array([read_image_bytes_and_encode(image) for image in test_data], dtype="object")
            content_type = "application/x-npy"
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_path is not None or test_data_image_column is not None:
                assert test_data_image_path is not None and test_data_image_column is not None, \
                    "Please specify both `test_data_image_path` and `test_data_image_column` when involves image modality"
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(
                    dataframe=test_data,
                    image_root_path=test_data_image_path,
                    image_column=test_data_image_column
                )
            content_type = "application/x-parquet"

        # Providing content type here because sagemaker serializer doesn't support change content type dynamically.
        # Pass to `endpoint.predict()` call as `initial_args` instead
        return self._predict_real_time(test_data=test_data, accept=accept, ContentType=content_type)

    def predict(
        self,
        test_data,
        test_data_image_column=None,
        **kwargs,
    ):
        """
        test_data: str
            The test data to be inferenced.
            Can be a pandas.DataFrame, a local path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a local path to a directory containing the images.
                or a local path to a single image.
        test_data_image_column: Optional(str)
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            Images have to live in the same directory specified by the column.
        kwargs:
            Refer to `CloudPredictor.predict()`
        """
        image_modality_only = False
        if isinstance(test_data, str):
            if os.path.isdir(test_data) or is_image_file(test_data):
                image_modality_only = True

        if image_modality_only:
            split_type = None
            content_type = "application/x-image"
            kwargs = copy.deepcopy(kwargs)
            transformer_kwargs = kwargs.pop("transformer_kwargs", dict())
            transformer_kwargs["strategy"] = "SingleRecord"
            super().predict(
                test_data,
                test_data_image_column=None,
                split_type=split_type,
                content_type=content_type,
                transformer_kwargs=transformer_kwargs,
                **kwargs,
            )
        else:
            super().predict(
                test_data,
                test_data_image_column=test_data_image_column,
                **kwargs,
            )

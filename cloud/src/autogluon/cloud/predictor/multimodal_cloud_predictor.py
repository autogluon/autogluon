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

    def predict_real_time(self, test_data, test_data_image_column=None, accept="application/x-parquet"):
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be a pandas.DataFrame, a local path or a s3 path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a pandas.DataFrame, a local path or a s3 path to a csv file.
                    Similarly, You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
                Or a local path to a single image file.
                Or a list of local paths to image files.
        test_data_image_column: Optional(str)
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
            if is_s3_url(test_data):
                test_data = load_pd.load(test_data)
            else:
                if is_image_file(test_data):
                    test_data = np.array([read_image_bytes_and_encode(test_data)], dtype="object")
                    content_type = "application/x-npy"
                else:
                    test_data = load_pd.load(test_data)
        if isinstance(test_data, list):
            images = []
            test_data = np.array([read_image_bytes_and_encode(image) for image in images], dtype="object")
            content_type = "application/x-npy"
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_column is not None:
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)
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
            Can be a pandas.DataFrame, a local path or a s3 path to a csv file.
            When predicting multimodality with image modality:
                You need to specify `test_data_image_column`, and make sure the image column contains relative path to the image.
            When predicting with only images:
                Can be a local path or a s3 path to a directory containing the images.
                or a local path or a s3 path to a single image.
        test_data_image_column: Optional(str)
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            Images have to live in the same directory specified by the column.
        kwargs:
            Refer to `CloudPredictor.predict()`
        """
        image_modality_only = False
        if isinstance(test_data, str):
            if is_s3_url(test_data):
                if is_s3_folder(test_data, session=self.sagemaker_session):
                    image_modality_only = True
                else:
                    bucket, prefix = s3_path_to_bucket_prefix(test_data)
                    utils_folder = "utils"
                    self.sagemaker_session.download_data(utils_folder, bucket, key_prefix=prefix)
                    filename = prefix.rsplit("/")[-1]
                    if is_image_file(os.path.join(utils_folder, filename)):
                        image_modality_only = True
            elif os.path.isdir(test_data) or is_image_file(test_data):
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

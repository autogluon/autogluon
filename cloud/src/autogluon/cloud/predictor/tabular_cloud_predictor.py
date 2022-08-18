import copy
import os
import pandas as pd
import yaml

from autogluon.common.loaders import load_pd

from .cloud_predictor import CloudPredictor
from ..utils.constants import VALID_ACCEPT
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe


class TabularCloudPredictor(CloudPredictor):

    predictor_file_name = 'TabularCloudPredictor.pkl'

    @property
    def predictor_type(self):
        return 'tabular'

    def _get_local_predictor_cls(self):
        from autogluon.tabular import TabularPredictor
        predictor_cls = TabularPredictor
        return predictor_cls

    def fit(
        self,
        *,
        predictor_init_args,
        predictor_fit_args,
        image_path=None,
        image_column_name=None,
        **kwargs
    ):
        if image_path is not None:
            assert image_column_name is not None, 'Please provide `image_column_name` when training multimodality with image modality'
        if image_column_name is not None:
            assert image_path is not None, 'Please provide `image_path` when training multimodality with image modality'
        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            image_path=image_path,
            image_column_name=image_column_name,
            **kwargs
        )

    def predict_real_time(
            self,
            test_data,
            test_data_image_column=None,
            accept='application/x-parquet'
        ):
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
        assert self.endpoint, 'Please call `deploy()` to deploy an endpoint first.'
        assert accept in VALID_ACCEPT, f'Invalid accept type. Options are {VALID_ACCEPT}.'

        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(test_data, pd.DataFrame):
            if test_data_image_column is not None:
                test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)

        return self._predict_real_time(test_data=test_data, accept=accept)

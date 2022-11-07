import copy
import logging
import os
import pandas as pd
import yaml

from autogluon.common.loaders import load_pd
from botocore.exceptions import ClientError

from .cloud_predictor import CloudPredictor
from ..utils.constants import VALID_ACCEPT
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe


logger = logging.getLogger(__name__)


class TabularCloudPredictor(CloudPredictor):

    predictor_file_name = 'TabularCloudPredictor.pkl'

    @property
    def predictor_type(self):
        return 'tabular'

    def _get_local_predictor_cls(self):
        from autogluon.tabular import TabularPredictor
        predictor_cls = TabularPredictor
        return predictor_cls

    def _construct_config(self, predictor_init_args, predictor_fit_args, leaderboard, **kwargs):
        assert self.predictor_type is not None
        if 'feature_metadata' in predictor_fit_args:
            predictor_fit_args = copy.deepcopy(predictor_fit_args)
            feature_metadata = predictor_fit_args.pop('feature_metadata')
            feature_metadata = dict(
                type_map_raw=feature_metadata.type_map_raw,
                type_map_special=feature_metadata.get_type_map_special(),
            )
            assert 'feature_metadata' not in kwargs, 'feature_metadata in both `predictor_fit_args` and kwargs. This should not happen.'
            kwargs['feature_metadata'] = feature_metadata
        config = dict(
            predictor_type=self.predictor_type,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
            **kwargs
        )
        path = os.path.join(self.local_output_path, 'utils', 'config.yaml')
        with open(path, 'w') as f:
            yaml.dump(config, f)
        return path

    def fit(
        self,
        *,
        predictor_init_args,
        predictor_fit_args,
        image_path=None,
        image_column=None,
        **kwargs
    ):
        """
        predictor_init_args: dict
            Init args for the predictor
        predictor_fit_args: dict
            Fit args for the predictor
        image_path: str, default = None
            A local path or s3 path to the images. This parameter is REQUIRED if you want to train predictor with image modality.
            If you provided this parameter, the image path inside your train/tune data MUST be relative.
            If local path, path needs to be either a compressed file containing the images or a folder containing the images.
            If it's a folder, we will zip it for you and upload it to the s3.
            If s3 path, the path needs to be a path to a compressed file containing the images

            Example:
            If your images live under a root directory `example_images/`, then you would provide `example_images` as the `image_path`.
            And you want to make sure in your training/tuning file, the column corresponding to the images is a relative path prefix with the root directory.
            For example, `example_images/train/image1.png`. An absolute path will NOT work as the file will be moved to a remote system.
        image_column: str, default = None
            The column name in the training/tuning data that contains the image paths.
            This is REQUIRED if you want to train multimodality with image modality with one exception:
            If you pass in an `autogluon.tabular.FeatureMetadata` object that contains `image_path` special type to `predictor_fit_args`.
        kwargs,
            Refer to `CloudPredictor`
        """
        if image_path is not None:
            assert image_column is not None or 'feature_metadata' in predictor_fit_args,\
                'Please provide `image_column` when training multimodality with image modality'
        if image_column is not None:
            assert image_path is not None, 'Please provide `image_path` when training multimodality with image modality'
        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            image_path=image_path,
            image_column=image_column,
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
        try:
            return self._predict_real_time(test_data=test_data, accept=accept)
        except ClientError as e:
            fail_to_load_on_cpu_error_msg = "GPUAccelerator can not run on your system since the accelerator is not available. The following accelerator(s) is available and can be passed into `accelerator` argument of `Trainer`: ['cpu']."
            if fail_to_load_on_cpu_error_msg in e.response['Error']['Message']:
                logger.warning(e.response['Error']['Message'])
                logger.warning('Warning: Having trouble load gpu trained model on a cpu machine. This is a known issue of AutoGluon and will be fixed in future containers')
                logger.warning('Warning: You can either try deploy on a gpu machine')
                logger.warning('Warning: or download the trained artifact and modify `num_gpus` to be `-1` in the config file located at `models/TextPredictor/config.yaml`')
                logger.warning('Warning: then try to deploy with the modified artifact')
                return None
            raise e

    def predict(
        self,
        test_data,
        test_data_image_column=None,
        **kwargs,
    ):
        """
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, a local path or a s3 path.
        test_data_image_column: Optional(str)
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            Images have to live in the same directory specified by the column.
        kwargs:
            Refer to `CloudPredictor.predict()`
        """
        # TODO: remove this after fix is out for 0.6 release
        # This is because text predictor cannot be saved standalone with current autogluon tabular setting.
        # And sagemaker batch inference container has trouble connecting to hugging face; hence not able to load the model
        logger.warning('Tabular does not support multimodal batch inference yet.')
        super().predict(
            test_data,
            test_data_image_column=test_data_image_column,
            **kwargs
        )

import boto3
import copy
import json
import os
import yaml
import tarfile
import logging
import pandas as pd
import sagemaker

from abc import ABC, abstractmethod
from botocore.exceptions import ClientError
from datetime import datetime
from typing import Optional

from autogluon.common.loaders import load_pd
from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix
from autogluon.common.utils.utils import setup_outputdir

from ..data import FormatConverterFactory
from ..job import SageMakerFitJob, SageMakerBatchTransformationJob
from ..scripts import ScriptManager
from ..utils.ag_sagemaker import (
    AutoGluonRepackInferenceModel,
    AutoGluonNonRepackInferenceModel,
    AutoGluonRealtimePredictor,
    AutoGluonBatchPredictor
)
from ..utils.aws_utils import setup_sagemaker_session
from ..utils.constants import (
    VALID_ACCEPT,
    SAGEMAKER_RESOURCE_PREFIX
)
from ..utils.iam import (
    TRUST_RELATIONSHIP_FILE_NAME,
    IAM_POLICY_FILE_NAME,
    SAGEMAKER_TRUST_RELATIONSHIP,
    SAGEMAKER_CLOUD_POLICY
)
from ..utils.iam import (
    replace_iam_policy_place_holder,
    replace_trust_relationship_place_holder
)
from ..utils.misc import MostRecentInsertedOrderedDict
from ..utils.sagemaker_utils import (
    retrieve_available_framework_versions,
    retrieve_py_versions,
    retrieve_latest_framework_version
)
from ..utils.utils import (
    convert_image_path_to_encoded_bytes_in_dataframe,
    zipfolder,
    is_compressed_file,
    is_image_file,
    unzip_file,
    rename_file_with_uuid
)

logger = logging.getLogger(__name__)


class CloudPredictor(ABC):

    predictor_file_name = 'CloudPredictor.pkl'

    def __init__(
        self,
        cloud_output_path,
        local_output_path=None,
        verbosity=2
    ):
        """
        Parameters
        ----------
        cloud_output_path: str
            Path to s3 location where intermediate artifacts will be uploaded and trained models should be saved.
            This has to be provided because s3 buckets are unique globally, so it is hard to create one for you.
            If you only provided the bucket but not the subfolder, a time-stamped folder called "YOUR_BUCKET/ag-[TIMESTAMP]" will be created.
            If you provided both the bucket and the subfolder, then we will use that instead.
            Note: To call `fit()` twice and save all results of each fit,
            you must either specify different `cloud_output_path` locations or only provide the bucket but not the subfolder.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        local_output_path: str
            Path to directory where downloaded trained predictor, batch transform results, and intermediate outputs should be saved
            If unspecified, a time-stamped folder called "AutogluonCloudPredictor/ag-[TIMESTAMP]" will be created in the working directory to store all downloaded trained predictor, batch transform results, and intermediate outputs.
            Note: To call `fit()` twice and save all results of each fit, you must specify different `local_output_path` locations or don't specify `local_output_path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
        """
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        try:
            self.role_arn = sagemaker.get_execution_role()
        except ClientError as e:
            logger.warning(
                'Failed to get IAM role. Did you configure and authenticate the IAM role?',
                'For more information, https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html',
                f'If you do not have a role created yet, \
                You can use {self.__class__.__name__}.generate_trust_relationship_and_iam_policy_file() to get the required trust relationship and iam policy',
                'You can then use the generated trust relationship and IAM policy to create an IAM role',
                'For more information, https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html',
                'IMPORTANT: Please review the generated trust relationship and IAM policy before you create an IAM role with them'
            )
            raise e
        self.sagemaker_session = setup_sagemaker_session()
        self.local_output_path = self._setup_local_output_path(local_output_path)
        self.cloud_output_path = self._setup_cloud_output_path(cloud_output_path)
        self.endpoint = None

        self._region = self.sagemaker_session.boto_region_name
        self._fit_job = SageMakerFitJob(session=self.sagemaker_session)
        # This holds the Model object after training.
        # After saving or loading, this can only hold a string representing the name of the model in sagemaker model
        self._fitted_sagemaker_model_entity = None
        self._batch_transform_jobs = MostRecentInsertedOrderedDict()

    @property
    @abstractmethod
    def predictor_type(self):
        raise NotImplementedError

    @property
    def _realtime_predictor_cls(self):
        return AutoGluonRealtimePredictor

    @property
    def is_fit(self):
        return self._fit_job.completed

    @property
    def endpoint_name(self):
        """
        Return the CloudPredictor deployed endpoint name
        """
        if self.endpoint:
            return self.endpoint.endpoint_name
        return None

    @staticmethod
    def generate_trust_relationship_and_iam_policy_file(
        account_id: str,
        cloud_output_bucket: str,
        output_path: Optional[str] = None
    ):
        """
        Generate required trust relationship and IAM policy file in json format for CloudPredictor with SageMaker backend.
        Users can use the generated files to create an IAM role for themselves.
        IMPORTANT: Make sure you review both files before creating the role!

        Parameters
        ----------
        account_id: str
            The AWS account ID you plan to use for CloudPredictor.
        cloud_output_bucket: str
            s3 bucket name where intermediate artifacts will be uploaded and trained models should be saved.
            You need to create this bucket beforehand and we would put this bucket in the policy being created.
        output_path: str
            Where you would like the generated file being written to.
            If not specified, will write to the current folder.

        Return
        ------
        A dict containing the trust relationship and IAM policy files paths
        """
        if output_path is None:
            output_path = '.'
        trust_relationship_file_path = os.path.join(output_path, TRUST_RELATIONSHIP_FILE_NAME)
        iam_policy_file_path = os.path.join(output_path, IAM_POLICY_FILE_NAME)

        trust_relationship = replace_trust_relationship_place_holder(
            trust_relationship_document=SAGEMAKER_TRUST_RELATIONSHIP,
            account_id=account_id
        )
        iam_policy = replace_iam_policy_place_holder(
            policy_document=SAGEMAKER_CLOUD_POLICY,
            account_id=account_id,
            bucket=cloud_output_bucket
        )
        with open(trust_relationship_file_path, 'w') as file:
            json.dump(trust_relationship, file, indent=4)

        with open(iam_policy_file_path, 'w') as file:
            json.dump(iam_policy, file, indent=4)

        logger.info(f'Generated trust relationship to {trust_relationship_file_path}')
        logger.info(f'Generated iam policy to {iam_policy_file_path}')
        logger.info('IMPORTANT: Please review the trust relationship and iam policy before you use them to create an IAM role')
        logger.info('Please refer to AWS documentation on how to create an IAM role: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html')

        return {
            'trust_relationship': trust_relationship_file_path,
            'iam_policy': iam_policy_file_path
        }

    def info(self):
        """
        Return general info about CloudPredictor
        """
        info = dict(
            local_output_path=self.local_output_path,
            cloud_output_path=self.cloud_output_path,
            fit_job=self._fit_job.info(),
            recent_transform_job=self._batch_transform_jobs.last_value.info() if len(self._batch_transform_jobs) > 0 else None,
            transform_jobs=[job_name for job_name in self._batch_transform_jobs.keys()],
            endpoint=self.endpoint_name
        )
        return info

    def _setup_predictor_type(self):
        self._train_script_path = None
        self._serve_script_path = None

    def _setup_local_output_path(self, path):
        if path is None:
            utcnow = datetime.utcnow()
            timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
            path = f'AutogluonCloudPredictor/ag-{timestamp}{os.path.sep}'
        path = setup_outputdir(path)
        util_path = os.path.join(path, 'utils')
        try:
            os.makedirs(util_path)
        except FileExistsError:
            logger.warning(f'Warning: path already exists! This predictor may overwrite an existing predictor! path="{path}"')
        return os.path.abspath(path)

    def _setup_cloud_output_path(self, path):
        if path.endswith('/'):
            path = path[:-1]
        path_cleaned = path
        try:
            path_cleaned = path.split('://', 1)[1]
        except:
            pass
        path_split = path_cleaned.split('/', 1)
        # If user only provided the bucket, we create a subfolder with timestamp for them
        if len(path_split) == 1:
            path = os.path.join(path, f'ag-{sagemaker.utils.sagemaker_timestamp()}')
        if is_s3_url(path):
            return path
        return 's3://' + path

    def _retrieve_latest_framework_version(self, framework_type='training'):
        return retrieve_latest_framework_version(framework_type)

    def _parse_framework_version(self, framework_version, framework_type, py_version=None):
        if framework_version == 'latest':
            framework_version, py_versions = self._retrieve_latest_framework_version(framework_type)
            py_version = py_versions[0]
        else:
            valid_options = retrieve_available_framework_versions(framework_type)
            assert framework_version in valid_options, f'{framework_version} is not a valid option. Options are: {valid_options}'

            valid_py_versions = retrieve_py_versions(framework_version, framework_type)
            if py_version is not None:
                assert py_version in valid_py_versions, f'{py_version} is no a valid option. Options are {valid_py_versions}'
            else:
                py_version = valid_py_versions[0]
        return framework_version, py_version

    def _construct_config(self, predictor_init_args, predictor_fit_args, leaderboard, **kwargs):
        assert self.predictor_type is not None
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

    def _setup_bucket(self, bucket):
        s3 = boto3.resource('s3')
        if not s3.Bucket(bucket) in s3.buckets.all():
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={
                    'LocationConstraint': self.sagemaker_session.boto_region_name,
                }
            )

    def _prepare_data(self, data, filename, output_type='csv'):
        path = os.path.join(self.local_output_path, 'utils')
        converter = FormatConverterFactory.get_converter(output_type)
        return converter.convert(data, path, filename)

    def _upload_fit_image_artifact(
        self,
        images,
        bucket,
        key_prefix
    ):
        if images is not None:
            if is_s3_url(images):
                fileexts = ('.tar.gz', '.bz2', '.zip')  # More extensions?
                assert images.endswith(fileexts), 'Please provide a s3 path to a compressed file'
            else:
                image_zip_filename = images
                if os.path.isdir(images):
                    image_zip_filename = os.path.basename(os.path.normpath(images))
                    zipfolder(image_zip_filename, images)
                    image_zip_filename += '.zip'
                else:
                    assert is_compressed_file(images), 'Please provide a compressed file or a folder containing the images'
                logger.log(20, 'Uploading images ...')
                images = self.sagemaker_session.upload_data(
                    path=image_zip_filename,
                    bucket=bucket,
                    key_prefix=key_prefix,
                )
                logger.log(20, 'Images uploaded successfully')
        return images

    def _upload_fit_artifact(
        self,
        train_data,
        tune_data,
        config,
        serving_script,
        images=None,
    ):
        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(self.cloud_output_path)
        util_key_prefix = cloud_key_prefix + '/utils'
        train_input = train_data
        if isinstance(train_data, pd.DataFrame) or not is_s3_url(train_data):
            train_data = self._prepare_data(train_data, 'train')
            logger.log(20, 'Uploading train data...')
            train_input = self.sagemaker_session.upload_data(
                path=train_data,
                bucket=cloud_bucket,
                key_prefix=util_key_prefix
            )
            logger.log(20, 'Train data uploaded successfully')

        tune_input = tune_data
        if tune_data is not None:
            if isinstance(tune_data, pd.DataFrame) or not is_s3_url(tune_data):
                tune_data = self._prepare_data(tune_data, 'tune')
                logger.log(20, 'Uploading tune data...')
                tune_input = self.sagemaker_session.upload_data(
                    path=tune_data,
                    bucket=cloud_bucket,
                    key_prefix=util_key_prefix
                )
                logger.log(20, 'Tune data uploaded successfully')

        config_input = self.sagemaker_session.upload_data(
            path=config,
            bucket=cloud_bucket,
            key_prefix=util_key_prefix
        )

        serving_input = self.sagemaker_session.upload_data(
            path=serving_script,
            bucket=cloud_bucket,
            key_prefix=util_key_prefix
        )

        images_input = self._upload_fit_image_artifact(
            images=images,
            bucket=cloud_bucket,
            key_prefix=util_key_prefix
        )
        inputs = dict(train=train_input, config=config_input, serving=serving_input)
        if tune_input is not None:
            inputs['tune'] = tune_input
        if images_input is not None:
            inputs['images'] = images_input

        return inputs

    def fit(
        self,
        *,
        predictor_init_args,
        predictor_fit_args,
        image_path=None,
        image_column=None,
        leaderboard=True,
        framework_version='latest',
        job_name=None,
        instance_type='ml.m5.2xlarge',
        instance_count=1,
        volume_size=100,
        wait=True,
        autogluon_sagemaker_estimator_kwargs=dict(),
        **kwargs
    ):
        """
        Fit the predictor with SageMaker.
        This function will first upload necessary config and train data to s3 bucket.
        Then launch a SageMaker training job with the AutoGluon training container.

        Parameters
        ----------
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
        leaderboard: bool, default = True
            Whether to include the leaderboard in the output artifact
        framework_version: str, default = `latest`
            Training container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
        job_name: str, default = None
            Name of the launched training job.
            If None, CloudPredictor will create one with prefix ag-cloudpredictor
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance type the predictor will be trained on with SageMaker.
        instance_count: int, default = 1
            Number of instance used to fit the predictor.
        volumes_size: int, default = 30
            Size in GB of the EBS volume to use for storing input data during training (default: 30).
            Must be large enough to store training data if File Mode is used (which is the default).
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        autogluon_sagemaker_estimator_kwargs: dict, default = dict()
            Any extra arguments needed to initialize AutoGluonSagemakerEstimator
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework for all options
        **kwargs:
            Any extra arguments needed to pass to fit.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework.fit for all options

        Returns
        -------
        `CloudPredictor` object. Returns self.
        """
        assert not self._fit_job.completed, 'Predictor is already fit! To fit additional models, create a new `CloudPredictor`'
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop('train_data')
        tune_data = predictor_fit_args.pop('tuning_data', None)
        framework_version, py_version = self._parse_framework_version(framework_version, 'training')
        logger.log(20, f'Training with framework_version=={framework_version}')

        if not job_name:
            job_name = sagemaker.utils.unique_name_from_base(SAGEMAKER_RESOURCE_PREFIX)

        autogluon_sagemaker_estimator_kwargs = copy.deepcopy(autogluon_sagemaker_estimator_kwargs)
        autogluon_sagemaker_estimator_kwargs.pop('output_path', None)
        output_path = self.cloud_output_path + '/output'
        cloud_bucket, _ = s3_path_to_bucket_prefix(self.cloud_output_path)

        self._train_script_path = ScriptManager.get_train_script(self.predictor_type, framework_version)
        entry_point = self._train_script_path
        user_entry_point = autogluon_sagemaker_estimator_kwargs.pop(entry_point, None)
        if user_entry_point:
            logger.warning(f'Providing a custom entry point could break the fit. Please refer to `{entry_point}` for our implementation')
            entry_point = user_entry_point
        else:
            # Avoid user passing in source_dir without specifying entry point
            autogluon_sagemaker_estimator_kwargs.pop('source_dir', None)

        self._setup_bucket(cloud_bucket)
        config_args = dict(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            leaderboard=leaderboard,
        )
        if image_column is not None:
            config_args['image_column'] = image_column
        config = self._construct_config(**config_args)
        inputs = self._upload_fit_artifact(
            train_data=train_data,
            tune_data=tune_data,
            config=config,
            images=image_path,
            serving_script=ScriptManager.get_serve_script(self.predictor_type, framework_version)  # Training and Inference should have the same framework_version
        )

        self._fit_job.run(
            role=self.role_arn,
            entry_point=entry_point,
            region=self._region,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            framework_version=framework_version,
            py_version=py_version,
            base_job_name="autogluon-cloudpredictor-train",
            output_path=output_path,
            inputs=inputs,
            wait=wait,
            job_name=job_name,
            autogluon_sagemaker_estimator_kwargs=autogluon_sagemaker_estimator_kwargs,
            **kwargs
        )
        return self

    def attach_job(self, job_name):
        """
        Attach to a sagemaker training job.
        This is useful when the local process crashed and you want to reattach to the previous job

        Parameters
        ----------
        job_name: str
            The name of the job being attached

        Returns
        -------
        `CloudPredictor` object. Returns self.
        """
        self._fit_job = SageMakerFitJob.attach(job_name)
        return self

    def get_fit_job_status(self):
        """
        Get the status of the training job.
        This is useful when the user made an asynchronous call to the `fit()` function

        Returns
        -------
        str,
        Valid Values: InProgress | Completed | Failed | Stopping | Stopped | NotCreated
        """
        return self._fit_job.get_job_status()

    def download_trained_predictor(self, save_path=None):
        """
        Download the trained predictor from the cloud.

        Parameters
        ----------
        save_path: str
            Path to save the model.
            If None, CloudPredictor will create a folder 'AutogluonModels' for the model under `local_output_path`.

        Returns
        -------
        save_path: str
            Path to the saved model.
        """
        path = self._fit_job.get_output_path()
        if not save_path:
            save_path = self.local_output_path
        save_path = self._download_predictor(path, save_path)
        return save_path

    def _get_local_predictor_cls(self):
        raise NotImplementedError

    def to_local_predictor(self, save_path=None):
        """
        Convert the SageMaker trained predictor to a local TabularPredictor or TextPredictor.

        Parameters
        ----------
        save_path: str
            Path to save the model.
            If None, CloudPredictor will create a folder for the model.

        Returns
        -------
        AutoGluon Predictor,
            TabularPredictor or TextPredictor based on `predictor_type`
        """
        predictor_cls = self._get_local_predictor_cls()
        local_model_path = self.download_trained_predictor(save_path)
        return predictor_cls.load(local_model_path)

    def _upload_predictor(self, predictor_path, key_prefix):
        cloud_bucket, _ = s3_path_to_bucket_prefix(self.cloud_output_path)
        if not is_s3_url(predictor_path):
            if os.path.isfile(predictor_path):
                if tarfile.is_tarfile(predictor_path):
                    predictor_path = self.sagemaker_session.upload_data(
                        path=predictor_path,
                        bucket=cloud_bucket,
                        key_prefix=key_prefix
                    )
                else:
                    raise ValueError('Please provide a tarball containing the model')
            else:
                raise ValueError('Please provide a valid path to the model tarball.')
        return predictor_path

    def deploy(
        self,
        predictor_path=None,
        endpoint_name=None,
        framework_version='latest',
        instance_type='ml.m5.2xlarge',
        initial_instance_count=1,
        wait=True,
        model_kwargs=dict(),
        **kwargs
    ):
        """
        Deploy a predictor as a SageMaker endpoint, which can be used to do real-time inference later.
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        and then deploy it to the endpoint.

        Parameters
        ----------
        predictor_path: str
            Path to the predictor tarball you want to deploy.
            Path can be both a local path or a S3 location.
            If None, will deploy the most recent trained predictor trained with `fit()`.
        endpoint_name: str
            The endpoint name to use for the deployment.
            If None, CloudPredictor will create one with prefix `ag-cloudpredictor`
        framework_version: str, default = `latest`
            Inference container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be deployed for the endpoint
        initial_instance_count: int, default = 1,
            Initial number of instances to be deployed for the endpoint
        wait: Bool, default = True,
            Whether to wait for the endpoint to be deployed.
            To be noticed, the function won't return immediately because there are some preparations needed prior deployment.
        model_kwargs: dict, default = dict()
            Any extra arguments needed to initialize Sagemaker Model
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
        **kwargs:
            Any extra arguments needed to pass to deploy.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy for all options
        """
        assert self.endpoint is None, 'There is an endpoint already attached. Either detach it with `detach` or clean it up with `cleanup_deployment`'
        if not predictor_path:
            predictor_path = self._fit_job.get_output_path()
            assert predictor_path, 'No cloud trained model found.'
        predictor_path = self._upload_predictor(predictor_path, f'endpoints/{endpoint_name}/predictor')

        if not endpoint_name:
            endpoint_name = sagemaker.utils.unique_name_from_base(SAGEMAKER_RESOURCE_PREFIX)
        framework_version, py_version = self._parse_framework_version(framework_version, 'inference')
        logger.log(20, f'Deploying with framework_version=={framework_version}')

        self._serve_script_path = ScriptManager.get_serve_script(self.predictor_type, framework_version)
        entry_point = self._serve_script_path
        model_kwargs = copy.deepcopy(model_kwargs)
        user_entry_point = model_kwargs.pop('entry_point', None)
        if user_entry_point:
            logger.warning(f'Providing a custom entry point could break the deployment. Please refer to `{entry_point}` for our implementation')
            entry_point = user_entry_point

        repack_model = False
        if predictor_path != self._fit_job.get_output_path() or user_entry_point is not None:
            # Not inference on cloud trained model or not using inference on cloud trained model
            # Need to repack the code into model. This will slow down batch inference and deployment
            repack_model = True
        predictor_cls = self._realtime_predictor_cls
        user_predictor_cls = model_kwargs.pop('predictor_cls', None)
        if user_predictor_cls:
            logger.warning('Providing a custom predictor_cls could break the deployment. Please refer to `AutoGluonRealtimePredictor` for how to provide a custom predictor')
            predictor_cls = user_predictor_cls

        if repack_model:
            model_cls = AutoGluonRepackInferenceModel
        else:
            model_cls = AutoGluonNonRepackInferenceModel
        model = model_cls(
            model_data=predictor_path,
            role=self.role_arn,
            region=self._region,
            framework_version=framework_version,
            py_version=py_version,
            instance_type=instance_type,
            entry_point=entry_point,
            predictor_cls=predictor_cls,
            **model_kwargs
        )

        logger.log(20, 'Deploying model to the endpoint')
        self.endpoint = model.deploy(
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            wait=wait,
            **kwargs
        )

    def attach_endpoint(self, endpoint):
        """
        Attach the current CloudPredictor to an existing SageMaker endpoint.

        Parameters
        ----------
        endpoint: str or  :class:`AutoGluonRealtimePredictor` or :class:`AutoGluonImageRealtimePredictor`
            If str is passed, it should be the name of the endpoint being attached to.
        """
        assert self.endpoint is None, 'There is an endpoint already attached. Either detach it with `detach` or clean it up with `cleanup_deployment`'
        if type(endpoint) == str:
            self.endpoint = self._realtime_predictor_cls(
                endpoint_name=endpoint,
                sagemaker_session=self.sagemaker_session,
            )
        elif isinstance(endpoint, self._realtime_predictor_cls):
            self.endpoint = endpoint
        else:
            raise ValueError(f'Please provide either an endpoint name or an endpoint of type `{self._realtime_predictor_cls.__name__}`')

    def detach_endpoint(self):
        """
        Detach the current endpoint and return it.

        Returns
        -------
        `AutoGluonRealtimePredictor` or `AutoGluonImageRealtimePredictor` object.
        """
        assert self.endpoint is not None, 'There is no attached endpoint'
        detached_endpoint = self.endpoint
        self.endpoint = None
        return detached_endpoint

    def _predict_real_time(self, test_data, accept, **initial_args):
        try:
            return self.endpoint.predict(test_data, initial_args={'Accept': accept, **initial_args})
        except ClientError as e:
            if e.response['Error']['Code'] == '413':  # Error code for pay load too large
                logger.warning('The invocation of endpoint failed with Error Code 413. This is likely due to pay load size being too large.')
                logger.warning('SageMaker endpoint could only take maximum 5MB. Please consider reduce test data size or use `predict()` instead.')
            raise e

    def predict_real_time(
        self,
        test_data,
        accept='application/x-parquet'
    ):
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, a local path or a s3 path.
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
        if type(test_data) == str:
            test_data = load_pd.load(test_data)
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError('test_data must be either a pandas.DataFrame, a local path or a s3 path')

        return self._predict_real_time(test_data=test_data, accept=accept)

    def _upload_batch_predict_data(self, test_data, bucket, key_prefix):
        if isinstance(test_data, str) and is_s3_url(test_data):
            test_input = test_data
        else:
            # If a directory of images, upload directly
            if isinstance(test_data, str) and not os.path.isdir(test_data):
                # either a file to a dataframe, or a file to an image
                if is_image_file(test_data):
                    logger.warning('Are you sure you want to do batch inference on a single image? You might want to try `deploy()` and `predict_realtime()` instead')
                else:
                    test_data = load_pd.load(test_data)

            if isinstance(test_data, pd.DataFrame):
                test_data = self._prepare_data(test_data, 'test', output_type='csv')
            logger.log(20, 'Uploading data...')
            test_input = self.sagemaker_session.upload_data(
                path=test_data,
                bucket=bucket,
                key_prefix=key_prefix + '/data'
            )
            logger.log(20, 'Data uploaded successfully')

        return test_input

    def predict(
        self,
        test_data,
        test_data_image_column=None,
        predictor_path=None,
        framework_version='latest',
        job_name=None,
        instance_type='ml.m5.2xlarge',
        instance_count=1,
        wait=True,
        model_kwargs=dict(),
        transformer_kwargs=dict(),
        **kwargs,
    ):
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        then create a transformer with it, and call transform in the end.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced. Can be a pandas.DataFrame, a local path or a s3 path.
        test_data_image_column: Optional(str)
            If test_data involves image modality, you must specify the column name corresponding to image paths.
            Images have to live in the same directory specified by the column.
        predictor_path: str
            Path to the predictor tarball you want to use to predict.
            Path can be both a local path or a S3 location.
            If None, will use the most recent trained predictor trained with `fit()`.
        framework_version: str, default = `latest`
            Inference container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
        job_name: str, default = None
            Name of the launched training job.
            If None, CloudPredictor will create one with prefix ag-cloudpredictor.
        instance_count: int, default = 1,
            Number of instances used to do batch transform.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be used for batch transform.
        wait: bool, default = True
            Whether to wait for batch transform to complete.
            To be noticed, the function won't return immediately because there are some preparations needed prior transform.
        model_kwargs: dict, default = dict()
            Any extra arguments needed to initialize Sagemaker Model
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
        transformer_kwargs: dict
            Any extra arguments needed to pass to transformer.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer for all options.
        **kwargs:
            Any extra arguments needed to pass to transform.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer.transform for all options.
        """
        # Sagemaker batch transformation is able to take in headers during the most recent test
        # logger.warning('Please remove headers of the test data and make sure the columns are in the same order as the training data.')
        if not predictor_path:
            predictor_path = self._fit_job.get_output_path()
            assert predictor_path, 'No cloud trained model found.'

        framework_version, py_version = self._parse_framework_version(framework_version, 'inference')
        logger.log(20, f'Predicting with framework_version=={framework_version}')

        output_path = kwargs.get('output_path', None)
        if not output_path:
            output_path = self.cloud_output_path
        assert is_s3_url(output_path)
        output_path = output_path + '/batch_transform' + f'/{sagemaker.utils.sagemaker_timestamp()}'

        cloud_bucket, cloud_key_prefix = s3_path_to_bucket_prefix(output_path)
        logger.log(20, 'Preparing autogluon predictor...')
        predictor_path = self._upload_predictor(predictor_path, cloud_key_prefix + '/predictor')

        if not job_name:
            job_name = sagemaker.utils.unique_name_from_base(SAGEMAKER_RESOURCE_PREFIX)

        if test_data_image_column is not None:
            logger.warning('Batch inference with image modality could be slow because of some technical details.')
            logger.warning('You can always retrieve the model trained with CloudPredictor and do batch inference using your custom solution.')
            test_data = load_pd.load(test_data)
            test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)
        test_input = self._upload_batch_predict_data(test_data, cloud_bucket, cloud_key_prefix)

        self._serve_script_path = ScriptManager.get_serve_script(self.predictor_type, framework_version)
        entry_point = self._serve_script_path
        model_kwargs = copy.deepcopy(model_kwargs)
        user_entry_point = model_kwargs.pop('entry_point', None)
        repack_model = False
        if predictor_path != self._fit_job.get_output_path() or user_entry_point is not None:
            # Not inference on cloud trained model or not using inference on cloud trained model
            # Need to repack the code into model. This will slow down batch inference and deployment
            repack_model = True
        if user_entry_point:
            entry_point = user_entry_point

        predictor_cls = AutoGluonBatchPredictor
        user_predictor_cls = model_kwargs.pop('predictor_cls', None)
        if user_predictor_cls:
            logger.warning('Providing a custom predictor_cls could break the deployment. Please refer to `AutoGluonBatchPredictor` for how to provide a custom predictor')
            predictor_cls = user_predictor_cls

        kwargs = copy.deepcopy(kwargs)
        content_type = kwargs.pop('content_type', None)
        if 'split_type' not in kwargs:
            split_type = 'Line'
        else:
            split_type = kwargs.pop('split_type')
        if not content_type:
            content_type = 'text/csv'

        batch_transform_job = SageMakerBatchTransformationJob(session=self.sagemaker_session)
        batch_transform_job.run(
            model_data=predictor_path,
            role=self.role_arn,
            region=self._region,
            framework_version=framework_version,
            py_version=py_version,
            instance_count=instance_count,
            instance_type=instance_type,
            entry_point=entry_point,
            predictor_cls=predictor_cls,
            output_path=output_path + '/results',
            test_input=test_input,
            job_name=job_name,
            split_type=split_type,
            content_type=content_type,
            wait=wait,
            transformer_kwargs=transformer_kwargs,
            model_kwargs=model_kwargs,
            repack_model=repack_model,
            **kwargs
        )
        self._batch_transform_jobs[job_name] = batch_transform_job

    def download_predict_results(self, job_name=None, save_path=None):
        """
        Download batch transform result

        Parameters
        ----------
        job_name: str
            The specific batch transform job result to download.
            If None, will download the most recent job result.
        save_path: str
            Path to save the downloaded result.
            If None, CloudPredictor will create one.
        """
        if not job_name:
            job_name = self._batch_transform_jobs.last
        assert job_name is not None, 'There is no batch transform job.'
        job = self._batch_transform_jobs.get(job_name, None)
        assert job is not None, f'Could not find the batch transform job that matches name {job_name}'
        result_path = job.get_output_path()
        assert result_path is not None, 'No predict results found.'
        file_name = result_path.split('/')[-1]
        if not save_path:
            save_path = self.local_output_path
        save_path = os.path.expanduser(save_path)
        save_path = os.path.abspath(save_path)
        results_save_path = os.path.join(save_path, 'batch_transform', job_name)
        if not os.path.isdir(results_save_path):
            os.makedirs(results_save_path)
        temp_results_save_path = os.path.join(results_save_path, file_name)
        if os.path.isfile(temp_results_save_path):
            logger.warning('File already exists. Will rename the file to avoid overwrite.')
            file_name = rename_file_with_uuid(file_name)
        results_save_path = os.path.join(results_save_path, file_name)
        results_bucket, results_key_prefix = s3_path_to_bucket_prefix(result_path)
        self.sagemaker_session.download_data(
            path=results_save_path,
            bucket=results_bucket,
            key_prefix=results_key_prefix
        )

    def get_batch_transform_job_status(self, job_name=None):
        """
        Get the status of the batch transform job.
        This is useful when the user made an asynchronous call to the `predict()` function

        Parameters
        ----------
        job_name: str
            The name of the job being checked.
            If None, will check the most recent job status.

        Returns
        -------
        str,
        Valid Values: InProgress | Completed | Failed | Stopping | Stopped | NotCreated
        """
        if not job_name:
            job_name = self._batch_transform_jobs.last
        job = self._batch_transform_jobs.get(job_name, None)
        if job:
            return job.get_job_status()
        return 'NotCreated'

    def cleanup_deployment(self):
        """
        Delete endpoint, endpoint configuration and deployed model
        """
        self._delete_endpoint_model()
        self._delete_endpoint()

    def _delete_endpoint(self, delete_endpoint_config=True):
        assert self.endpoint, 'There is no endpoint deployed yet'
        logger.log(20, 'Deleteing endpoint')
        self.endpoint.delete_endpoint(delete_endpoint_config=delete_endpoint_config)
        logger.log(20, 'Endpoint deleted')
        self.endpoint = None

    def _delete_endpoint_model(self):
        assert self.endpoint, 'There is no endpoint deployed yet'
        logger.log(20, 'Deleting endpoint model')
        self.endpoint.delete_model()
        logger.log(20, 'Endpoint model deleted')

    def _download_predictor(self, path, save_path):
        logger.log(20, 'Downloading trained models to local directory')
        predictor_bucket, predictor_key_prefix = s3_path_to_bucket_prefix(path)
        tarball_path = os.path.join(save_path, 'model.tar.gz')
        self.sagemaker_session.download_data(
            path=tarball_path,
            bucket=predictor_bucket,
            key_prefix=predictor_key_prefix,
        )
        logger.log(20, 'Extracting the trained model tarball')
        save_path = os.path.join(save_path, 'AutogluonModels')
        unzip_file(tarball_path, save_path)
        return save_path

    def save(self, silent=False):
        """
        Save the CloudPredictor so that user can later reload the predictor to gain access to deployed endpoint.
        """
        path = self.local_output_path
        predictor_file_name = self.predictor_file_name
        temp_session = self.sagemaker_session
        temp_region = self._region
        self.sagemaker_session = None
        self._region = None
        temp_endpoint = None
        if self.endpoint:
            temp_endpoint = self.endpoint
            self._endpoint_saved = self.endpoint_name
            self.endpoint = None

        save_pkl.save(path=os.path.join(path, predictor_file_name), object=self)
        self.sagemaker_session = temp_session
        self._region = temp_region
        if temp_endpoint:
            self.endpoint = temp_endpoint
            self._endpoint_saved = None
        if not silent:
            logger.log(20, f'{type(self).__name__} saved. To load, use: predictor = {type(self).__name__}.load("{self.local_output_path}")')

    def _load_jobs(self):
        self._fit_job.session = self.sagemaker_session
        for job in self._batch_transform_jobs:
            job.session = self.sagemaker_session

    @classmethod
    def load(cls, path, verbosity=None):
        """
        Load the CloudPredictor

        Parameters
        ----------
        path: str
            The path to directory in which this Predictor was previously saved

        Returns
        -------
        `CloudPredictor` object.
        """
        if verbosity is not None:
            set_logger_verbosity(verbosity, logger=logger)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        predictor: CloudPredictor = load_pkl.load(path=os.path.join(path, cls.predictor_file_name))
        predictor.sagemaker_session = setup_sagemaker_session()
        predictor._region = predictor.sagemaker_session.boto_region_name
        predictor._load_jobs()
        if hasattr(predictor, '_endpoint_saved') and predictor._endpoint_saved:
            predictor.attach_endpoint(predictor._endpoint_saved)
            predictor._endpoint_saved = None
        # TODO: Version compatibility check
        return predictor

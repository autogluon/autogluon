import copy
import os
import sagemaker
from sagemaker import fw_utils
from sagemaker.estimator import Framework, Estimator
from sagemaker.mxnet.estimator import MXNet
from sagemaker import image_uris, vpc_utils
from sagemaker.model import FrameworkModel, Model, DIR_PARAM_NAME, SCRIPT_PARAM_NAME
from sagemaker.predictor import Predictor
from sagemaker.mxnet import MXNetModel
from sagemaker.serializers import CSVSerializer, NumpySerializer
from sagemaker.fw_utils import (
    model_code_key_prefix,
)
from .serializers import ParquetSerializer, MultiModalSerializer, JsonLineSerializer
from .deserializers import PandasDeserializer
from .sagemaker_utils import retrieve_latest_framework_version


# Framework documentation: https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework
class AutoGluonSagemakerEstimator(Estimator):
    def __init__(
        self,
        entry_point,
        region,
        framework_version,
        py_version,
        instance_type,
        source_dir=None,
        hyperparameters=None,
        **kwargs,
    ):
        self.framework_version = framework_version
        self.py_version = py_version
        self.image_uri = image_uris.retrieve(
            "autogluon",
            region=region,
            version=framework_version,
            py_version=py_version,
            image_scope="training",
            instance_type=instance_type,
        )
        super().__init__(
            entry_point=entry_point,
            source_dir=source_dir,
            hyperparameters=hyperparameters,
            instance_type=instance_type,
            image_uri=self.image_uri,
            **kwargs,
        )

    def _configure_distribution(self, distributions):
        return

    def create_model(
        self,
        region,
        framework_version,
        py_version,
        instance_type,
        source_dir=None,
        entry_point=None,
        role=None,
        image_uri=None,
        predictor_cls=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        **kwargs,
    ):
        image_uri = image_uris.retrieve(
            "autogluon",
            region=region,
            version=framework_version,
            py_version=py_version,
            image_scope="inference",
            instance_type=instance_type,
        )
        if predictor_cls is None:

            def predict_wrapper(endpoint, session):
                return Predictor(endpoint, session)

            predictor_cls = predict_wrapper

        role = role or self.role

        if "enable_network_isolation" not in kwargs:
            kwargs["enable_network_isolation"] = self.enable_network_isolation()

        return AutoGluonNonRepackModel(
            image_uri=image_uri,
            source_dir=source_dir,
            entry_point=entry_point,
            model_data=self.model_data,
            role=role,
            vpc_config=self.get_vpc_config(vpc_config_override),
            sagemaker_session=self.sagemaker_session,
            predictor_cls=predictor_cls,
            **kwargs,
        )
        # return super().create_model(
        #     image_uri=image_uri,
        #     source_dir=source_dir,
        #     entry_point=entry_point,
        #     **kwargs
        # )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        init_params = super()._prepare_init_params_from_job_description(job_details, model_channel_name=model_channel_name)
        # This two parameters will not be used, but is required to reattach the job
        init_params['region'] = 'us-east-1'
        init_params['framework_version'] = retrieve_latest_framework_version()
        return init_params


# Documentation for FrameworkModel: https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.FrameworkModel
class AutoGluonSagemakerInferenceModel(Model):
    def __init__(
        self,
        model_data,
        role,
        entry_point,
        region,
        framework_version,
        py_version,
        instance_type,
        **kwargs
    ):
        image_uri = image_uris.retrieve(
            "autogluon",
            region=region,
            version=framework_version,
            py_version=py_version,
            image_scope="inference",
            instance_type=instance_type,
        )
        super().__init__(
            model_data=model_data,
            role=role,
            entry_point=entry_point,
            image_uri=image_uri,
            **kwargs
        )

    def transformer(
        self,
        instance_count,
        instance_type,
        strategy='MultiRecord',
        max_payload=6,  # Maximum size of the payload in a single HTTP request to the container in MB. Will split into multiple batches if a request is more than max_payload
        max_concurrent_transforms=1,  # The maximum number of HTTP requests to be made to each individual transform container at one time.
        accept='application/json',
        assemble_with='Line',
        **kwargs
    ):
        return super().transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            strategy=strategy,
            max_payload=max_payload,
            max_concurrent_transforms=max_concurrent_transforms,
            accept=accept,
            assemble_with=assemble_with,
            **kwargs
        )


class AutoGluonRepackInferenceModel(AutoGluonSagemakerInferenceModel):

    def prepare_container_def(
        self,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
    ):  # pylint: disable=unused-argument
        """Return a dict created by ``sagemaker.container_def()``.

        It is used for deploying this model to a specified instance type.

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.

        Returns:
            dict: A container definition object usable with the CreateModel API.
        """
        deploy_key_prefix = fw_utils.model_code_key_prefix(
            self.key_prefix, self.name, self.image_uri
        )
        deploy_env = copy.deepcopy(self.env)
        self._upload_code(deploy_key_prefix, repack=True)
        deploy_env.update(self._script_mode_env_vars())
        return sagemaker.container_def(
            self.image_uri,
            self.repacked_model_data or self.model_data,
            deploy_env,
            image_config=self.image_config,
        )


class AutoGluonNonRepackInferenceModel(AutoGluonSagemakerInferenceModel):

    def prepare_container_def(
        self,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
    ):  # pylint: disable=unused-argument
        """Return a dict created by ``sagemaker.container_def()``.

        It is used for deploying this model to a specified instance type.

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.

        Returns:
            dict: A container definition object usable with the CreateModel API.
        """
        # c_def = super().prepare_container_def(
        #     instance_type=instance_type,
        #     accelerator_type=accelerator_type,
        #     serverless_inference_config=serverless_inference_config
        # ).copy()
        deploy_env = copy.deepcopy(self.env)
        deploy_env.update(self._script_mode_env_vars())
        deploy_env[SCRIPT_PARAM_NAME.upper()] = os.path.basename(deploy_env[SCRIPT_PARAM_NAME.upper()])
        deploy_env[DIR_PARAM_NAME.upper()] = '/opt/ml/model/code'

        return sagemaker.container_def(
            self.image_uri,
            self.model_data,
            deploy_env,
            image_config=self.image_config,
        )

        # if 'Environment' in c_def and DIR_PARAM_NAME in c_def['Environment']:
        #     c_def['Environment'][DIR_PARAM_NAME] = '/opt/ml/model/code'
        # return sagemaker.container_def(
        #     image_uri=c_def.get('Image', None),
        #     model_data_url=c_def.get('ModelDataUrl', None),
        #     env=c_def.get('Environment', None),
        #     container_mode=c_def.get('Mode', None),
        #     image_config=c_def.get('ImageConfig', None)
        # )


# Predictor documentation: https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
class AutoGluonRealtimePredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=ParquetSerializer(),
            deserializer=PandasDeserializer(),
            **kwargs
        )


class AutoGluonImageRealtimePredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=NumpySerializer(),
            deserializer=PandasDeserializer(),
            **kwargs
        )


class AutoGluonMultiModalRealtimePredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=MultiModalSerializer(),
            deserializer=PandasDeserializer(),
            **kwargs
        )


# Predictor documentation: https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
# SageMaker can only take in csv format for batch transformation because files need to be easily splitable to be batch processed.
class AutoGluonBatchPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=CSVSerializer(),
            **kwargs
        )




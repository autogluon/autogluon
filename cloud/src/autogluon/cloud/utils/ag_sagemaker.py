import sagemaker
from sagemaker.estimator import Framework
from sagemaker import image_uris
from sagemaker.model import FrameworkModel
from sagemaker.predictor import Predictor
from sagemaker.mxnet import MXNetModel
from sagemaker.serializers import CSVSerializer
from sagemaker.fw_utils import (
    model_code_key_prefix,
)
from .serializers import ParquetSerializer, JsonLineSerializer
from .deserializers import PandasDeserializer
from .sagemaker_utils import retrieve_latest_framework_version


# Framework documentation: https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework
class AutoGluonSagemakerEstimator(Framework):
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
            entry_point,
            source_dir,
            hyperparameters,
            instance_type=instance_type,
            image_uri=self.image_uri,
            **kwargs,
        )

    def _configure_distribution(self, distributions):
        return

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=None,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        image_name=None,
        **kwargs,
    ):
        return None

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        init_params = super()._prepare_init_params_from_job_description(job_details, model_channel_name=model_channel_name)
        # This two parameters will not be used, but is required to reattach the job
        init_params['region'] = 'us-east-1'
        init_params['framework_version'] = retrieve_latest_framework_version()
        return init_params


# Predictor documentation: https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
class AutoGluonRealtimePredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=ParquetSerializer(),
            deserializer=PandasDeserializer(),
            **kwargs
        )


# Predictor documentation: https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
# SageMaker can only take in csv format for batch transformation because files need to be easily splitable to  be batch processed.
class AutoGluonBatchPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=CSVSerializer(),
            **kwargs
        )


# Documentation for FrameworkModel: https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.FrameworkModel
class AutoGluonSagemakerInferenceModel(MXNetModel):
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
            model_data,
            role,
            entry_point,
            image_uri=image_uri,
            framework_version="1.8.0",
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

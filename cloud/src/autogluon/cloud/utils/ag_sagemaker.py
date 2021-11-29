import sagemaker
from sagemaker.estimator import Framework
from sagemaker.mxnet import MXNetModel
from sagemaker import image_uris
from sagemaker.model import FrameworkModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.fw_utils import (
    model_code_key_prefix,
)
from .serializers import ParquetSerializer, JsonLineSerializer
from .deserializers import PandasDeserializer
from .sagemaker_utils import retrieve_latest_framework_version


class AutoGluonSagemakerEstimator(Framework):
    def __init__(
        self,
        entry_point,
        region,
        framework_version,
        instance_type,
        image_uri=None,
        source_dir=None,
        hyperparameters=None,
        **kwargs,
    ):
        if not image_uri:
            image_uri = image_uris.retrieve(
                "autogluon",
                region=region,
                version=framework_version,
                py_version="py37",
                image_scope="training",
                instance_type=instance_type,
            )
        super().__init__(
            entry_point,
            source_dir,
            hyperparameters,
            instance_type=instance_type,
            image_uri=image_uri,
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


class AutoGluonRealtimePredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=ParquetSerializer(),
            deserializer=PandasDeserializer(),
            **kwargs
        )


class AutoGluonBatchPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=CSVSerializer(),
            **kwargs
        )


# FIXME: Change to FrameworkModel
class AutoGluonSagemakerInferenceModel(FrameworkModel):
    def __init__(
        self,
        model_data,
        role,
        entry_point,
        region,
        framework_version,
        instance_type,
        **kwargs
    ):
        image_uri = image_uris.retrieve(
            "autogluon",
            region=region,
            version=framework_version,
            py_version="py37",
            image_scope="inference",
            instance_type=instance_type,
        )
        super().__init__(
            model_data,
            image_uri,
            role,
            entry_point,
            **kwargs
        )

    def transformer(
        self,
        instance_count,
        instance_type,
        strategy='MultiRecord',
        max_payload=6,
        max_concurrent_transforms=1,
        accept='application/json',
        assemble_with='Line',
        **kwargs
    ):
        return super().transformer(
            instance_count,
            instance_type,
            strategy=strategy,
            max_payload=max_payload,
            max_concurrent_transforms=max_concurrent_transforms,
            accept=accept,
            assemble_with=assemble_with,
            **kwargs
        )

    def prepare_container_def(self, instance_type=None, accelerator_type=None):
        """Return a container definition with framework configuration.

        Framework configuration is set in model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.

        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """
        deploy_image = self.image_uri
        deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, deploy_image)
        self._upload_code(deploy_key_prefix, True)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())

        return sagemaker.container_def(
            deploy_image, self.repacked_model_data or self.model_data, deploy_env
        )

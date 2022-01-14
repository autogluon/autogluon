import logging
import sagemaker

from abc import ABC, abstractmethod
from ..utils.ag_sagemaker import (
    AutoGluonSagemakerEstimator,
    AutoGluonSagemakerInferenceModel,
)

logger = logging.getLogger(__name__)


class SageMakerJob(ABC):

    def __init__(self, session=None):
        self.session = session or sagemaker.session.Session()
        self._job_name = None

    @classmethod
    @abstractmethod
    def attach(cls, job_name):
        pass

    @property
    @abstractmethod
    def job_name(self):
        return self._job_name

    @property
    @abstractmethod
    def completed(self):
        if not self.job_name:
            return False
        return self.get_job_status() == 'Completed'

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def _get_job_status(self):
        raise NotImplementedError

    @abstractmethod
    def _get_output_path(self):
        raise NotImplementedError

    def get_job_status(self):
        if not self.job_name:
            return 'NotCreated'
        return self._get_job_status()

    def get_output_path(self):
        if not self.completed:
            return None
        return self._get_output_path()

    def __getstate__(self):
        self.session = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


class SageMakerFitJob(SageMakerJob):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._framework_version = None

    @classmethod
    def attach(cls, job_name):
        # FIXME: find a way to recover framework version
        logger.warning('Reattach to a job does not support real-time logging. Logs will be printed once the training job completes')
        obj = cls()
        obj._job_name = job_name
        sagemaker_estimator = AutoGluonSagemakerEstimator.attach(job_name)
        sagemaker_estimator.logs()
        return obj

    @property
    def framework_version(self):
        return self._framework_version

    @property
    def completed(self):
        if not self.job_name:
            return False
        return self.get_job_status() == 'Completed'

    def info(self):
        info = dict(
            name=self.job_name,
            status=self.get_job_status(),
            framework_version=self.framework_version,
            artifact_path=self.get_output_path(),
        )
        return info

    def _get_job_status(self):
        return self.session.describe_training_job(self.job_name)['TrainingJobStatus']

    def _get_output_path(self):
        return self.session.describe_training_job(self.job_name)["ModelArtifacts"]["S3ModelArtifacts"]

    def run(
        self,
        role,
        entry_point,
        region,
        instance_type,
        instance_count,
        volume_size,
        framework_version,
        base_job_name,
        output_path,
        inputs,
        wait,
        job_name,
        autogluon_sagemaker_estimator_kwargs,
        **kwargs
    ):
        sagemaker_estimator = AutoGluonSagemakerEstimator(
            role=role,
            entry_point=entry_point,
            region=region,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            framework_version=framework_version,
            base_job_name=base_job_name,
            output_path=output_path,
            **autogluon_sagemaker_estimator_kwargs
        )
        logger.log(20, f'Start sagemaker training job `{job_name}`')
        try:
            sagemaker_estimator.fit(
                inputs=inputs,
                wait=wait,
                job_name=job_name,
                **kwargs
            )
            self._job_name = job_name
            self._framework_version = framework_version
        except Exception as e:
            logger.error(f'Training failed. Please check sagemaker console training jobs {job_name} for details.')
            raise e


class SageMakerBatchTransformationJob(SageMakerJob):

    @classmethod
    def attach(cls, job_name):
        raise NotImplementedError

    def info(self):
        info = dict(
            name=self.job_name,
            status=self.get_job_status(),
            result_path=self._get_output_path(),
        )
        return info

    def _get_job_status(self):
        return self.session.describe_transform_job(self.job_name)['TransformJobStatus']

    def _get_output_path(self):
        return self.session.describe_transform_job(self.job_name)['TransformOutput']['S3OutputPath']

    def run(
        self,
        model_data,
        role,
        region,
        framework_version,
        instance_count,
        instance_type,
        entry_point,
        predictor_cls,
        output_path,
        test_input,
        job_name,
        split_type,
        content_type,
        wait,
        autogluon_sagemaker_inference_model_kwargs,
        transformer_kwargs,
        **kwargs
    ):
        logger.log(20, 'Creating inference model...')
        model = AutoGluonSagemakerInferenceModel(
            model_data=model_data,
            role=role,
            region=region,
            framework_version=framework_version,
            instance_type=instance_type,
            entry_point=entry_point,
            predictor_cls=predictor_cls,
            **autogluon_sagemaker_inference_model_kwargs
        )
        logger.log(20, 'Inference model created successfully')

        logger.log(20, 'Creating transformer...')
        transformer = model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_path + '/results',
            **transformer_kwargs
        )
        logger.log(20, 'Transformer created successfully')

        try:
            transformer.transform(
                test_input,
                job_name=job_name,
                split_type=split_type,
                content_type=content_type,
                wait=wait,
                **kwargs
            )
            self._job_name = job_name
        except Exception as e:
            transformer.delete_model()
            raise e

        test_data_filename = test_input.split('/')[-1]
        self._recent_batch_transform_results_path = output_path + '/results/' + test_data_filename + '.out'
        if wait:
            transformer.delete_model()
            logger.log(20, f'Predict results have been saved to {self.get_output_path()}')
        else:
            logger.log(20, 'Predict asynchronously. You can use `info()` or `get_job_status()` to check the status.')

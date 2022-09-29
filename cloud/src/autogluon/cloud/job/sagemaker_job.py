import logging
import sagemaker

from abc import ABC, abstractmethod
from ..utils.ag_sagemaker import (
    AutoGluonSagemakerEstimator,
    AutoGluonRepackInferenceModel,
    AutoGluonNonRepackInferenceModel,
)

logger = logging.getLogger(__name__)


class SageMakerJob(ABC):

    def __init__(self, session=None):
        self.session = session or sagemaker.session.Session()
        self._job_name = None

    @classmethod
    @abstractmethod
    def attach(cls, job_name):
        """
        Reattach to a job given its name.

        Parameters:
        -----------
        job_name: str
            Name of the job to be attached.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> dict:
        """
        Give general information about the job.

        Returns:
        ------
        dict
            A dictionary containing the general information about the job.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs):
        """Execute the job"""
        raise NotImplementedError

    @abstractmethod
    def _get_job_status(self):
        raise NotImplementedError

    @abstractmethod
    def _get_output_path(self):
        raise NotImplementedError

    @property
    def job_name(self):
        return self._job_name

    @property
    def completed(self):
        if not self.job_name:
            return False
        return self.get_job_status() == 'Completed'

    def get_job_status(self) -> str:
        """
        Get job status

        Returns:
        --------
        str:
            Valid Values: InProgress | Completed | Failed | Stopping | Stopped | NotCreated
        """
        if not self.job_name:
            return 'NotCreated'
        return self._get_job_status()

    def get_output_path(self):
        """
        Get the output path of the job generated artifacts if any.

        Returns:
        --------
        str:
            Output path of the job generated artifacts if any.
            If no artifact, return None
        """
        if not self.completed:
            return None
        return self._get_output_path()

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        state_dict['session'] = None
        return state_dict

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
        py_version,
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
            py_version=py_version,
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._output_filename = None

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
        return self.session.describe_transform_job(self.job_name)['TransformOutput']['S3OutputPath'] + '/' + self._output_filename

    def run(
        self,
        model_data,
        role,
        region,
        framework_version,
        py_version,
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
        model_kwargs,
        transformer_kwargs,
        repack_model=False,
        **kwargs
    ):
        if repack_model:
            model_cls = AutoGluonRepackInferenceModel
        else:
            model_cls = AutoGluonNonRepackInferenceModel
        logger.log(20, 'Creating inference model...')
        model = model_cls(
            model_data=model_data,
            role=role,
            region=region,
            framework_version=framework_version,
            py_version=py_version,
            instance_type=instance_type,
            entry_point=entry_point,
            predictor_cls=predictor_cls,
            **model_kwargs
        )
        logger.log(20, 'Inference model created successfully')
        logger.log(20, 'Creating transformer...')
        transformer = model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_path,
            **transformer_kwargs
        )
        logger.log(20, 'Transformer created successfully')

        try:
            logger.log(20, 'Transforming')
            transformer.transform(
                test_input,
                job_name=job_name,
                split_type=split_type,
                content_type=content_type,
                wait=wait,
                **kwargs
            )
            self._job_name = job_name
            logger.log(20, 'Transform done')
        except Exception as e:
            transformer.delete_model()
            raise e

        self._output_filename = test_input.split('/')[-1] + '.out'

        if wait:
            transformer.delete_model()
            logger.log(20, f'Predict results have been saved to {self.get_output_path()}')
        else:
            logger.log(20, 'Predict asynchronously. You can use `info()` or `get_job_status()` to check the status.')

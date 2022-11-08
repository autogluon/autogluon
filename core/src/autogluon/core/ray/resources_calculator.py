import logging
import math

from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class ResourceCalculator(ABC):

    @property
    @abstractmethod
    def calc_type(self):
        """Type of the resource calculator"""
        raise NotImplementedError

    @abstractmethod
    def get_resources_per_job(self, **kwargs) -> dict:
        """Calculate resources per trial and return additional info"""
        raise NotImplementedError

    def wrap_resources_per_job_into_placement_group(self, resources_per_job):
        """
        When doing parallel training inside parallel trials, Ray requires to provide placement group for resource scheduling
        We wrap a group where the resource requirement is 0 because the trial only spread the task and doesn't require too much resources.
        """
        from ray import tune
        num_cpus = resources_per_job.get('cpu', 0)
        num_gpus = resources_per_job.get('gpu', 0)
        return tune.PlacementGroupFactory([{'CPU': 0.0}] + [{'CPU': num_cpus, 'GPU': num_gpus}])


class CpuResourceCalculator(ResourceCalculator):

    @property
    def calc_type(self):
        return 'cpu'

    def get_resources_per_job(
        self,
        total_num_cpus,
        num_jobs,
        minimum_cpu_per_job,
        model_estimate_memory_usage=None,
        wrap_resources_per_job_into_placement_group=False,
        user_resources_per_job=None,
        **kwargs,
    ):
        if user_resources_per_job is not None:
            cpu_per_job = user_resources_per_job.get('num_cpus', 0)
            assert cpu_per_job <= total_num_cpus, \
                f"Detected model level cpu requirement = {cpu_per_job} > total cpu granted to AG predictor = {total_num_cpus}"
            assert cpu_per_job >= minimum_cpu_per_job, \
                f"The model requires minimum cpu {minimum_cpu_per_job}, but you only specified {cpu_per_job}"
            num_parallel_jobs = total_num_cpus // cpu_per_job
            batches = math.ceil(num_jobs / num_parallel_jobs)
        else:
            cpu_per_job = max(minimum_cpu_per_job, int(total_num_cpus // num_jobs))
            max_jobs_in_parallel_memory = num_jobs

            if model_estimate_memory_usage is not None:
                import psutil
                mem_available = psutil.virtual_memory().available
                # calculate how many jobs can run in parallel given memory available
                max_jobs_in_parallel_memory = max(1, int(mem_available // model_estimate_memory_usage))
            num_parallel_jobs = min(num_jobs, total_num_cpus // cpu_per_job, max_jobs_in_parallel_memory)
            if num_parallel_jobs == 0:
                error_msg = ('Cannot train model with provided resources! '
                                f'num_cpus=={total_num_cpus} | '
                                f'min_cpus=={minimum_cpu_per_job}')
                if model_estimate_memory_usage is not None:
                    error_msg += (
                        f' | mem_available=={mem_available} | '
                        f'model_estimate_memory_usage=={model_estimate_memory_usage}'
                    )
                raise AssertionError(error_msg)
            cpu_per_job = int(total_num_cpus // num_parallel_jobs)  # update cpu_per_job in case memory is not enough and can use more cores for each job
            batches = math.ceil(num_jobs / num_parallel_jobs)

        resources_per_job = dict(cpu=cpu_per_job)
        if wrap_resources_per_job_into_placement_group:
            resources_per_job = self.wrap_resources_per_job_into_placement_group(resources_per_job)

        resources_info = dict(
            resources_per_job=resources_per_job,
            num_parallel_jobs=num_parallel_jobs,
            batches=batches,
            cpu_per_job=cpu_per_job
        )
        logger.log(10, f'Resources info for {self.__class__.__name__}: {resources_info}')

        return resources_info


class GpuResourceCalculator(ResourceCalculator):

    @property
    def calc_type(self):
        return 'gpu'

    def get_resources_per_job(
        self,
        total_num_cpus,
        total_num_gpus,
        num_jobs,
        minimum_cpu_per_job,
        minimum_gpu_per_job,
        wrap_resources_per_job_into_placement_group=False,
        user_resources_per_job=None,
        **kwargs,
    ):
        if user_resources_per_job is not None:
            cpu_per_job = user_resources_per_job.get('num_cpus', minimum_cpu_per_job)
            assert cpu_per_job <= total_num_cpus, \
                f"Detected model level cpu requirement = {cpu_per_job} > total cpu granted to AG predictor = {total_num_cpus}"
            assert cpu_per_job >= minimum_cpu_per_job, \
                f"The model requires minimum cpu {minimum_cpu_per_job}, but you only specified {cpu_per_job}"
            gpu_per_job = user_resources_per_job.get('num_gpus', minimum_gpu_per_job)
            assert gpu_per_job <= total_num_gpus, \
                f"Detected model level gpu requirement = {gpu_per_job} > total cpu granted to AG predictor = {total_num_gpus}"
            assert gpu_per_job >= minimum_gpu_per_job, \
                f"The model requires minimum gpu {minimum_gpu_per_job}, but you only specified {gpu_per_job}"
            num_parallel_jobs = min(total_num_cpus // cpu_per_job, total_num_gpus / gpu_per_job)
            batches = math.ceil(num_jobs / num_parallel_jobs)
        else:
            cpu_per_job = max(minimum_cpu_per_job, int(total_num_cpus // num_jobs))
            gpu_per_job = max(minimum_gpu_per_job, total_num_gpus / num_jobs)
            num_parallel_jobs = num_jobs
            if cpu_per_job:
                num_parallel_jobs = min(num_parallel_jobs, total_num_cpus // cpu_per_job)
            if gpu_per_job:
                num_parallel_jobs = min(num_parallel_jobs, total_num_gpus // gpu_per_job)
            if num_parallel_jobs == 0:
                raise AssertionError('Cannot train model with provided resources! '
                                        f'(num_cpus, num_gpus)==({total_num_cpus}, {total_num_gpus}) | '
                                        f'(min_cpus, min_gpus)==({minimum_cpu_per_job}, {minimum_gpu_per_job})')
            cpu_per_job = int(total_num_cpus // num_parallel_jobs)
            gpu_per_job = total_num_gpus / num_parallel_jobs
            batches = math.ceil(num_jobs / num_parallel_jobs)

        resources_per_job = dict(cpu=cpu_per_job, gpu=gpu_per_job)
        if wrap_resources_per_job_into_placement_group:
            resources_per_job = self.wrap_resources_per_job_into_placement_group(resources_per_job)

        resources_info = dict(
            resources_per_job=resources_per_job,
            num_parallel_jobs=num_parallel_jobs,
            batches=batches,
            cpu_per_job=cpu_per_job,
            gpu_per_job=gpu_per_job,
        )
        logger.log(10, f'Resources info for {self.__class__.__name__}: {resources_info}')

        return resources_info


class NonParallelGpuResourceCalculator(ResourceCalculator):
    """
    This calculator will only assign < 1 gpu to each job because some job cannot be parallelized
    """

    @property
    def calc_type(self):
        return 'non_parallel_gpu'

    def get_resources_per_job(
        self,
        total_num_cpus,
        total_num_gpus,
        num_jobs,
        minimum_cpu_per_job,
        minimum_gpu_per_job,
        wrap_resources_per_job_into_placement_group=False,
        **kwargs,
    ):
        assert 0 < minimum_gpu_per_job <= 1, f'{self.__class__.__name__} only supports assigning < 1 gpu to each job' 
        cpu_per_job = max(minimum_cpu_per_job, int(total_num_cpus // num_jobs))
        gpu_per_job = min(minimum_gpu_per_job, 1)
        num_parallel_jobs = num_jobs
        if cpu_per_job:
            num_parallel_jobs = min(num_parallel_jobs, total_num_cpus // cpu_per_job)
        if gpu_per_job:
            num_parallel_jobs = min(num_parallel_jobs, total_num_gpus // gpu_per_job)
        if num_parallel_jobs == 0:
            raise AssertionError('Cannot train model with provided resources! '
                                 f'(num_cpus, num_gpus)==({total_num_cpus}, {total_num_gpus}) | '
                                 f'(min_cpus, min_gpus)==({cpu_per_job}, {gpu_per_job})')
        cpu_per_job = int(total_num_cpus // num_parallel_jobs)
        gpu_per_job = min(1, total_num_gpus / num_parallel_jobs)

        resources_per_job = dict(cpu=cpu_per_job, gpu=gpu_per_job)
        if wrap_resources_per_job_into_placement_group:
            resources_per_job = self.wrap_resources_per_job_into_placement_group(resources_per_job)
        batches = math.ceil(num_jobs / num_parallel_jobs)

        resources_info = dict(
            resources_per_job=resources_per_job,
            num_parallel_jobs=num_parallel_jobs,
            batches=batches,
            cpu_per_job=cpu_per_job,
            gpu_per_job=gpu_per_job,
        )
        logger.log(10, f'Resources info for {self.__class__.__name__}: {resources_info}')

        return resources_info


class RayLightningCpuResourceCalculator(ResourceCalculator):

    @property
    def calc_type(self):
        return 'ray_lightning_cpu'

    def get_resources_per_job(
        self,
        total_num_cpus,
        num_jobs,
        minimum_cpu_per_job,
        model_estimate_memory_usage=None,
        **kwargs,
    ):
        from ray_lightning.tune import get_tune_resources
        # TODO: for cpu case, is it better to have more workers or more cpus per worker?
        cpu_per_job = max(minimum_cpu_per_job, total_num_cpus // num_jobs)
        max_jobs_in_parallel_memory = num_jobs
        if model_estimate_memory_usage is not None:
            import psutil
            mem_available = psutil.virtual_memory().available
            # calculate how many jobs can run in parallel given memory available
            max_jobs_in_parallel_memory = max(1, int(mem_available // model_estimate_memory_usage))
        num_parallel_jobs = min(num_jobs, total_num_cpus // cpu_per_job, max_jobs_in_parallel_memory)
        if num_parallel_jobs == 0:
            error_msg = ('Cannot train model with provided resources! '
                         f'num_cpus=={total_num_cpus} | '
                         f'min_cpus=={minimum_cpu_per_job}')
            if model_estimate_memory_usage is not None:
                error_msg += (
                    f' | mem_available=={mem_available} | '
                    f'model_estimate_memory_usage=={model_estimate_memory_usage}'
                )
            raise AssertionError(error_msg)
        num_workers = max(minimum_cpu_per_job, cpu_per_job - 1)  # 1 cpu for master process
        cpu_per_worker = 1
        resources_per_job = get_tune_resources(
            num_workers=num_workers,
            num_cpus_per_worker=cpu_per_worker,
            use_gpu=False
        )
        batches = math.ceil(num_jobs / num_parallel_jobs)

        resources_info = dict(
            resources_per_job=resources_per_job,
            num_parallel_jobs=num_parallel_jobs,
            batches=batches,
            cpu_per_job=cpu_per_job,
            num_workers=num_workers,
        )
        logger.log(10, f'Resources info for {self.__class__.__name__}: {resources_info}')

        return resources_info


class RayLightningGpuResourceCalculator(ResourceCalculator):

    @property
    def calc_type(self):
        return 'ray_lightning_gpu'

    def get_resources_per_job(
        self,
        total_num_cpus,
        total_num_gpus,
        num_jobs,
        minimum_cpu_per_job,
        minimum_gpu_per_job,
        **kwargs,
    ):
        from ray_lightning.tune import get_tune_resources
        # Ray Tune requires 1 additional CPU per trial to use for the Trainable driver. 
        # So the actual number of cpu resources each trial requires is num_workers * num_cpus_per_worker + 1
        # Each ray worker will reserve 1 gpu
        # The num_workers in ray stands for worker process to train the model
        # The num_workers in AutoMM stands for worker process to load data
        gpu_per_job = max(int(minimum_gpu_per_job), total_num_gpus // num_jobs)
        num_workers = gpu_per_job  # each worker uses 1 gpu
        num_parallel_jobs = min(num_jobs, total_num_gpus // gpu_per_job)
        if num_parallel_jobs == 0:
            raise AssertionError('Cannot train model with provided resources! '
                                 f'(num_cpus, num_gpus)==({total_num_cpus}, {total_num_gpus}) | '
                                 f'(min_cpus, min_gpus)==({minimum_cpu_per_job}, {minimum_gpu_per_job})')
        num_cpus = (total_num_cpus - num_parallel_jobs)  # reserve cpus for the master process
        assert num_cpus > 0
        cpu_per_job = max(minimum_cpu_per_job, num_cpus // num_parallel_jobs)
        cpu_per_worker = max(1, cpu_per_job // num_workers)
        resources_per_job = get_tune_resources(
            num_workers=num_workers,
            num_cpus_per_worker=cpu_per_worker,
            use_gpu=True
        )
        batches = math.ceil(num_jobs / num_parallel_jobs)

        resources_info = dict(
            resources_per_job=resources_per_job,
            num_parallel_jobs=num_parallel_jobs,
            batches=batches,
            cpu_per_job=cpu_per_job,
            gpu_per_job=gpu_per_job,
            num_workers=num_workers,
            cpu_per_worker=cpu_per_worker,
        )
        logger.log(10, f'Resources info for {self.__class__.__name__}: {resources_info}')

        return resources_info


class ResourceCalculatorFactory:

    __supported_calculators = [
        CpuResourceCalculator,
        GpuResourceCalculator,
        NonParallelGpuResourceCalculator,
        RayLightningCpuResourceCalculator,
        RayLightningGpuResourceCalculator
    ]
    __type_to_calculator = {cls().calc_type: cls for cls in __supported_calculators}

    @staticmethod
    def get_resource_calculator(calculator_type: str) -> ResourceCalculator:
        """Return the resource calculator"""
        assert calculator_type in ResourceCalculatorFactory.__type_to_calculator, f'{calculator_type} not supported'
        return ResourceCalculatorFactory.__type_to_calculator[calculator_type]()

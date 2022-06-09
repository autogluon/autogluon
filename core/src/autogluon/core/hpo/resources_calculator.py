import psutil

from abc import ABC, abstractmethod
from ray_lightning.tune import get_tune_resources


class ResourceCalculator(ABC):
    
    @property
    @abstractmethod
    def calc_type(self):
        """Type of the resource calculator"""
        raise NotImplementedError
    
    @abstractmethod
    def get_resources_per_trial(self, **kwargs) -> dict:
        """Calculate resources per trial and return additional info"""
        raise NotImplementedError
    

class CpuResourceCalculator(ResourceCalculator):
    
    @property
    def calc_type(self):
        return 'cpu'
    
    def get_resources_per_trial(
        self,
        total_num_cpus,
        num_samples,
        minimum_cpu_per_trial,
        model_estimate_memory_usage=None,
        **kwargs,
    ):
        cpu_per_job = max(minimum_cpu_per_trial, int(total_num_cpus // num_samples))
        max_jobs_in_parallel_memory = num_samples

        if model_estimate_memory_usage is not None:
            mem_available = psutil.virtual_memory().available
            # calculate how many jobs can run in parallel given memory available
            max_jobs_in_parallel_memory = max(1, int(mem_available // model_estimate_memory_usage))
        num_parallel_jobs = min(num_samples, total_num_cpus // cpu_per_job, max_jobs_in_parallel_memory)
        cpu_per_job = max(minimum_cpu_per_trial, int(total_num_cpus // num_parallel_jobs))  # update cpu_per_job in case memory is not enough and can use more cores for each job
        resources_per_trial = dict(cpu=cpu_per_job)
        
        return dict(
                    resources_per_trial=resources_per_trial,
                    num_parallel_jobs=num_parallel_jobs,
                    cpu_per_job=cpu_per_job
                )
    
    
class GpuResourceCalculator(ResourceCalculator):
    
    @property
    def calc_type(self):
        return 'gpu'
    
    def get_resources_per_trial(
        self,
        total_num_cpus,
        total_num_gpus,
        num_samples,
        minimum_cpu_per_trial,
        minimum_gpu_per_trial,
        **kwargs,
    ):
        gpu_per_job = max(minimum_gpu_per_trial, total_num_gpus // num_samples)
        num_parallel_jobs = total_num_gpus // gpu_per_job  # num_parallel_jobs purely based on gpu
        cpu_per_job = max(minimum_cpu_per_trial, total_num_cpus // num_parallel_jobs)
        num_parallel_jobs = min(num_parallel_jobs, total_num_cpus // cpu_per_job)  # update num_parallel_jobs in case cpu is not enough
        gpu_per_job = max(minimum_gpu_per_trial, total_num_gpus // num_parallel_jobs)  # update gpu_per_job in case cpu was the bottleneck
        resources_per_trial = dict(cpu=cpu_per_job, gpu=gpu_per_job)
        
        return dict(
                    resources_per_trial=resources_per_trial,
                    num_parallel_jobs=num_parallel_jobs,
                    cpu_per_job=cpu_per_job,
                    gpu_per_job=gpu_per_job,
                )
    
    
class RayLightningCpuResourceCalculator(ResourceCalculator):
    
    @property
    def calc_type(self):
        return 'ray_lightning_cpu'
    
    def get_resources_per_trial(
        self,
        total_num_cpus,
        num_samples,
        minimum_cpu_per_trial,
        model_estimate_memory_usage=None,
        **kwargs,
    ):
        # TODO: for cpu case, is it better to have more workers or more cpus per worker?
        cpu_per_job = max(minimum_cpu_per_trial, total_num_cpus // num_samples)
        max_jobs_in_parallel_memory = num_samples
        if model_estimate_memory_usage is not None:
            mem_available = psutil.virtual_memory().available
            # calculate how many jobs can run in parallel given memory available
            max_jobs_in_parallel_memory = max(1, int(mem_available // model_estimate_memory_usage))
        num_parallel_jobs = min(num_samples, total_num_cpus // cpu_per_job, max_jobs_in_parallel_memory)
        num_workers = max(minimum_cpu_per_trial, cpu_per_job - 1)  # 1 cpu for master process
        cpu_per_worker = 1
        resources_per_trial = get_tune_resources(
            num_workers=num_workers,
            num_cpus_per_worker=cpu_per_worker,
            use_gpu=False
        )
        
        return dict(
            resources_per_trial=resources_per_trial,
            num_parallel_jobs=num_parallel_jobs,
            cpu_per_job=cpu_per_job,
            num_workers=num_workers,
        )
        
        
class RayLightningGpuResourceCalculator(ResourceCalculator):
    
    @property
    def calc_type(self):
        return 'ray_lightning_gpu'
    
    def get_resources_per_trial(
        self,
        total_num_cpus,
        total_num_gpus,
        num_samples,
        minimum_cpu_per_trial,
        minimum_gpu_per_trial,
        **kwargs,
    ):
        # Ray Tune requires 1 additional CPU per trial to use for the Trainable driver. 
        # So the actual number of cpu resources each trial requires is num_workers * num_cpus_per_worker + 1
        # Each ray worker will reserve 1 gpu
        # The num_workers in ray stands for worker process to train the model
        # The num_workers in AutoMM stands for worker process to load data
        gpu_per_job = max(int(minimum_gpu_per_trial), total_num_gpus // num_samples)
        num_workers = gpu_per_job  # each worker uses 1 gpu
        num_parallel_jobs = min(num_samples, total_num_gpus // gpu_per_job)
        num_cpus = (total_num_cpus - num_parallel_jobs)  # reserve cpus for the master process
        assert num_cpus > 0
        cpu_per_job = max(minimum_cpu_per_trial, num_cpus // num_parallel_jobs)
        cpu_per_worker = max(1, cpu_per_job // num_workers)
        resources_per_trial = get_tune_resources(
            num_workers=num_workers,
            num_cpus_per_worker=cpu_per_worker,
            use_gpu=True
        )
        
        return dict(
            resources_per_trial=resources_per_trial,
            num_parallel_jobs=num_parallel_jobs,
            cpu_per_job=cpu_per_job,
            gpu_per_job=gpu_per_job,
            num_workers=num_workers,
            cpu_per_worker=cpu_per_worker,
        )


class ResourceCalculatorFactory:
    
    __supported_calculators = [
        CpuResourceCalculator,
        GpuResourceCalculator,
        RayLightningCpuResourceCalculator,
        RayLightningGpuResourceCalculator
    ]
    __type_to_calculator = {cls().calc_type: cls for cls in __supported_calculators}

    @staticmethod
    def get_resource_calculator(calculator_type: str) -> ResourceCalculator:
        """Return the resource calculator"""
        assert calculator_type in ResourceCalculatorFactory.__type_to_calculator, f'{calculator_type} not supported'
        return ResourceCalculatorFactory.__type_to_calculator[calculator_type]()

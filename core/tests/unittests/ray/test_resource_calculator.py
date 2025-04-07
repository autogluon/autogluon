import pytest

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.ray.resources_calculator import (
    CpuResourceCalculator,
    GpuResourceCalculator,
    NonParallelGpuResourceCalculator,
    ResourceCalculatorFactory,
)


def test_cpu_calculator_no_bottleneck():
    num_cpus = 32
    num_jobs = 20

    calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type="cpu")
    assert type(calculator) == CpuResourceCalculator

    resources_info = calculator.get_resources_per_job(
        total_num_cpus=num_cpus,
        num_jobs=num_jobs,
        minimum_cpu_per_job=4,  # allows 8 jobs to run in parallel
    )

    expected_resources_per_trial = dict(
        cpu=4,
    )
    expected_num_parallel_jobs = 8
    expected_batches = 3

    assert expected_resources_per_trial == resources_info["resources_per_job"]
    assert expected_num_parallel_jobs == resources_info["num_parallel_jobs"]
    assert expected_batches == resources_info["batches"]


def test_cpu_calculator_mem_bottleneck():
    num_cpus = 32
    num_jobs = 20
    mem_available = ResourceManager.get_available_virtual_mem()
    model_estimate_memory_usage = mem_available // 2.5  # allows 2 jobs to run in parallel

    calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type="cpu")
    assert type(calculator) == CpuResourceCalculator

    resources_info = calculator.get_resources_per_job(
        total_num_cpus=num_cpus,
        num_jobs=num_jobs,
        model_estimate_memory_usage=model_estimate_memory_usage,
        minimum_cpu_per_job=4,  # allows 8 jobs to run in parallel
    )

    expected_num_parallel_jobs = (
        2  # even user wants to run 20 jobs in prallel, cpu can run 8 jobs in parallel, memory only allows for 2 jobs
    )
    expected_resources_per_trial = dict(
        cpu=16,
    )
    expected_batches = 10

    assert expected_resources_per_trial == resources_info["resources_per_job"]
    assert expected_num_parallel_jobs == resources_info["num_parallel_jobs"]
    assert expected_batches == resources_info["batches"]


def test_gpu_calculator_no_bottleneck():
    num_cpus = 32
    num_gpus = 4
    num_jobs = 20

    calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type="gpu")
    assert type(calculator) == GpuResourceCalculator

    resources_info = calculator.get_resources_per_job(
        total_num_cpus=num_cpus,
        total_num_gpus=num_gpus,
        num_jobs=num_jobs,
        minimum_cpu_per_job=1,  # allows 32 jobs to run in parallel
        minimum_gpu_per_job=0.5,  # allows 8 jobs to run in parallel
    )

    expected_num_parallel_jobs = 8
    expected_resources_per_trial = dict(
        cpu=4,
        gpu=0.5,
    )
    expected_batches = 3

    assert expected_resources_per_trial == resources_info["resources_per_job"]
    assert expected_num_parallel_jobs == resources_info["num_parallel_jobs"]
    assert expected_batches == resources_info["batches"]


def test_gpu_calculator_cpu_bottleneck():
    num_cpus = 4
    num_gpus = 4
    num_jobs = 20

    calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type="gpu")
    assert type(calculator) == GpuResourceCalculator

    resources_info = calculator.get_resources_per_job(
        total_num_cpus=num_cpus,
        total_num_gpus=num_gpus,
        num_jobs=num_jobs,
        minimum_cpu_per_job=1,  # allows 4 jobs to run in parallel
        minimum_gpu_per_job=0.5,  # allows 8 jobs to run in parallel
    )

    expected_num_parallel_jobs = 4
    expected_resources_per_trial = dict(
        cpu=1,
        gpu=1,
    )
    expected_batches = 5

    assert expected_resources_per_trial == resources_info["resources_per_job"]
    assert expected_num_parallel_jobs == resources_info["num_parallel_jobs"]
    assert expected_batches == resources_info["batches"]


def test_non_parallel_gpu_calculator():
    num_cpus = 32
    num_gpus = 4
    num_jobs = 2

    calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type="non_parallel_gpu")
    assert type(calculator) == NonParallelGpuResourceCalculator

    resources_info = calculator.get_resources_per_job(
        total_num_cpus=num_cpus,
        total_num_gpus=num_gpus,
        num_jobs=num_jobs,
        minimum_cpu_per_job=1,
        minimum_gpu_per_job=1,
    )

    expected_num_parallel_jobs = 2
    expected_resources_per_trial = dict(
        cpu=16,
        gpu=1,
    )
    expected_batches = 1

    assert expected_resources_per_trial == resources_info["resources_per_job"]
    assert expected_num_parallel_jobs == resources_info["num_parallel_jobs"]
    assert expected_batches == resources_info["batches"]


@pytest.mark.parametrize("calculator_type", ["cpu", "gpu", "non_parallel_gpu"])
def test_resource_not_enough(calculator_type):
    num_cpus = 0
    num_gpus = 0
    num_jobs = 20

    calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type=calculator_type)
    with pytest.raises(Exception, match=r"Cannot train model with provided resources! .*") as e_info:
        resources_info = calculator.get_resources_per_job(
            total_num_cpus=num_cpus,
            total_num_gpus=num_gpus,
            num_jobs=num_jobs,
            minimum_cpu_per_job=1,
            minimum_gpu_per_job=1,
        )

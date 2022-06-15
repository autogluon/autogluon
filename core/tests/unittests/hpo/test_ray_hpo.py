import psutil
import pytest
import tempfile

from autogluon.core.hpo import RayTuneAdapter, TabularRayTuneAdapter, AutommRayTuneAdapter, run
from autogluon.core.utils import get_cpu_count, get_gpu_count_all
from autogluon.core.hpo.constants import SEARCHER_PRESETS, SCHEDULER_PRESETS
from ray import tune


class DummyAdapter(RayTuneAdapter):
    
    supported_searchers = list(SEARCHER_PRESETS.keys())
    supported_schedulers = list(SCHEDULER_PRESETS.keys())
    
    def get_resources_per_trial(self, total_resources, num_samples, **kwargs):
        return {'cpu':1}
    
    def trainable_args_update_method(self, trainable_args):
        return {}


DUMMY_SEARCH_SPACE = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 20)}


def _dummy_objective(x, a, b):
    return a * (x ** 0.5) + b


def _dummy_trainable(config):

    for x in range(20):
        score = _dummy_objective(x, config["a"], config["b"])

        tune.report(score=score)
        
        
def test_tabular_resource_allocation_no_gpu_no_bottleneck():
    num_cpus = 32
    num_gpus = 0
    num_trials = 100
    
    adapter = TabularRayTuneAdapter()
    total_resources = dict(num_cpus=num_cpus, num_gpus=num_gpus,)
    resources_per_trial = adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_trials,
        minimum_cpu_per_trial=1,  # allows 32 trials to run in parallel
    )

    expected_num_parallel_jobs = 32  # even user wants to run 1000 jobs in prallel, cpu can run 4 jobs in parallel, memory only allows for 2 jobs
    expected_resources_per_trial = dict(
        cpu = 1,
    )

    assert expected_resources_per_trial == resources_per_trial
    assert expected_num_parallel_jobs ==  adapter.num_parallel_jobs


def test_tabular_resource_allocation_no_gpu_mem_bottleneck():
    num_cpus = 32
    num_gpus = 0
    mem_available = psutil.virtual_memory().available
    num_trials = 100
    model_estimate_memory_usage = mem_available // 2.5  # allows 2 trials to run in parallel
    
    adapter = TabularRayTuneAdapter()
    total_resources = dict(num_cpus=num_cpus, num_gpus=num_gpus,)
    resources_per_trial = adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_trials,
        model_estimate_memory_usage=model_estimate_memory_usage,
        minimum_cpu_per_trial=1  # allows 32 trials to run in parallel
    )

    expected_num_parallel_jobs = 2  # even user wants to run 1000 jobs in prallel, cpu can run 4 jobs in parallel, memory only allows for 2 jobs
    expected_resources_per_trial = dict(
        cpu = 16,
    )

    assert expected_resources_per_trial == resources_per_trial
    assert expected_num_parallel_jobs ==  adapter.num_parallel_jobs
    

def test_tabular_resource_allocation_with_gpu_no_bottleneck():
    num_cpus = 32
    num_gpus = 4
    num_trials = 100
    
    adapter = TabularRayTuneAdapter()
    total_resources = dict(num_cpus=num_cpus, num_gpus=num_gpus,)
    resources_per_trial = adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_trials,
        minimum_cpu_per_trial=1,  # allows 32 trials to run in parallel
        minimum_gpu_per_trial=0.5,  # allows 8 trials to run in parallel
    )

    expected_num_parallel_jobs = 8
    expected_resources_per_trial = dict(
        cpu = 4,
        gpu = 0.5,
    )

    assert expected_resources_per_trial == resources_per_trial
    assert expected_num_parallel_jobs ==  adapter.num_parallel_jobs
    
    
def test_tabular_resource_allocation_with_gpu_cpu_bottleneck():
    num_cpus = 4
    num_gpus = 4
    num_trials = 100
    
    adapter = TabularRayTuneAdapter()
    total_resources = dict(num_cpus=num_cpus, num_gpus=num_gpus,)
    resources_per_trial = adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_trials,
        minimum_cpu_per_trial=1,  # allows 4 trials to run in parallel
        minimum_gpu_per_trial=0.5,  # allows 8 trials to run in parallel
    )

    expected_num_parallel_jobs = 4
    expected_resources_per_trial = dict(
        cpu = 1,
        gpu = 1,
    )

    assert expected_resources_per_trial == resources_per_trial
    assert expected_num_parallel_jobs ==  adapter.num_parallel_jobs


def test_automm_resource_allocation():
    num_cpus = get_cpu_count()
    num_gpus = get_gpu_count_all()
    num_trials = 1  # TODO: update to more trials when CI supports multiple GPUs
    
    adapter = AutommRayTuneAdapter()
    total_resources = dict(num_cpus=num_cpus, num_gpus=num_gpus,)
    resources_per_trial = adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_trials,
    )
    
    expected_num_parallel_jobs = 1
    # For cpu, each trial uses 1 cpu for the master process, and worker process can split the rest
    expected_resources_per_trial = dict(
        cpu = (num_cpus - expected_num_parallel_jobs) // expected_num_parallel_jobs + expected_num_parallel_jobs,
        gpu = num_gpus / expected_num_parallel_jobs,
    )
    
    # assert expected_resources_per_trial['gpu'] == resources_per_trial.required_resources['GPU']
    expected_resources_per_trial == resources_per_trial.required_resources
    assert expected_num_parallel_jobs ==  adapter.num_parallel_jobs
    

def test_invalid_searcher():
    hyperparameter_tune_kwargs = dict(
        searcher='abc',
        scheduler='FIFO',
        num_trials=1,
    )
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=DUMMY_SEARCH_SPACE,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric='score',
                mode='min',
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
                stop={"training_iteration": 20},
            )
    
def test_invalid_scheduler():
    hyperparameter_tune_kwargs = dict(
        searcher='random',
        scheduler='abc',
        num_trials=1,
    )
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=DUMMY_SEARCH_SPACE,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric='score',
                mode='min',
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
                stop={"training_iteration": 20},
            )
            
            
def test_invalid_preset():
    hyperparameter_tune_kwargs = 'abc'
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=DUMMY_SEARCH_SPACE,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric='score',
                mode='min',
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
                stop={"training_iteration": 20},
            )
            
            
def test_empty_search_space():
    hyperparameter_tune_kwargs = dict(
        searcher='random',
        scheduler='FIFO',
        num_trials=1,
    )
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=dict(),
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric='score',
                mode='min',
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
                stop={"training_iteration": 20},
            )


@pytest.mark.parametrize('searcher', list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize('scheduler', list(SCHEDULER_PRESETS.keys()))
def test_run(searcher, scheduler):
    hyperparameter_tune_kwargs = dict(
        searcher=searcher,
        scheduler=scheduler,
        num_trials=2,
    )
    with tempfile.TemporaryDirectory() as root:
        analysis = run(
            trainable=_dummy_trainable,
            trainable_args=dict(),
            search_space=DUMMY_SEARCH_SPACE,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            metric='score',
            mode='min',
            save_dir=root,
            ray_tune_adapter=DummyAdapter(),
            stop={"training_iteration": 20},
        )
        assert analysis is not None

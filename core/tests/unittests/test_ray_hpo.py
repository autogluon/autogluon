import psutil
import pytest
import tempfile

from autogluon.core.hpo import RayTuneAdapter, TabularRayTuneAdapter, AutommRayTuneAdapter, run
from autogluon.core.utils import get_cpu_count, get_gpu_count_all
from autogluon.core.hpo.ray_hpo import searcher_presets, scheduler_presets
from ray import tune


class DummyAdapter(RayTuneAdapter):
    
    def get_resources_per_trial(total_resources, num_samples, **kwargs):
        return {}
    
    def trainable_args_update_method(trainable_args):
        return {}


DUMMY_SEARCH_SPACE = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 20)}


def _dummy_objective(x, a, b):
    return a * (x ** 0.5) + b


def _dummy_trainable(config):

    for x in range(20):
        score = _dummy_objective(x, config["a"], config["b"])

        tune.report(score=score)


def test_tabular_resource_allocation():
    num_cpus = get_cpu_count(logical=False)
    num_gpus = get_gpu_count_all()
    mem_available = psutil.virtual_memory().available
    num_trials = num_cpus // 4  # can run 4 jobs in parallel
    model_estimate_memory_usage = mem_available // 2.5  # can run 2 jobs in parallel
    
    adapter = TabularRayTuneAdapter()
    total_resources = dict(num_cpus=num_cpus, num_gpus=num_gpus,)
    resources_per_trial = adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_trials,
        model_estimate_memory_usage=model_estimate_memory_usage,
    )

    expected_num_parallel_jobs = 2  # even user wants to run 1000 jobs in prallel, cpu can run 4 jobs in parallel, memory only allows for 2 jobs
    expected_resources_per_trial = dict(
        cpu = num_cpus / expected_num_parallel_jobs,
        gpu = num_gpus / expected_num_parallel_jobs,
    )

    assert expected_resources_per_trial == resources_per_trial
    assert expected_num_parallel_jobs ==  adapter.num_parallel_jobs


def test_automm_resource_allocation():
    num_cpus = get_cpu_count(logical=False)
    num_gpus = get_gpu_count_all()
    num_trials = 1  # TODO: update to more trials when CI supports multiple GPUs
    
    adapter = AutommRayTuneAdapter()
    total_resources = dict(num_cpus=num_cpus, num_gpus=num_gpus,)
    resources_per_trial = adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_trials,
    )
    
    expected_num_parallel_jobs = 1
    expected_resources_per_trial = dict(
        cpu = num_cpus / expected_num_parallel_jobs,
        gpu = num_gpus / expected_num_parallel_jobs,
    )
    
    assert expected_resources_per_trial == resources_per_trial.required_resources
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
        

@pytest.mark.parametrize(
    "searcher, scheduler",
    list(zip(searcher_presets.keys(), scheduler_presets.keys()))
)     
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

import math
import psutil
import time

import autogluon.core as ag
from autogluon.core.scheduler.local_scheduler import LocalParallelScheduler
from autogluon.core.utils import get_cpu_count, get_gpu_count_all
from autogluon.core.space import Real


def _get_dummy_inputs():
    search_space = dict(
        lr=Real(1E-5, 1E-3),
    )

    def train_fn(args, reporter):
        pass
    
    return search_space, train_fn


def test_empty_resource_handling():
    search_space, train_fn = _get_dummy_inputs()

    scheduler = LocalParallelScheduler(
        train_fn,
        search_space=search_space,
        resource=None,
        num_trials=10,
    )

    expected_resource = dict(
        num_cpus = get_cpu_count(),
        num_gpus = 0,
    )

    assert scheduler.total_resource == expected_resource


def test_all_resource_handling():
    search_space, train_fn = _get_dummy_inputs()

    scheduler = LocalParallelScheduler(
        train_fn,
        search_space=search_space,
        resource=dict(num_cpus='all', num_gpus='all'),
        num_trials=10,
    )

    expected_resource = dict(
        num_cpus = get_cpu_count(),
        num_gpus = get_gpu_count_all(),
    )

    assert scheduler.total_resource == expected_resource


def test_resource_allocation():
    num_cpus = get_cpu_count()
    num_gpus = get_gpu_count_all()
    mem_available = psutil.virtual_memory().available
    search_space, train_fn = _get_dummy_inputs()
    num_trials = num_cpus // 4  # can run 4 jobs in parallel
    model_estimate_memory_usage = mem_available // 2.5  # can run 2 jobs in parallel

    scheduler = LocalParallelScheduler(
        train_fn,
        search_space=search_space,
        resource=dict(num_cpus=num_cpus, num_gpus=num_gpus),
        num_trials=num_trials,
        model_estimate_memory_usage=model_estimate_memory_usage,
    )

    expected_num_parallel_jobs = 2  # even cpu can run 4 jobs in parallel, memory only allows for 2 jobs
    expected_trial_resource = dict(
        num_cpus = num_cpus / expected_num_parallel_jobs,
        num_gpus = num_gpus / expected_num_parallel_jobs,
    )
    expected_batches = math.ceil(num_trials / expected_num_parallel_jobs)
    
    assert expected_trial_resource == scheduler.trial_resource
    assert expected_batches == scheduler.batches
    assert expected_num_parallel_jobs ==  scheduler.num_parallel_jobs


def test_parallel_scheduler_can_handle_failing_jobs():
    expected_outcome = [False, False, False, False, False, False, False, True, True, True]
    best_result = [0.7]

    search_space = dict(a=ag.space.Categorical(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))

    def train_fn(args, reporter):
        test_should_fail = args['a'] > 0.7
        if test_should_fail:
            raise Exception('Failed Trial')
        elif args['a'] > best_result[0]:
            best_result[0] = args['a']
        reporter(epoch=1, accuracy=args['a'], test_should_fail=test_should_fail)

    scheduler = LocalParallelScheduler(
        train_fn,
        searcher='local_grid',
        search_space=search_space,
        resource={'num_cpus': 'all', 'num_gpus': 0},
        num_trials=10,
        reward_attr='accuracy',
        time_attr='epoch',
        checkpoint=None,
    )

    scheduler.run()
    scheduler.join_jobs()

    actual_runs = []
    for trial in scheduler.training_history.values():
        is_failed = False
        for i in trial:
            if 'traceback' in i:
                is_failed = True
                break
        actual_runs.append(is_failed)
    assert expected_outcome == actual_runs
    assert (scheduler.get_best_reward() == best_result[0])
    assert (scheduler.get_best_config() == {'a': best_result[0]})


def test_timeout_parallel_scheduler():
    # There are overhead for spinning up ray workers.
    # This test is only able to do 1 parallel run(2 jobs), even mathmatically it should be able to do 2 parallel runs.
    search_space = dict(
        lr=Real(1E-5, 1E-3),
    )

    def train_fn(args, reporter):
        start_tick = time.time()
        time.sleep(0.1)
        reporter(reward=time.time() - start_tick, time_attr=0)

    scheduler = LocalParallelScheduler(train_fn,
                                         search_space=search_space,
                                         resource=dict(num_cpus=2),
                                         num_trials=7,
                                         time_attr='time_attr',
                                         time_out=0.25)
    scheduler.run()
    scheduler.join_jobs()
    assert len(scheduler.config_history) <= 2

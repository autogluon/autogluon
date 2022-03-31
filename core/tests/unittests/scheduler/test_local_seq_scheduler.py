import time

import autogluon.core as ag
from autogluon.core import Real
from autogluon.core.scheduler.local_scheduler import LocalSequentialScheduler


def test_timeout_sequential_scheduler():
    search_space = dict(
        lr=Real(1E-5, 1E-3),
    )

    def train_fn(args, reporter):
        start_tick = time.time()
        time.sleep(0.01)
        reporter(reward=time.time() - start_tick, time_attr=0)

    scheduler = LocalSequentialScheduler(train_fn,
                                         search_space=search_space,
                                         num_trials=7,
                                         time_attr='time_attr',
                                         time_out=0.025)
    scheduler.run()
    scheduler.join_jobs()
    assert len(scheduler.config_history) <= 2


def test_sequential_scheduler_can_handle_failing_jobs():
    trails_outcomes = []
    best_result = [-1]

    search_space = dict(a=ag.space.Real(0, 1))

    def train_fn(args, reporter):
        test_should_fail = args['a'] > 0.7
        trails_outcomes.append(test_should_fail)
        if test_should_fail:
            raise Exception('Failed Trial')
        elif args['a'] > best_result[0]:
            best_result[0] = args['a']
        reporter(epoch=1, accuracy=args['a'])

    scheduler = LocalSequentialScheduler(
        train_fn,
        search_space=search_space,
        resource={'num_cpus': 'all', 'num_gpus': 0},
        num_trials=10,
        reward_attr='accuracy',
        time_attr='epoch',
        checkpoint=None,
    )

    scheduler.run()

    actual_runs = []
    for trial in scheduler.training_history.values():
        is_failed = False
        for i in trial:
            if 'traceback' in i:
                is_failed = True
                break
        actual_runs.append(is_failed)

    assert trails_outcomes == actual_runs
    assert (scheduler.get_best_reward() == best_result[0])
    assert (scheduler.get_best_config() == {'a': best_result[0]})

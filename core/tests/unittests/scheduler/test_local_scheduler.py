import numpy as np
import pickle
import pytest

import autogluon.core as ag

from autogluon.core.space import Real
from autogluon.core.scheduler import LocalScheduler, LocalSequentialScheduler, LocalParallelScheduler


cls = LocalScheduler
scheduler_clses = [LocalSequentialScheduler, LocalParallelScheduler]


def test_get_average_trial_time_():
    running_time = cls.get_average_trial_time_(0, avg_trial_run_time=None, trial_start_time=100, time_end=102)
    assert running_time == 2
    running_time = cls.get_average_trial_time_(1, avg_trial_run_time=running_time, trial_start_time=110, time_end=114)
    assert running_time == 3.0
    running_time = cls.get_average_trial_time_(2, avg_trial_run_time=running_time, trial_start_time=120, time_end=126)
    assert running_time == 4.0


def test_has_enough_time_for_trial__enough_time__no_avg_time():
    # Enough time - no average time
    assert cls.has_enough_time_for_trial_(time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=None)


def test_has_enough_time_for_trial__enough_time__avg_time_allows_trials():
    # Enough time - average time allows more trial
    assert cls.has_enough_time_for_trial_(time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=1)


def test_has_enough_time_for_trial__enough_time__avg_time_not_allows_trials():
    # Enough time - average time does not allow more trial
    assert not cls.has_enough_time_for_trial_(time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=5)


def test_has_enough_time_for_trial__time_exceeded_no_avg_time():
    # Time exceeded - no average time
    assert not cls.has_enough_time_for_trial_(time_out=10, time_start=100, trial_start_time=105, trial_end_time=116, avg_trial_run_time=None)


def test_has_enough_time_for_trial__avg_time():
    # Time exceeded - no average time
    assert not cls.has_enough_time_for_trial_(time_out=10, time_start=100, trial_start_time=105, trial_end_time=116, avg_trial_run_time=0)


def test_has_enough_time_for_trial__enough_time__avg_time_not_allows_trials_by_fill_factor():
    # Enough time - average time does not allow more trial
    assert not cls.has_enough_time_for_trial_(time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=1, fill_factor=5)


def test_search_space():
    search_space = dict(
        a=ag.space.Real(1e-3, 1e-2, log=True),
        b=ag.space.Real(1e-3, 1e-2),
        c=ag.space.Int(1, 10),
        d=ag.space.Categorical('a', 'b', 'c', 'd'),
        e=ag.space.Bool(),
    )

    def train_fn(args, reporter):
        a, b, c, d, e = args['a'], args['b'], args['c'], args['d'], args['e']

        assert a <= 1e-2 and a >= 1e-3
        assert b <= 1e-2 and b >= 1e-3
        assert c <= 10 and c >= 1
        assert d in ['a', 'b', 'c', 'd']
        assert e in [True, False]
        reporter(epoch=1, accuracy=0)

    for scheduler_cls in scheduler_clses:
        scheduler = scheduler_cls(
            train_fn,
            search_space=search_space,
            resource={'num_cpus': 'all', 'num_gpus': 0},
            num_trials=10,
            reward_attr='accuracy',
            time_attr='epoch',
            checkpoint=None
        )

        scheduler.run()


def test_local_scheduler():
    search_space = dict(
        lr=Real(1e-3, 1e-2, log=True),
        wd=Real(1e-3, 1e-2),
        epochs=10,
    )

    def train_fn(args, reporter):
        for e in range(args['epochs']):
            dummy_reward = 1 - np.power(1.8, -np.random.uniform(e, 2 * e))
            reporter(epoch=e + 1, reward=dummy_reward, lr=args.get('lr'), wd=args.get('wd'))

    for scheduler_cls in scheduler_clses:
        scheduler = scheduler_cls(train_fn,
                                    search_space=search_space,
                                    num_trials=10)
        scheduler.run()
        scheduler.join_jobs()
        best_config = scheduler.get_best_config()
        best_task_id = scheduler.get_best_task_id()
        assert pickle.dumps(scheduler.config_history[best_task_id]) == pickle.dumps(best_config)

def test_LocalSequentialScheduler_no_criteria():
    search_space = {'lr': Real(1e-2, 1e-1, log=True)}

    def _train_fn_():
        pass

    with pytest.raises(AssertionError, match="Need stopping criterion: Either num_trials or time_out"):
        for scheduler_cls in scheduler_clses:
            scheduler_cls(train_fn=_train_fn_, search_space=search_space, reward_attr='reward_attr', resource={})

from time import sleep

import pytest

import autogluon.core as ag
from autogluon.core import args, Real
from autogluon.core.scheduler.seq_scheduler import LocalSequentialScheduler

cls = LocalSequentialScheduler


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


def test_LocalSequentialScheduler_no_criteria():
    @args(lr=Real(1e-2, 1e-1, log=True))
    def _train_fn_():
        pass

    with pytest.raises(AssertionError, match="Need stopping criterion: Either num_trials or time_out"):
        LocalSequentialScheduler(train_fn=_train_fn_, reward_attr='reward_attr', resource={})


def test_search_space():
    @ag.args(
        a=ag.space.Real(1e-3, 1e-2, log=True),
        b=ag.space.Real(1e-3, 1e-2),
        c=ag.space.Int(1, 10),
        d=ag.space.Categorical('a', 'b', 'c', 'd'),
        e=ag.space.Bool(),
        f=ag.space.List(
            ag.space.Int(1, 2),
            ag.space.Categorical(4, 5),
        ),
        g=ag.space.Dict(
            a=ag.Real(0, 10),
            obj=ag.space.Categorical('auto', 'gluon'),
        ),
        h=ag.space.Categorical('test', ag.space.Categorical('auto', 'gluon')),
        i=ag.space.Categorical('mxnet', 'pytorch'),
    )
    def train_fn(args, reporter):
        a, b, c, d, e, f, g, h, i = args.a, args.b, args.c, args.d, args.e, args.f, args.g, args.h, args.i

        class MyObj:
            def __init__(self, name):
                self.name = name

        def myfunc(framework):
            return framework

        assert a <= 1e-2 and a >= 1e-3
        assert b <= 1e-2 and b >= 1e-3
        assert c <= 10 and c >= 1
        assert d in ['a', 'b', 'c', 'd']
        assert e in [True, False]
        assert f[0] in [1, 2]
        assert f[1] in [4, 5]
        assert g['a'] <= 10 and g['a'] >= 0
        assert MyObj(g.obj).name in ['auto', 'gluon']
        assert e in [True, False]
        assert h in ['test', 'auto', 'gluon']
        assert myfunc(i) in ['mxnet', 'pytorch']
        reporter(epoch=1, accuracy=0)

    scheduler = LocalSequentialScheduler(
        train_fn,
        resource={'num_cpus': 'all', 'num_gpus': 0},
        num_trials=10,
        reward_attr='accuracy',
        time_attr='epoch',
        checkpoint=None
    )

    scheduler.run()


def test_scheduler_can_handle_failing_jobs():
    trails_outcomes = []
    best_result = [-1]

    @ag.args(a=ag.space.Real(0, 1))
    def train_fn(args, reporter):
        test_should_fail = args.a > 0.7
        trails_outcomes.append(test_should_fail)
        if test_should_fail:
            raise Exception('Failed Trial')
        elif args.a > best_result[0]:
            best_result[0] = args.a

        sleep(0.2)
        reporter(epoch=1, accuracy=args.a)

    scheduler = LocalSequentialScheduler(
        train_fn,
        resource={'num_cpus': 'all', 'num_gpus': 0},
        time_out=3,
        reward_attr='accuracy',
        time_attr='epoch',
        checkpoint=None
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

import pytest

from autogluon.common import space
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
    assert cls.has_enough_time_for_trial_(
        time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=None
    )


def test_has_enough_time_for_trial__enough_time__avg_time_allows_trials():
    # Enough time - average time allows more trial
    assert cls.has_enough_time_for_trial_(
        time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=1
    )


def test_has_enough_time_for_trial__enough_time__avg_time_not_allows_trials():
    # Enough time - average time does not allow more trial
    assert not cls.has_enough_time_for_trial_(
        time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=5
    )


def test_has_enough_time_for_trial__time_exceeded_no_avg_time():
    # Time exceeded - no average time
    assert not cls.has_enough_time_for_trial_(
        time_out=10, time_start=100, trial_start_time=105, trial_end_time=116, avg_trial_run_time=None
    )


def test_has_enough_time_for_trial__avg_time():
    # Time exceeded - no average time
    assert not cls.has_enough_time_for_trial_(
        time_out=10, time_start=100, trial_start_time=105, trial_end_time=116, avg_trial_run_time=0
    )


def test_has_enough_time_for_trial__enough_time__avg_time_not_allows_trials_by_fill_factor():
    # Enough time - average time does not allow more trial
    assert not cls.has_enough_time_for_trial_(
        time_out=10, time_start=100, trial_start_time=105, trial_end_time=106, avg_trial_run_time=1, fill_factor=5
    )


def test_LocalSequentialScheduler_no_criteria():
    search_space = {"lr": space.Real(1e-2, 1e-1, log=True)}

    def _train_fn_():
        pass

    with pytest.raises(AssertionError, match="Need stopping criterion: Either num_trials or time_out"):
        LocalSequentialScheduler(
            train_fn=_train_fn_, search_space=search_space, reward_attr="reward_attr", resource={}
        )


def test_search_space():
    search_space = dict(
        a=space.Real(1e-3, 1e-2, log=True),
        b=space.Real(1e-3, 1e-2),
        c=space.Int(1, 10),
        d=space.Categorical("a", "b", "c", "d"),
        e=space.Bool(),
    )

    def train_fn(args, reporter):
        a, b, c, d, e = args["a"], args["b"], args["c"], args["d"], args["e"]

        assert a <= 1e-2 and a >= 1e-3
        assert b <= 1e-2 and b >= 1e-3
        assert c <= 10 and c >= 1
        assert d in ["a", "b", "c", "d"]
        assert e in [True, False]
        reporter(epoch=1, accuracy=0)

    scheduler = LocalSequentialScheduler(
        train_fn,
        search_space=search_space,
        resource={"num_cpus": "all", "num_gpus": 0},
        num_trials=10,
        reward_attr="accuracy",
        time_attr="epoch",
        checkpoint=None,
    )

    scheduler.run()


def test_scheduler_can_handle_failing_jobs():
    trails_outcomes = []
    best_result = [-1]

    search_space = dict(a=space.Real(0, 1))

    def train_fn(args, reporter):
        test_should_fail = args["a"] > 0.7
        trails_outcomes.append(test_should_fail)
        if test_should_fail:
            raise Exception("Failed Trial")
        elif args["a"] > best_result[0]:
            best_result[0] = args["a"]
        reporter(epoch=1, accuracy=args["a"])

    scheduler = LocalSequentialScheduler(
        train_fn,
        search_space=search_space,
        resource={"num_cpus": "all", "num_gpus": 0},
        num_trials=10,
        reward_attr="accuracy",
        time_attr="epoch",
        checkpoint=None,
    )

    scheduler.run()

    actual_runs = []
    for trial in scheduler.training_history.values():
        is_failed = False
        for i in trial:
            if "traceback" in i:
                is_failed = True
                break
        actual_runs.append(is_failed)

    assert trails_outcomes == actual_runs
    assert scheduler.get_best_reward() == best_result[0]
    assert scheduler.get_best_config() == {"a": best_result[0]}

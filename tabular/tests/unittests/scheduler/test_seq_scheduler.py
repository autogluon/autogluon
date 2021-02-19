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

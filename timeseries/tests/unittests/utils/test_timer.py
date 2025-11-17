import time

import pytest

from autogluon.timeseries.utils.timer import SplitTimer


@pytest.mark.parametrize("strict_stop", [True, False])
def test_when_timer_not_started_then_get_raises_error(strict_stop):
    timer = SplitTimer(10.0, rounds=2, strict_stop=strict_stop)
    with pytest.raises(RuntimeError, match="Timer has not been started"):
        timer.get()


@pytest.mark.parametrize("strict_stop", [True, False])
def test_when_timer_not_started_then_time_elapsed_raises_error(strict_stop):
    timer = SplitTimer(10.0, rounds=2, strict_stop=strict_stop)
    with pytest.raises(RuntimeError, match="Timer has not been started"):
        timer.time_elapsed()


@pytest.mark.parametrize("strict_stop", [True, False])
def test_when_time_limit_is_none_then_get_returns_none(strict_stop):
    timer = SplitTimer(None, rounds=2, strict_stop=strict_stop).start()
    assert timer.get() is None


@pytest.mark.parametrize("strict_stop", [True, False])
def test_when_timer_started_then_get_returns_split_time(strict_stop):
    timer = SplitTimer(10.0, rounds=2, strict_stop=strict_stop).start()
    first_split = timer.get()

    assert first_split is not None
    assert abs(first_split - 5.0) < 0.01


@pytest.mark.parametrize("strict_stop", [True, False])
def test_when_timer_split_then_remaining_time_adjusts(strict_stop):
    timer = SplitTimer(10.0, rounds=2, strict_stop=strict_stop).start()
    timer.split()
    second_split = timer.get()
    assert second_split is not None
    assert abs(second_split - 10.0) < 0.01  # All remaining time for last round


def test_when_all_rounds_used_then_get_returns_zero():
    timer = SplitTimer(10.0, rounds=2).start()
    timer.split()
    timer.split()
    assert timer.get() == 0.0


@pytest.mark.parametrize("strict_stop", [True, False])
def test_when_time_elapsed_then_returns_correct_duration(strict_stop):
    timer = SplitTimer(10.0, rounds=2, strict_stop=strict_stop).start()
    time.sleep(0.01)
    elapsed = timer.time_elapsed()
    assert elapsed >= 0.01


@pytest.mark.parametrize("rounds", [1, 3, 5])
def test_when_iterator_used_then_yields_correct_number_of_splits(rounds):
    timer = SplitTimer(10.0, rounds=rounds)
    splits = list(timer)
    assert len(splits) == rounds


def test_when_iterator_used_then_splits_sum_to_total_time():
    timer = SplitTimer(10.0, rounds=4)
    splits = list(timer)
    expected_splits = [2.5, 3.33333333, 5.0, 10.0]
    for actual, expected in zip(splits, expected_splits):
        assert abs(actual - expected) < 0.01  # type: ignore


def test_when_start_called_then_timer_resets():
    timer = SplitTimer(10.0, rounds=2).start()
    timer.split()
    assert timer.round_index == 1

    timer.start()
    assert timer.round_index == 0
    assert abs(timer.get() - 5.0) < 0.01  # type: ignore


def test_when_strict_stop_enabled_and_rounds_exceeded_then_raises_error():
    timer = SplitTimer(0.5, rounds=2, strict_stop=True).start()
    timer.split()
    timer.split()
    with pytest.raises(RuntimeError, match="Timer has run out of rounds"):
        timer.get()


def test_when_strict_stop_enabled_and_time_exceeded_then_raises_error():
    timer = SplitTimer(0.1, rounds=5, strict_stop=True).start()
    time.sleep(0.15)
    with pytest.raises(RuntimeError, match="Time limit exceeded"):
        timer.get()


def test_when_strict_stop_disabled_and_time_exceeded_then_returns_negative():
    timer = SplitTimer(0.1, rounds=2, strict_stop=False).start()
    time.sleep(0.15)
    result = timer.get()
    assert result < 0  # type: ignore


def test_when_strict_stop_enabled_in_iterator_and_time_exceeded_then_stops_gracefully():
    timer = SplitTimer(0.2, rounds=10, strict_stop=True)
    splits = []
    for split_time in timer:
        splits.append(split_time)
        time.sleep(0.05)

    assert len(splits) < 10  # Should stop before all rounds


def test_when_strict_stop_disabled_in_iterator_then_completes_all_rounds():
    timer = SplitTimer(0.1, rounds=5, strict_stop=False)
    splits = []
    for split_time in timer:
        splits.append(split_time)
        time.sleep(0.05)

    assert len(splits) == 5  # all rounds have run
    assert splits[-1] < 0

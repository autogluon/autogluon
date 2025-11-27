import time

import pytest

from autogluon.timeseries.utils.timer import SplitTimer, Timer


class TestTimer:
    def test_when_timer_not_started_then_time_elapsed_raises_error(self):
        timer = Timer(time_limit=10.0)
        with pytest.raises(RuntimeError, match="Timer has not been started"):
            timer.time_elapsed()

    def test_when_timer_not_started_then_time_remaining_raises_error(self):
        timer = Timer(time_limit=10.0)
        with pytest.raises(RuntimeError, match="Timer has not been started"):
            timer.time_remaining()

    def test_when_timer_not_started_then_timed_out_raises_error(self):
        timer = Timer(time_limit=10.0)
        with pytest.raises(RuntimeError, match="Timer has not been started"):
            timer.timed_out()

    def test_when_timer_started_then_time_elapsed_returns_positive(self):
        timer = Timer(time_limit=10.0).start()
        time.sleep(0.01)
        assert timer.time_elapsed() >= 0.01

    def test_when_timer_started_then_time_remaining_decreases(self):
        timer = Timer(time_limit=1.0).start()
        initial_remaining = timer.time_remaining()
        time.sleep(0.01)
        later_remaining = timer.time_remaining()
        assert later_remaining < initial_remaining  # type: ignore

    def test_when_time_limit_none_then_time_remaining_returns_none(self):
        timer = Timer(time_limit=None).start()
        assert timer.time_remaining() is None

    def test_when_time_limit_none_then_timed_out_returns_false(self):
        timer = Timer(time_limit=None).start()
        time.sleep(0.01)
        assert timer.timed_out() is False

    def test_when_time_not_exceeded_then_timed_out_returns_false(self):
        timer = Timer(time_limit=10.0).start()
        assert timer.timed_out() is False

    def test_when_time_exceeded_then_timed_out_returns_true(self):
        timer = Timer(time_limit=0.01).start()
        time.sleep(0.02)
        assert timer.timed_out() is True

    def test_when_start_called_twice_then_timer_resets(self):
        timer = Timer(time_limit=10.0).start()
        time.sleep(0.01)
        first_elapsed = timer.time_elapsed()

        timer.start()
        second_elapsed = timer.time_elapsed()
        assert second_elapsed < first_elapsed


class TestSplitTimer:
    def test_when_timer_not_started_then_round_time_remaining_raises_error(self):
        timer = SplitTimer(10.0, rounds=2)
        with pytest.raises(RuntimeError, match="Timer has not been started"):
            timer.round_time_remaining()

    def test_when_timer_not_started_then_time_elapsed_raises_error(self):
        timer = SplitTimer(10.0, rounds=2)
        with pytest.raises(RuntimeError, match="Timer has not been started"):
            timer.time_elapsed()

    def test_when_timer_not_started_then_round_time_elapsed_raises_error(self):
        timer = SplitTimer(10.0, rounds=2)
        with pytest.raises(RuntimeError, match="Timer has not been started"):
            timer.round_time_elapsed()

    def test_when_timer_not_started_then_next_round_raises_error(self):
        timer = SplitTimer(10.0, rounds=2)
        with pytest.raises(RuntimeError, match="Timer has not been started"):
            timer.next_round()

    def test_when_time_limit_is_none_then_round_time_remaining_returns_none(self):
        timer = SplitTimer(None, rounds=2).start()
        assert timer.round_time_remaining() is None

    def test_when_timer_started_then_round_time_remaining_returns_split_time(self):
        timer = SplitTimer(10.0, rounds=2).start()
        first_split = timer.round_time_remaining()

        assert first_split is not None
        assert abs(first_split - 5.0) < 0.01

    def test_when_timer_next_round_then_remaining_time_adjusts(self):
        timer = SplitTimer(10.0, rounds=2).start()
        timer.next_round()
        second_split = timer.round_time_remaining()
        assert second_split is not None
        assert abs(second_split - 10.0) < 0.01  # All remaining time for last round

    def test_when_all_rounds_used_then_round_time_remaining_returns_zero(self):
        timer = SplitTimer(10.0, rounds=2).start()
        timer.next_round()
        timer.next_round()
        assert timer.round_time_remaining() == 0.0

    def test_when_time_elapsed_then_returns_correct_duration(self):
        timer = SplitTimer(10.0, rounds=2).start()
        time.sleep(0.01)
        elapsed = timer.time_elapsed()
        assert elapsed >= 0.01

    def test_when_round_time_elapsed_then_returns_correct_duration(self):
        timer = SplitTimer(10.0, rounds=2).start()
        time.sleep(0.01)
        elapsed = timer.round_time_elapsed()
        assert elapsed >= 0.01

    def test_when_next_round_then_round_time_elapsed_resets(self):
        timer = SplitTimer(10.0, rounds=2).start()
        time.sleep(0.02)
        first_round_elapsed = timer.round_time_elapsed()
        assert first_round_elapsed >= 0.02

        timer.next_round()
        second_round_elapsed = timer.round_time_elapsed()
        assert second_round_elapsed < first_round_elapsed

    def test_when_start_called_then_timer_resets(self):
        timer = SplitTimer(10.0, rounds=2).start()
        timer.next_round()
        assert timer.round_index == 1

        timer.start()
        assert timer.round_index == 0
        assert abs(timer.round_time_remaining() - 5.0) < 0.01  # type: ignore

    def test_when_time_exceeded_then_returns_negative(self):
        timer = SplitTimer(0.1, rounds=2).start()
        time.sleep(0.15)
        result = timer.round_time_remaining()
        assert result < 0  # type: ignore

    def test_when_round_uses_less_time_then_next_round_gets_more(self):
        """Test that unused time from one round is redistributed to remaining rounds."""
        timer = SplitTimer(10.0, rounds=3).start()

        # Round 1: should get ~3.33s (10.0 / 3)
        round_1_time = timer.round_time_remaining()
        assert round_1_time is not None
        assert abs(round_1_time - 3.33) < 0.1

        time.sleep(0.01)
        timer.next_round()

        round_2_time = timer.round_time_remaining()
        assert round_2_time is not None
        assert round_2_time > 4.9

        time.sleep(0.02)
        timer.next_round()

        round_3_time = timer.round_time_remaining()
        assert round_3_time is not None
        assert round_3_time > 9.5

    def test_when_single_round_then_gets_all_time(self):
        timer = SplitTimer(10.0, rounds=1).start()
        round_time = timer.round_time_remaining()
        assert round_time is not None
        assert abs(round_time - 10.0) < 0.01

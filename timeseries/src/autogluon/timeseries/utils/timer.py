import time
from typing import Optional

from typing_extensions import Self


class Timer:
    """A timer class that tracks a start time, and computes the time elapsed and
    time remaining, used for handling `time_limit` parameters in AutoGluon.

    Parameters
    ----------
    time_limit
        The time limit to set. If None, then `time_remaining` will return None, and
        `timed_out` will return False.

    Examples
    --------
    Basic usage with time limit:

    >>> timer = Timer(time_limit=10.0).start()
    >>> # Do some work...
    >>> if timer.timed_out():
    ...     print("Time limit exceeded!")
    >>> print(f"Time remaining: {timer.time_remaining():.2f}s")

    Using as a stopwatch (no time limit):

    >>> timer = Timer(time_limit=None).start()
    >>> # Do some work...
    >>> print(f"Elapsed time: {timer.time_elapsed():.2f}s")

    Checking time in a loop:

    >>> timer = Timer(time_limit=5.0).start()
    >>> for i in range(100):
    ...     if timer.timed_out():
    ...         break
    ...     # Do work for iteration i
    """

    def __init__(
        self,
        time_limit: Optional[float],
    ):
        self.time_limit = time_limit

        self.start_time = None

    def start(self) -> Self:
        """Start or reset the timer."""
        self.start_time = time.monotonic()
        return self

    def time_elapsed(self) -> float:
        """Time since the timer was started. This method can also be used when
        `time_limit` is set to None to count time forward (i.e., as opposed to
        a countdown timer which other methods imply)."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")
        return time.monotonic() - self.start_time

    def time_remaining(self) -> Optional[float]:
        """Time remaining on the timer. If `time_limit` is None, this method also
        returns None."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")
        if self.time_limit is None:
            return None
        return self.time_limit - (time.monotonic() - self.start_time)

    def timed_out(self) -> bool:
        """Whether the timer has timed out. If `time_limit` is None, this method
        always returns False."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")
        if self.time_limit is None:
            return False
        return self.time_elapsed() >= self.time_limit


class SplitTimer(Timer):
    """A timer that splits remaining time across multiple rounds.

    Extends Timer to divide the total time limit across a specified number of rounds,
    useful for allocating time budgets to sequential operations. At each call of
    `get`, the timer re-distributes the remaining time evenly among the remaining
    rounds.

    Parameters
    ----------
    time_limit
        Total time limit to split across all rounds. If None, `get` returns None.
    rounds
        Number of rounds to split the time across. Default is 1.

    Examples
    --------
    Split time equally across 3 rounds:

    >>> timer = SplitTimer(time_limit=30.0, rounds=3).start()
    >>> for time_for_round in timer:
    ...     print(f"Time for this round: {time_for_round:.1f}s")
    ...     # Do work with time_for_round budget

    Manual splitting with get() and split():

    >>> timer = SplitTimer(time_limit=10.0, rounds=2).start()
    >>> time_round_1 = timer.get()  # Returns ~5.0
    >>> # Do work for round 1, e.g., for 7 seconds
    >>> timer.split()
    >>> time_round_2 = timer.get()  # Returns remaining time = ~3 seconds
    >>> # Do work for round 2
    """

    def __init__(
        self,
        time_limit: Optional[float],
        rounds: int = 1,
    ):
        super().__init__(time_limit)
        self.rounds = rounds

        self.round_index = 0

    def start(self) -> Self:
        """Reset and start the timer."""
        super().start()
        self.round_index = 0
        return self

    def get(self) -> Optional[float]:
        """Get the time budget for the current round.

        Calculates the time allocation by dividing the remaining time equally among
        the remaining rounds. This means if a previous round used less time than
        allocated, subsequent rounds get more time, and vice versa.

        Returns time budget for the current round in seconds. Returns None if `time_limit`
        is None. Returns 0.0 if all rounds have been exhausted.
        """
        if self.time_limit is None:
            return None
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")

        remaining_rounds = self.rounds - self.round_index

        if remaining_rounds <= 0:
            return 0.0
        return self.time_remaining() / remaining_rounds  # type: ignore

    def split(self) -> Self:
        """Advance to the next round.

        Increments the round counter, which affects the time allocation returned
        by subsequent `get` calls.
        """
        self.round_index += 1
        return self

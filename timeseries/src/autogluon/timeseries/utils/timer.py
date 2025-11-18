import time
from typing import Optional

from typing_extensions import Self


class SplitTimer:
    def __init__(
        self,
        time_limit: Optional[float],
        rounds: int = 1,
        strict_stop: bool = False,
    ):
        self.time_limit = time_limit
        self.rounds = rounds
        self.strict_stop = strict_stop

        self.round_index = 0
        self.start_time = None

    def start(self) -> Self:
        """Reset and start the timer"""
        self.start_time = time.monotonic()
        self.round_index = 0
        return self

    def get(self) -> Optional[float]:
        """Get the next time limit"""
        if self.time_limit is None:
            return None
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")
        if self.strict_stop:
            if self.round_index >= self.rounds:
                raise RuntimeError("Timer has run out of rounds")
            if self.time_elapsed() >= self.time_limit:
                raise RuntimeError("Time limit exceeded")

        remaining_rounds = self.rounds - self.round_index

        if remaining_rounds <= 0:
            return 0.0
        return self.time_remaining() / remaining_rounds  # type: ignore

    def time_elapsed(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")
        return time.monotonic() - self.start_time

    def time_remaining(self) -> Optional[float]:
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")
        if self.time_limit is None:
            return None
        return self.time_limit - (time.monotonic() - self.start_time)

    def split(self) -> Self:
        self.round_index += 1
        return self

    def __iter__(self):
        timer = self.start()
        while timer.round_index < timer.rounds:
            try:
                yield timer.get()
                timer.split()
            except RuntimeError:
                break

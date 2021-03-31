
class AbstractES:
    def update(self, cur_round, is_best=False) -> bool:
        raise NotImplementedError

    def early_stop(self, cur_round, is_best=False) -> bool:
        raise NotImplementedError


class SimpleES(AbstractES):
    def __init__(self, stopping_rounds=10):
        self.stopping_rounds = stopping_rounds
        self.best_round = 0

    def update(self, cur_round, is_best=False):
        if is_best:
            print(f'new best: \t{cur_round}\t{self.best_round}\t{cur_round - self.best_round}\t{(cur_round - self.best_round) / self.stopping_rounds}')
            self.best_round = cur_round
        return self.early_stop(cur_round, is_best=is_best)

    def early_stop(self, cur_round, is_best=False):
        if is_best:
            return False
        return cur_round - self.best_round >= self.stopping_rounds


# TODO: Add time component
#  if given a large amount of time and training is fast, should check more rounds before early stopping
# TODO: Incorporate score, rolling window
class AdaptiveES(AbstractES):
    """
    Implements Adaptive Early Stopping

    stopping_rounds is adaptively adjusted across training instead of being a fixed value.

    Parameters
    ----------
    adaptive_rate : float, default 0.2
        The rate of increase in stopping_rounds.
        Set to 0 to disable, or negative to shrink the early stopping rounds during training.
    adaptive_offset : int, default 10
        The initial stopping_rounds when cur_round is 0.
    min_rounds : int, default 10
        The minimum value of stopping_rounds.
    max_rounds : int, default 10000
        The maximum value of stopping_rounds.

    Attributes
    ----------
    best_round : int
        The most recent round passed to self.update with `is_best=True`.
        Dictates stopping_rounds and is used to determine if self.early_stop() returns True.
    stopping_rounds : int
        If no improvement occurs in `stopping_rounds` rounds or greater, self.early_stop will return True.
        stopping_rounds is dictated by the following formula:
        stopping_rounds = min(self.max_rounds, (max(self.min_rounds, round(self.best_round * self.adaptive_rate + self.adaptive_offset))))
        Effectively, stopping_rounds = self.best_round * self.adaptive_rate + self.adaptive_offset, bound by min_rounds and max_rounds
    """
    def __init__(self, adaptive_rate=0.2, adaptive_offset=10, min_rounds=10, max_rounds=10000):
        self.adaptive_rate = adaptive_rate
        self.adaptive_offset = adaptive_offset
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.best_round = 0
        self.stopping_rounds = self._update_stopping_rounds(self.best_round)

    def update(self, cur_round, is_best=False):
        """
        Updates the state of the object. Identical to calling self.early_stop, but if `is_best=True`, it will set `self.best_round=cur_round`.
        If cur_round achieved a new best score, set `is_best=True`.
        Ideally, this should be called every round during training, with the output used to determine if the model should stop training.
        """
        if is_best:
            print(f'new best: \t{cur_round}\t{self.best_round}\t{cur_round-self.best_round}\t{(cur_round-self.best_round)/self.stopping_rounds}')
            self.best_round = cur_round
            self.stopping_rounds = self._update_stopping_rounds(self.best_round)
        return self.early_stop(cur_round, is_best=is_best)

    def early_stop(self, cur_round, is_best=False):
        """
        Returns True if (cur_round - self.best_round) equals or exceeds self.stopping_rounds, otherwise returns False.
        This can be used to indicate if training should stop.
        """
        if is_best:
            return False
        return cur_round - self.best_round >= self.stopping_rounds

    def _update_stopping_rounds(self, best_round):
        return min(
            self.max_rounds,
            (
                max(
                    self.min_rounds,
                    round(best_round * self.adaptive_rate + self.adaptive_offset),
                )
            )
        )


ES_CLASS_MAP = {
    'simple': SimpleES,
    'adaptive': AdaptiveES,
}

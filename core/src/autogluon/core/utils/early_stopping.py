
class AbstractES:
    """
    Abstract early stopping class
    """
    def update(self, cur_round, is_best=False) -> bool:
        raise NotImplementedError

    def early_stop(self, cur_round, is_best=False) -> bool:
        raise NotImplementedError


class SimpleES(AbstractES):
    """
    Implements early stopping with fixed patience

    Parameters
    ----------
    patience : int, default 10
        If no improvement occurs in `patience` rounds or greater, self.early_stop will return True.
    """
    def __init__(self, patience=10):
        self.patience = patience
        self.best_round = 0

    def update(self, cur_round, is_best=False):
        if is_best:
            self.best_round = cur_round
        return self.early_stop(cur_round, is_best=is_best)

    def early_stop(self, cur_round, is_best=False):
        if is_best:
            return False
        return cur_round - self.best_round >= self.patience


# TODO: Add time component
#  if given a large amount of time and training is fast, should check more rounds before early stopping
# TODO: Incorporate score, rolling window
class AdaptiveES(AbstractES):
    """
    Implements early stopping with adaptive patience

    Patience is adaptively adjusted across training instead of being a fixed value.
    This generally outperforms fixed patience strategies. Examples below:
    1. If the current best_round is 10000, it is reasonable to assume that it could take more than 100 rounds before finding a new best.
    2. If the current best_round is 3, it is unlikely that there will be 100 rounds before finding a new best at round 103.
    In the above examples, a fixed patience of 100 would be too little for round 10000, but too large for round 3.
    However, with `adaptive_rate=0.2`, `adaptive_offset=10`, round 3 would have a patience of ~10, while round 10000 would have a patience of ~2000.

    Parameters
    ----------
    adaptive_rate : float, default 0.2
        The rate of increase in patience.
        Set to 0 to disable, or negative to shrink patience during training.
    adaptive_offset : int, default 10
        The initial patience when cur_round is 0.
    min_patience : int, default 10
        The minimum value of patience.
    max_patience : int, default 10000
        The maximum value of patience.

    Attributes
    ----------
    best_round : int
        The most recent round passed to self.update with `is_best=True`.
        Dictates patience and is used to determine if self.early_stop() returns True.
    patience : int
        If no improvement occurs in `patience` rounds or greater, self.early_stop will return True.
        patience is dictated by the following formula:
        patience = min(self.max_patience, (max(self.min_patience, round(self.best_round * self.adaptive_rate + self.adaptive_offset))))
        Effectively, patience = self.best_round * self.adaptive_rate + self.adaptive_offset, bound by min_patience and max_patience
    """
    def __init__(self, adaptive_rate=0.2, adaptive_offset=10, min_patience=10, max_patience=10000):
        self.adaptive_rate = adaptive_rate
        self.adaptive_offset = adaptive_offset
        self.min_patience = min_patience
        self.max_patience = max_patience
        self.best_round = 0
        self.patience = self._update_patience(self.best_round)

    def update(self, cur_round, is_best=False):
        """
        Updates the state of the object. Identical to calling self.early_stop, but if `is_best=True`, it will set `self.best_round=cur_round`.
        If cur_round achieved a new best score, set `is_best=True`.
        Ideally, this should be called every round during training, with the output used to determine if the model should stop training.
        """
        if is_best:
            self.best_round = cur_round
            self.patience = self._update_patience(self.best_round)
        return self.early_stop(cur_round, is_best=is_best)

    def early_stop(self, cur_round, is_best=False):
        """
        Returns True if (cur_round - self.best_round) equals or exceeds self.patience, otherwise returns False.
        This can be used to indicate if training should stop.
        """
        if is_best:
            return False
        return cur_round - self.best_round >= self.patience

    def _update_patience(self, best_round):
        return min(
            self.max_patience,
            (
                max(
                    self.min_patience,
                    round(best_round * self.adaptive_rate + self.adaptive_offset),
                )
            )
        )


ES_CLASS_MAP = {
    'simple': SimpleES,
    'adaptive': AdaptiveES,
}

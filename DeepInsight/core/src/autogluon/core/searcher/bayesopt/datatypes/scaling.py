from abc import ABC, abstractmethod
import numpy as np


class Scaling(ABC):
    @abstractmethod
    def to_internal(self, value: float) -> float:
        pass

    @abstractmethod
    def from_internal(self, value: float) -> float:
        pass

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

    def __eq__(self, other):
        # For usage in tests. Make sure to edit if parameters are added.
        return self.__class__ == other.__class__


class LinearScaling(Scaling):
    def to_internal(self, value: float) -> float:
        return value

    def from_internal(self, value: float) -> float:
        return value


class LogScaling(Scaling):
    def to_internal(self, value: float) -> float:
        assert value > 0, "Value must be strictly positive to be log-scaled."
        return np.log(value)

    def from_internal(self, value: float) -> float:
        return np.exp(value)


class ReverseLogScaling(Scaling):
    def to_internal(self, value: float) -> float:
        assert 0 <= value < 1, \
            "Value must be between 0 (inclusive) and 1 (exclusive) to be reverse-log-scaled."
        return -np.log(1.0 - value)

    def from_internal(self, value: float) -> float:
        return 1.0 - np.exp(-value)

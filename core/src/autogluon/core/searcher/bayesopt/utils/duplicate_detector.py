from abc import ABC, abstractmethod
from typing import Set

from ..datatypes.common import Candidate
from ..datatypes.hp_ranges import HyperparameterRanges


class DuplicateDetector(ABC):
    @abstractmethod
    def contains(self, existing_candidates: Set[Candidate], new_candidate: Candidate) -> bool:
        pass


class DuplicateDetectorNoDetection(DuplicateDetector):
    def contains(self, existing_candidates: Set[Candidate], new_candidate: Candidate) -> bool:
        return False  # no duplicate detection at all


class DuplicateDetectorIdentical(DuplicateDetector):
    def contains(self, existing_candidates: Set[Candidate], new_candidate: Candidate) -> bool:
        return new_candidate in existing_candidates


DUPLICATE_DETECTION_EPSILON = 1e-8


class DuplicateDetectorEpsilon(DuplicateDetector):
    def __init__(self, hp_ranges: HyperparameterRanges):
        self.hp_ranges = hp_ranges

    def contains(self, existing_candidates: Set[Candidate], new_candidate: Candidate) -> bool:
        return any(self._almost_equal(c, new_candidate) for c in existing_candidates)

    def _almost_equal(self, candidate1, candidate2):
        assert len(candidate1) == len(candidate2), (candidate1, candidate2)
        np_cand1 = self.hp_ranges.to_ndarray(candidate1)
        np_cand2 = self.hp_ranges.to_ndarray(candidate2)
        assert np_cand1.shape == np_cand2.shape, (np_cand1, np_cand2)
        return all(abs(hp1 - hp2) < DUPLICATE_DETECTION_EPSILON for hp1, hp2 in zip(np_cand1, np_cand2))

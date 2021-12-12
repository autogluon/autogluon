from typing import Iterator, Set, List
import numpy as np
import logging

from .base_classes import CandidateGenerator
from ..datatypes.common import Candidate
from ..datatypes.hp_ranges import HyperparameterRanges
from ..datatypes.tuning_job_state import TuningJobState

logger = logging.getLogger(__name__)


class RandomStatefulCandidateGenerator(CandidateGenerator):
    """
    As opposed to RandomCandidateGenerator, this generator maintains a
    random state, so if generate_candidates is called several times, different
    sequences are returned.

    """
    def __init__(self, hp_ranges: HyperparameterRanges,
                 random_state: np.random.RandomState):
        self.hp_ranges = hp_ranges
        self.random_state = random_state

    def generate_candidates(self) -> Iterator[Candidate]:
        while True:
            yield self.hp_ranges.random_candidate(self.random_state)

    def generate_candidates_en_bulk(self, num_cands: int) -> List[Candidate]:
        return self.hp_ranges.random_candidates(self.random_state, num_cands)


def compute_blacklisted_candidates(state: TuningJobState) -> Set[Candidate]:
    return set([x.candidate for x in state.candidate_evaluations] + \
               state.pending_candidates + state.failed_candidates)


MAX_RETRIES_ON_DUPLICATES = 10000


# ATTENTION: If this is used with Candidate = CS.Configuration, the overhead
# for filtering out duplicates and blacklisted configs becomes large
def generate_unique_candidates(
        candidates_generator: CandidateGenerator, num_candidates: int,
        blacklisted_candidates: Set[Candidate]) -> List[Candidate]:
    blacklisted = set(blacklisted_candidates)  # copy
    result = []
    num_results = 0
    retries = 0
    for i, cand in enumerate(candidates_generator.generate_candidates()):
        if cand not in blacklisted:
            result.append(cand)
            num_results += 1
            blacklisted.add(cand)
            retries = 0
        else:
            # found a duplicate; retry
            retries += 1

        # End loop if enough candidates where generated, or after too many retries
        # (this latter can happen when most of them are duplicates, and must be done
        # to avoid infinite loops in the purely discrete case)
        if num_results == num_candidates or retries > MAX_RETRIES_ON_DUPLICATES:
            if retries > MAX_RETRIES_ON_DUPLICATES:
                logger.warning(
                    f"Reached limit of {MAX_RETRIES_ON_DUPLICATES} retries with i={i}. "
                    f"Returning {len(result)} candidates instead of the requested {num_candidates}"
                )
            break

    return result

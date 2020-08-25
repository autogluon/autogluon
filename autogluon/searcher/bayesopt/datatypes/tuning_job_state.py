from typing import NamedTuple, List

from .common import Candidate, CandidateEvaluation, PendingEvaluation
from .hp_ranges import HyperparameterRanges


class TuningJobState(NamedTuple):
    """
    Tuning job state (non disjoint: so for a single algorithm when tuning
    multiple algorithms)
    """
    hp_ranges: HyperparameterRanges
    candidate_evaluations: List[CandidateEvaluation]
    failed_candidates: List[Candidate]
    pending_evaluations: List[PendingEvaluation]

    @property
    def pending_candidates(self):
        return [x.candidate for x in self.pending_evaluations]

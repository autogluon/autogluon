from typing import NamedTuple, List

from autogluon.searcher.bayesopt.datatypes.common import Candidate, \
    CandidateEvaluation, PendingEvaluation
from autogluon.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRanges


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

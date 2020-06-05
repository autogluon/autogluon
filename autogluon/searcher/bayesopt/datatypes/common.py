from typing import Union, Tuple, NamedTuple, Dict
import numpy as np
import ConfigSpace as CS



# Allows underlying BO code to be used with different basic types for a
# candidate
Hyperparameter = Union[str, int, float]

Candidate = Union[Tuple[Hyperparameter, ...], CS.Configuration]


def candidate_for_print(candidate: Candidate):
    if isinstance(candidate, CS.Configuration):
        return candidate.get_dictionary()
    else:
        return candidate


class StateIdAndCandidate(NamedTuple):
    """
    Just used in utils/test_objects.py, could probably be removed
    """
    state_id: str
    candidate: Candidate


class CandidateEvaluation(NamedTuple):
    candidate: Candidate
    metrics: Dict[str, float]


class PendingEvaluation(object):
    """
    Maintains information for pending candidates (i.e. candidates which have
    been queried for labeling, but target feedback has not yet been obtained.

    The minimum information is the candidate which has been queried.
    """
    def __init__(self, candidate: Candidate):
        super(PendingEvaluation, self).__init__()
        self._candidate = candidate

    @property
    def candidate(self):
        return self._candidate


class FantasizedPendingEvaluation(PendingEvaluation):
    """
    Here, latent target values are integrated out by Monte Carlo samples,
    also called "fantasies".

    """
    def __init__(self, candidate: Candidate, fantasies: Dict[str, np.ndarray]):
        super(FantasizedPendingEvaluation, self).__init__(candidate)
        fantasy_sizes = [
            fantasy_values.size for fantasy_values in fantasies.values()]
        assert all(fantasy_size > 0 for fantasy_size in fantasy_sizes), \
            "fantasies must be non-empty"
        assert len(set(fantasy_sizes)) == 1, \
            "fantasies must all have the same length"
        self._fantasies = fantasies.copy()

    @property
    def fantasies(self):
        return self._fantasies

from typing import List, Callable, Optional, NamedTuple
import logging
import copy

from .gp_model import GaussProcSurrogateModel, GPModel
from .gpmodel_skipopt import SkipOptimizationPredicate, NeverSkipPredicate
from ..autogluon.gp_profiling import GPMXNetSimpleProfiler
from ..autogluon.debug_log import DebugLogPrinter
from ..datatypes.tuning_job_state import TuningJobState
from ..datatypes.common import Candidate, PendingEvaluation, CandidateEvaluation
from ..tuning_algorithms.base_classes import PendingCandidateStateTransformer
from ..tuning_algorithms.default_algorithm import DEFAULT_METRIC

logger = logging.getLogger(__name__)


class GPModelArgs(NamedTuple):
    num_fantasy_samples: int
    random_seed: int
    active_metric: str = DEFAULT_METRIC
    normalize_targets: bool = True


class GPModelPendingCandidateStateTransformer(PendingCandidateStateTransformer):
    """
    This class maintains the TuningJobState along an asynchronous GP-based
    HPO experiment, and manages the reaction to changes of this state.
    In particular, it provides a GaussProcSurrogateModel on demand, which
    encapsulates the GP posterior.

    Note: The GaussProcSurrogateModel can be accessed only once the state has
    at least one labeled case, since otherwise no posterior can be computed.

    skip_optimization is a predicate depending on TuningJobState, determining
    what is done at the next recent GaussProcSurrogateModel computation. If
    False, the GP hyperparameters are optimized. Otherwise, the current ones
    are not changed.

    Safeguard against multiple GP hyperparameter optimization while labeled
    data does not change:
    The posterior has to be recomputed every time the state changes, even if
    this only concerns pending evaluations. The expensive part of this is
    refitting the GP hyperparameters, which makes sense only when the labeled
    data in the state changes. We put a safeguard in place to avoid refitting
    when the labeled data is unchanged.

    """
    def __init__(
            self, gpmodel: GPModel, init_state: TuningJobState,
            model_args: GPModelArgs,
            skip_optimization: SkipOptimizationPredicate = None,
            profiler: GPMXNetSimpleProfiler = None,
            debug_log: Optional[DebugLogPrinter] = None):
        self._gpmodel = gpmodel
        self._state = copy.copy(init_state)
        self._model_args = model_args
        if skip_optimization is None:
            self.skip_optimization = NeverSkipPredicate()
        else:
            self.skip_optimization = skip_optimization
        self._profiler = profiler
        self._debug_log = debug_log
        # GPMXNetModel computed on demand
        self._model: GaussProcSurrogateModel = None
        self._candidate_evaluations = None
        # _model_params is returned by get_params. Careful: This is not just
        # self._gpmodel.get_params(), since the current GaussProcSurrogateModel
        # may append additional parameters
        self._model_params = gpmodel.get_params()
        # DEBUG
        self._debug_fantasy_values = None

    @property
    def state(self) -> TuningJobState:
        return self._state

    def model(self, **kwargs) -> GaussProcSurrogateModel:
        """
        If skip_optimization is given, it overrides the self.skip_optimization
        predicate.

        :return: GPMXNetModel for current state

        """
        if self._model is None:
            skip_optimization = kwargs.get('skip_optimization')
            self._compute_model(skip_optimization=skip_optimization)
        return self._model

    def get_params(self):
        return self._model_params

    def set_params(self, param_dict):
        self._gpmodel.set_params(param_dict)
        self._model_params = self._gpmodel.get_params()

    def append_candidate(self, candidate: Candidate):
        """
        Appends new pending candidate to the state.

        :param candidate: New pending candidate

        """
        self._model = None  # Invalidate
        self._state.pending_evaluations.append(PendingEvaluation(candidate))

    @staticmethod
    def _find_candidate(candidate: Candidate, lst: List):
        try:
            pos = next(
                i for i, x in enumerate(lst)
                if x.candidate == candidate)
        except StopIteration:
            pos = -1
        return pos

    def drop_candidate(self, candidate: Candidate):
        """
        Drop candidate (labeled or pending) from state.

        :param candidate: Candidate to be dropped

        """
        # Candidate may be labeled or pending. First, try labeled
        pos = self._find_candidate(
            candidate, self._state.candidate_evaluations)
        if pos != -1:
            self._model = None  # Invalidate
            self._state.candidate_evaluations.pop(pos)
            if self._debug_log is not None:
                deb_msg = "[GPModelPendingCandidateStateTransformer.drop_candidate]\n"
                deb_msg += ("- len(candidate_evaluations) afterwards = {}".format(
                    len(self.state.candidate_evaluations)))
                logger.info(deb_msg)
        else:
            # Try pending
            pos = self._find_candidate(
                candidate, self._state.pending_evaluations)
            assert pos != -1, \
                "Candidate {} not registered (neither labeled, nor pending)".format(
                    candidate)
            self._model = None  # Invalidate
            self._state.pending_evaluations.pop(pos)
            if self._debug_log is not None:
                deb_msg = "[GPModelPendingCandidateStateTransformer.drop_candidate]\n"
                deb_msg += ("- len(pending_evaluations) afterwards = {}\n".format(
                    len(self.state.pending_evaluations)))
                logger.info(deb_msg)

    def label_candidate(self, data: CandidateEvaluation):
        """
        Adds a labeled candidate. If it was pending before, it is removed as
        pending candidate.

        :param data: New labeled candidate

        """
        pos = self._find_candidate(
            data.candidate, self._state.pending_evaluations)
        if pos != -1:
            self._state.pending_evaluations.pop(pos)
        self._state.candidate_evaluations.append(data)
        self._model = None  # Invalidate

    def filter_pending_evaluations(
            self, filter_pred: Callable[[PendingEvaluation], bool]):
        """
        Filters state.pending_evaluations with filter_pred.

        :param filter_pred Filtering predicate

        """
        new_pending_evaluations = list(filter(
            filter_pred, self._state.pending_evaluations))
        if len(new_pending_evaluations) != len(self._state.pending_evaluations):
            if self._debug_log is not None:
                deb_msg = "[GPModelPendingCandidateStateTransformer.filter_pending_evaluations]\n"
                deb_msg += ("- from len {} to {}".format(
                    len(self.state.pending_evaluations), len(new_pending_evaluations)))
                logger.info(deb_msg)
            self._model = None  # Invalidate
            del self._state.pending_evaluations[:]
            self._state.pending_evaluations.extend(new_pending_evaluations)

    def mark_candidate_failed(self, candidate: Candidate):
        self._state.failed_candidates.append(candidate)

    def _compute_model(self, skip_optimization: bool = None):
        args = self._model_args
        if skip_optimization is None:
            skip_optimization = self.skip_optimization(self._state)
        fit_parameters = not skip_optimization
        if fit_parameters and self._candidate_evaluations:
            # Did the labeled data really change since the last recent refit?
            # If not, skip the refitting
            if self._state.candidate_evaluations == self._candidate_evaluations:
                fit_parameters = False
                logger.warning(
                    "Skipping the refitting of GP hyperparameters, since the "
                    "labeled data did not change since the last recent fit")
        self._model = GaussProcSurrogateModel(
            state=self._state,
            active_metric=args.active_metric,
            random_seed=args.random_seed,
            gpmodel=self._gpmodel,
            fit_parameters=fit_parameters,
            num_fantasy_samples=args.num_fantasy_samples,
            normalize_targets=args.normalize_targets,
            profiler=self._profiler,
            debug_log=self._debug_log,
            debug_fantasy_values=self._debug_fantasy_values)
        # DEBUG: Supplied values are used only once
        self._debug_fantasy_values = None
        # Note: This may be different than self._gpmodel.get_params(), since
        # the GaussProcSurrogateModel may append additional info
        self._model_params = self._model.get_params()
        if fit_parameters:
            # Keep copy of labeled data in order to avoid unnecessary
            # refitting
            self._candidate_evaluations = copy.copy(
                self._state.candidate_evaluations)

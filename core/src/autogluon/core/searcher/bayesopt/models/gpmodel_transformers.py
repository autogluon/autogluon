from typing import Dict, List, Callable, Optional
import logging
import copy

from .gp_model import GaussProcSurrogateModel, GaussianProcessRegression, GPRegressionMCMC, \
    GaussProcSurrogateOutputModel, GPModelArgsOutput, GPOutputModel
from .gpmodel_skipopt import SkipOptimizationPredicate, NeverSkipPredicate
from ..datatypes.common import Candidate, PendingEvaluation, CandidateEvaluation
from ..datatypes.tuning_job_state import TuningJobState
from ..tuning_algorithms.base_classes import PendingCandidateStateTransformer, dictionarize_objective, DEFAULT_METRIC
from ..utils.simple_profiler import SimpleProfiler
from ..utils.debug_log import DebugLogPrinter

logger = logging.getLogger(__name__)


def _assert_same_keys(dict1, dict2):
    assert set(dict1.keys()) == set(dict2.keys()), \
        f'{list(dict1.keys())} and {list(dict2.keys())} need to be the same keys. '


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

    Note that gpmodel can also be a dictionary mapping output names to
    GP models. In that case, the state is shared but the models for each
    output metric are updated independently.

    """
    def __init__(
            self, gpmodel: GPOutputModel, init_state: TuningJobState,
            model_args: GPModelArgsOutput,
            skip_optimization: Optional[SkipOptimizationPredicate] = None,
            profiler: Optional[SimpleProfiler] = None,
            debug_log: Optional[DebugLogPrinter] = None):
        self._use_single_model = False
        if isinstance(gpmodel, GaussianProcessRegression) or isinstance(gpmodel, GPRegressionMCMC):
            self._use_single_model = True
        if not self._use_single_model:
            assert isinstance(gpmodel, Dict), f'{gpmodel} is neither an instance of GaussianProcessRegression ' \
                                              f'nor GPRegressionMCMC. It is assumed that we are in the multi-output ' \
                                              f'case and that it must be a Dict. No other types are supported. '
            assert isinstance(model_args, Dict), f'{model_args} must be a Dict, consistently with {gpmodel}.'
            _assert_same_keys(gpmodel, model_args)
        self._gpmodel = gpmodel
        self._model_args = model_args
        self._state = copy.copy(init_state)
        if skip_optimization is None:
            self.skip_optimization = NeverSkipPredicate()
        else:
            self.skip_optimization = skip_optimization
        self._profiler = profiler
        self._debug_log = debug_log
        # GPMXNetModel computed on demand
        self._model: GaussProcSurrogateOutputModel = None
        self._candidate_evaluations = None
        # _model_params is returned by get_params. Careful: This is not just
        # self._gpmodel.get_params(), since the current GaussProcSurrogateModel
        # may append additional parameters
        self._assign_model_params(self._gpmodel)
        # DEBUG
        self._debug_fantasy_values = None

    @property
    def state(self) -> TuningJobState:
        return self._state

    def model(self, **kwargs) -> GaussProcSurrogateOutputModel:
        """
        If skip_optimization is given, it overrides the self.skip_optimization
        predicate.

        :return: GPMXNetModel for current state in the standard single model case;
                 in the multi-model case, it returns a dictionary mapping output names
                 to GPMXNetModel instances for current state (shared across models).
        """
        if self._model is None:
            skip_optimization = kwargs.get('skip_optimization')
            self._compute_model(skip_optimization=skip_optimization)
        return self._model

    def get_params(self):
        return self._model_params

    def set_params(self, param_dict):
        if self._use_single_model:
            self._gpmodel.set_params(param_dict)
        else:
            _assert_same_keys(self._gpmodel, param_dict)
            for key in self._gpmodel:
                self._gpmodel[key].set_params(param_dict[key])
        self._assign_model_params(self._gpmodel)

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

    def _compute_model(self, skip_optimization=None):
        if skip_optimization is None:
            skip_optimization = self.skip_optimization(self._state)
            if skip_optimization and self._debug_log is not None:
                logger.info("Skipping the refitting of GP hyperparameters")
        fit_parameters = not skip_optimization
        if fit_parameters and self._candidate_evaluations:
            # Did the labeled data really change since the last recent refit?
            # If not, skip the refitting
            if self._state.candidate_evaluations == self._candidate_evaluations:
                fit_parameters = False
                logger.info(
                    "Skipping the refitting of GP hyperparameters, since the "
                    "labeled data did not change since the last recent fit")

        if self._use_single_model:
            gpmodels = dictionarize_objective(self._gpmodel)
            model_args = dictionarize_objective(self._model_args)
        else:
            gpmodels = self._gpmodel
            model_args = self._model_args
        _assert_same_keys(gpmodels, model_args)
        output_models = {}
        for metric_name in gpmodels:
            args = model_args[metric_name]
            model = GaussProcSurrogateModel(
                state=self._state,
                active_metric=metric_name,
                random_seed=args.random_seed,
                gpmodel=gpmodels[metric_name],
                fit_parameters=fit_parameters,
                num_fantasy_samples=args.num_fantasy_samples,
                normalize_targets=args.normalize_targets,
                profiler=self._profiler,
                debug_log=self._debug_log,
                debug_fantasy_values=self._debug_fantasy_values)
            output_models[args.active_metric] = model
        if self._use_single_model:
            self._model = output_models[DEFAULT_METRIC]
        else:
            self._model = output_models
        # Note: This may be different than self._gpmodel.get_params(), since
        # the GaussProcSurrogateModel may append additional info
        self._assign_model_params(self._model)
        # DEBUG: Supplied values are used only once
        self._debug_fantasy_values = None
        if fit_parameters:
            # Keep copy of labeled data in order to avoid unnecessary
            # refitting
            self._candidate_evaluations = copy.copy(
                self._state.candidate_evaluations)

    def _assign_model_params(self, model):
        """
        Gets model parameters from model and assigns them to self._model_params.

        In the multi-model case, self._model_params is a dictionary mapping
        each model name to its parameters.
        """
        if self._use_single_model:
            self._model_params = model.get_params()
        else:
            self._model_params = {key: value.get_params()
                                  for key, value in model.items()}

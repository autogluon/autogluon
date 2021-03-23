from typing import Optional, List
import logging
import numpy as np

from .base_classes import NextCandidatesAlgorithm
from ..autogluon.gp_fifo_searcher import GET_CONFIG_RANDOM_RETRIES, \
    encode_state, create_initial_candidates_scorer
from ..autogluon.searcher_factory import gp_fifo_searcher_defaults, \
    gp_fifo_searcher_factory
from ..datatypes.common import Candidate, candidate_for_print, \
    CandidateEvaluation, PendingEvaluation
from ..datatypes.hp_ranges_cs import HyperparameterRanges_CS, \
    convert_hyperparameter_ranges, HyperparameterRanges_Impl
from ..datatypes.tuning_job_state import TuningJobState
from ..models.gpmodel_transformers import \
    GPModelPendingCandidateStateTransformer
from ..models.gpmodel_skipopt import AlwaysSkipPredicate
from ..tuning_algorithms.bo_algorithm import BayesianOptimizationAlgorithm
from ..tuning_algorithms.common import compute_blacklisted_candidates
from ..utils.debug_log import DebugLogPrinter
from ..utils.duplicate_detector import DuplicateDetectorIdentical
from ..utils.simple_profiler import SimpleProfiler
from ....utils.default_arguments import check_and_merge_defaults

logger = logging.getLogger(__name__)


def _convert_to_configspace(configs, hp_ranges: HyperparameterRanges_CS, keys):
    return [hp_ranges.tuple_to_config(x, keys=keys) for x in configs]


def _convert_from_configspace(
        configs, hp_ranges: HyperparameterRanges_CS, keys):
    return [hp_ranges.config_to_tuple(x, keys=keys) for x in configs]


class InitialConfigurationsAlgorithm(NextCandidatesAlgorithm):
    """
    Default algorithm for choosing the initial configurations.

    """
    def __init__(self, state: TuningJobState,
                 random_state: np.random.RandomState,
                 num_requested_candidates: int = 1,
                 debug_log: Optional[DebugLogPrinter] = None,
                 profiler: Optional[SimpleProfiler] = None,
                 first_is_default: bool = True,
                 hp_ranges_keys=None):
        self.state = state
        self.random_state = random_state
        self.num_requested_candidates = num_requested_candidates
        self.debug_log = debug_log
        self.profiler = profiler
        self.do_profile = (profiler is not None)
        self.first_is_default = first_is_default
        self.hp_ranges_keys = hp_ranges_keys

    def next_candidates(self) -> List[Candidate]:
        if self.do_profile:
            # Start new profiler block
            meta = {
                'fit_hyperparams': False,
                'num_observed': len(self.state.candidate_evaluations),
                'num_pending': len(self.state.pending_evaluations)
            }
            self.profiler.begin_block(meta)
            self.profiler.start('all')
        hp_ranges = self.state.hp_ranges
        blacklisted_candidates = compute_blacklisted_candidates(self.state)
        if self.debug_log is not None:
            self.debug_log.start_get_config('random')
        configs = []
        for iter in range(self.num_requested_candidates):
            config = None
            if self.first_is_default and (not blacklisted_candidates):
                # Use default configuration if there is one specified
                default_config = \
                    hp_ranges.config_space.get_default_configuration()
                if default_config and len(default_config.get_dictionary()) > 0:
                    config = default_config
                    if self.debug_log is not None:
                        logger.info("Start with default config:\n{}".format(
                            candidate_for_print(config)))
            if config is None:
                if self.do_profile:
                    self.profiler.start('random')
                for _ in range(GET_CONFIG_RANDOM_RETRIES):
                    _config = hp_ranges.random_candidate(self.random_state)
                    if _config not in blacklisted_candidates:
                        config = _config
                        break
                if self.do_profile:
                    self.profiler.stop('random')
                if config is None:
                    raise AssertionError(
                        "Failed to sample a configuration not already chosen "
                        "before. Maybe there are no free configurations left? "
                        "The blacklist size is {}".format(len(blacklisted_candidates)))
            configs.append(config)
            blacklisted_candidates.add(config)

        if self.hp_ranges_keys is not None:
            # Convert configs back to tuples
            configs = _convert_from_configspace(
                configs, hp_ranges, self.hp_ranges_keys)
        return configs


class BOWrapperAlgorithm(NextCandidatesAlgorithm):
    def __init__(self, algorithm: NextCandidatesAlgorithm,
                 hp_ranges: HyperparameterRanges_CS, hp_ranges_keys):
        self.algorithm = algorithm
        self.hp_ranges = hp_ranges
        self.hp_ranges_keys = hp_ranges_keys

    def next_candidates(self) -> List[Candidate]:
        configs = self.algorithm.next_candidates()
        return _convert_from_configspace(
            configs, self.hp_ranges, self.hp_ranges_keys)


def default_next_candidates_algorithm(
        state: TuningJobState,
        num_requested_candidates: int = 1, **kwargs) -> NextCandidatesAlgorithm:
    assert num_requested_candidates >= 1, \
        "num_requested_candidates = {} invalid (must be positive integer)".format(
            num_requested_candidates)
    hp_ranges = state.hp_ranges
    use_configspace = isinstance(hp_ranges, HyperparameterRanges_CS)
    hp_ranges_keys = None
    if not use_configspace:
        # Convert hp_ranges and state to use ConfigSpace
        assert isinstance(state.hp_ranges, HyperparameterRanges_Impl)
        hp_ranges, hp_ranges_keys = convert_hyperparameter_ranges(
            state.hp_ranges)
        eval_configs = _convert_to_configspace(
            [x.candidate for x in state.candidate_evaluations], hp_ranges,
            hp_ranges_keys)
        candidate_evaluations = [
            CandidateEvaluation(x, metrics=y.metrics)
            for x, y in zip(eval_configs, state.candidate_evaluations)]
        failed_candidates = _convert_to_configspace(
            state.failed_candidates, hp_ranges, hp_ranges_keys)
        pending_candidates = _convert_to_configspace(
            state.pending_candidates, hp_ranges, hp_ranges_keys)
        pending_evaluations = [
            PendingEvaluation(x) for x in pending_candidates]
        state = TuningJobState(
            hp_ranges=hp_ranges,
            candidate_evaluations=candidate_evaluations,
            failed_candidates=failed_candidates,
            pending_evaluations=pending_evaluations)
    # Deal with variables which are persistent across several calls
    _kwargs = kwargs.copy()
    debug_log = None
    if 'debug_log' in _kwargs:
        debug_log = _kwargs['debug_log']
        del _kwargs['debug_log']
    profiler = None
    if 'profiler' in _kwargs:
        profiler = _kwargs['profiler']
        del _kwargs['profiler']
    # Impute default values
    _kwargs['configspace'] = hp_ranges.config_space
    _kwargs = check_and_merge_defaults(
        _kwargs, *gp_fifo_searcher_defaults(), dict_name='kwargs')
    random_state = np.random.RandomState(_kwargs['random_seed'])

    len_blacklist = len(state.candidate_evaluations) + \
        len(state.pending_evaluations) + len(state.failed_candidates)
    pick_random = (len_blacklist < _kwargs['num_init_random']) or \
                  (not state.candidate_evaluations)
    if pick_random:
        # Note: If num_requested_candidates > 1, all of them are either picked
        # as initial candidates, or by BO
        return InitialConfigurationsAlgorithm(
            state=state,
            random_state=random_state,
            num_requested_candidates=num_requested_candidates,
            debug_log=debug_log,
            profiler=profiler,
            first_is_default=_kwargs['first_is_default'])
    else:
        # This is a little tricky here, and can for sure be refactored. In an
        # AutoGluon searcher, objects like _gp_searcher are persistent, they
        # are re-used for every 'next_candidates' call. Here, everything has to
        # be created from scratch for every 'next_candidates' call. I am
        # supporting this in 'GPFIFOSearcher', but only for checkpointing. This
        # is what I am using here.

        # Create GPFIFOSearcher, to pull components out
        # However, this searcher will be created with an empty TuningJobState.
        # It also does not have debug_log and profiler.
        gp_searcher = gp_fifo_searcher_factory(**_kwargs)
        # Now, I have to initialize the TuningJobState. To do this safely, I
        # use the (lightweight) mechanism for checkpointing
        gp_searcher_state = gp_searcher.get_state()
        gp_searcher_state['state'] = encode_state(state)
        gp_searcher = gp_searcher.clone_from_state(gp_searcher_state)
        # Finally, set debug_log and profiler
        gp_searcher.set_debug_log(debug_log)
        gp_searcher.set_profiler(profiler)
        # Copied from GPFIFOSearcher.get_config
        do_profile = (profiler is not None)
        if do_profile:
            profiler.start('gpmodel')
        # Note: Asking for the model triggers the posterior computation
        model = gp_searcher.state_transformer.model()
        if do_profile:
            profiler.stop('gpmodel')
        # Create BO algorithm
        blacklisted_candidates = compute_blacklisted_candidates(state)
        initial_candidates_scorer = create_initial_candidates_scorer(
            gp_searcher.initial_scoring, model, gp_searcher.acquisition_class,
            random_state)
        local_optimizer = gp_searcher.local_minimizer_class(
            state, model, gp_searcher.acquisition_class)
        pending_candidate_state_transformer = None
        if num_requested_candidates > 1:
            # Internally, if num_requested_candidates > 1, the candidates are
            # selected greedily. This needs model updates after each greedy
            # selection, because of one more pending evaluation.
            # Note: This code is quite complex. It would (IMHO) be simpler
            # to restrict the API such that 'next_candidates' can only return
            # a single config, and then just call it num_requested_candidates
            # times.
            state_transformer = gp_searcher.state_transformer
            pending_candidate_state_transformer = \
                GPModelPendingCandidateStateTransformer(
                    gpmodel=state_transformer._gpmodel,
                    init_state=state,
                    model_args=state_transformer._model_args,
                    skip_optimization=AlwaysSkipPredicate(),
                    profiler=profiler,
                    debug_log=debug_log)

        algorithm = BayesianOptimizationAlgorithm(
            initial_candidates_generator=gp_searcher.random_generator,
            initial_candidates_scorer=initial_candidates_scorer,
            num_initial_candidates=gp_searcher.num_initial_candidates,
            local_optimizer=local_optimizer,
            pending_candidate_state_transformer=pending_candidate_state_transformer,
            blacklisted_candidates=blacklisted_candidates,
            num_requested_candidates=num_requested_candidates,
            greedy_batch_selection=True,
            duplicate_detector=DuplicateDetectorIdentical(),
            profiler=profiler,
            sample_unique_candidates=False,
            debug_log=debug_log)
        if not use_configspace:
            algorithm = BOWrapperAlgorithm(
                algorithm=algorithm, hp_ranges=hp_ranges,
                hp_ranges_keys=hp_ranges_keys)
        return algorithm

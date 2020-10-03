import numpy as np
from typing import Callable, Type, NamedTuple, Optional
import copy
import logging
import ConfigSpace as CS

from .debug_log import DebugLogPrinter
from .gp_profiling import GPMXNetSimpleProfiler
from .hp_ranges import HyperparameterRanges_CS
from ..datatypes.common import CandidateEvaluation, Candidate, candidate_for_print, PendingEvaluation
from ..datatypes.hp_ranges import HyperparameterRanges
from ..datatypes.tuning_job_state import TuningJobState
from ..models.gpmxnet import GPModel, GPMXNetModel
from ..models.gpmxnet_skipopt import SkipOptimizationPredicate
from ..models.gpmxnet_transformers import GPMXNetPendingCandidateStateTransformer, GPMXNetModelArgs
from ..tuning_algorithms.base_classes import LocalOptimizer, AcquisitionFunction, ScoringFunction
from ..tuning_algorithms.bo_algorithm import BayesianOptimizationAlgorithm
from ..tuning_algorithms.bo_algorithm_components import IndependentThompsonSampling
from ..tuning_algorithms.common import RandomStatefulCandidateGenerator, compute_blacklisted_candidates
from ..tuning_algorithms.default_algorithm import dictionarize_objective, DEFAULT_METRIC, DEFAULT_LOCAL_OPTIMIZER_CLASS, DEFAULT_NUM_INITIAL_CANDIDATES, DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS
from ..utils.duplicate_detector import DuplicateDetectorIdentical

logger = logging.getLogger(__name__)

GET_CONFIG_RANDOM_RETRIES = 50


class MapReward(NamedTuple):
    forward: Callable[[float], float]
    reverse: Callable[[float], float]

    def __call__(self, x: float) -> float:
        return self.forward(x)


def accumulate_profiling_record(
        cum_records: dict, profiler: GPMXNetSimpleProfiler, pick_random: bool):
    # Pull out profiling data from this block
    block_id = profiler.id_counter - 1
    curr_record = {
        r.tag: r.duration for r in profiler.records
        if r.id == block_id}
    if pick_random:
        curr_record['num_random'] = 1
        curr_record['total_all'] = curr_record['random']
    else:
        curr_record['num_model'] = 1
    # Sum up entries
    for k, v in curr_record.items():
        if k in cum_records:
            cum_records[k] += v
        else:
            cum_records[k] = v


SUPPORTED_INITIAL_SCORING = {
    'thompson_indep',
    'acq_func'}


DEFAULT_INITIAL_SCORING = 'thompson_indep'


def create_initial_candidates_scorer(
        initial_scoring: str, model: GPMXNetModel,
        acquisition_class: Type[AcquisitionFunction],
        random_state: np.random.RandomState) -> ScoringFunction:
    if initial_scoring == 'thompson_indep':
        return IndependentThompsonSampling(model, random_state=random_state)
    else:
        return acquisition_class(model)


def check_initial_candidates_scorer(initial_scoring: str) -> str:
    if initial_scoring is None:
        return DEFAULT_INITIAL_SCORING
    else:
        assert initial_scoring in SUPPORTED_INITIAL_SCORING, \
            "initial_scoring = '{}' is not supported".format(
                initial_scoring)
        return initial_scoring


class GPFIFOSearcher(object):
    """
    Supports standard GP-based hyperparameter optimization, when used with a
    FIFO scheduler.

    """
    def __init__(
            self, hp_ranges: HyperparameterRanges, random_seed: int,
            gpmodel: GPModel, model_args: GPMXNetModelArgs,
            map_reward: MapReward,
            acquisition_class: Type[AcquisitionFunction],
            init_state: TuningJobState = None,
            local_minimizer_class: Type[LocalOptimizer] = DEFAULT_LOCAL_OPTIMIZER_CLASS,
            skip_optimization: SkipOptimizationPredicate = None,
            num_initial_candidates: int = DEFAULT_NUM_INITIAL_CANDIDATES,
            num_initial_random_choices: int = DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
            initial_scoring: Optional[str] = None,
            profiler: Optional[GPMXNetSimpleProfiler] = None,
            first_is_default: bool = True,
            debug_log: Optional[DebugLogPrinter] = None):
        """
        Note that the GPMXNetModel is created on demand (by the state
        transformer) in get_config, along with components needed for the BO
        algorithm.

        The searcher is supposed to maximize reward, while internally, the
        criterion is minimized. map_reward maps reward to internal criterion, it
        must be strictly decreasing.

        :param hp_ranges: Configuration space without resource attribute
        :param random_seed:
        :param gpmodel: GP regression model
        :param model_args: Arguments for GPMXNet model creation
        :param map_reward: Function mapping reward to criterion to be minimized
        :param acquisition_class: Type for acquisition function
        :param init_state: TuningJobState to start from (default is empty)
        :param local_minimizer_class: Type for local minimizer
        :param skip_optimization: Predicate, see
            GPMXNetPendingCandidateStateTransformer
        :param num_initial_candidates: See BayesianOptimizationAlgorithm
        :param num_initial_random_choices: Configs are sampled at random until
            this many candidates received label feedback
        :param initial_scoring: Scoring function to rank initial candidates.
            Default: thompson_indep (independent Thompson sampling)
        :param profiler: If given, HPO computations are profiled
        :param first_is_default: If true, the first config to be evaluated is
            the default of the search space. Otherwise, the first is sampled
            at random
        :param debug_log: DebugLogPrinter for debug logging (optional)

        """
        self.hp_ranges = hp_ranges
        self.random_seed = random_seed
        self.num_initial_candidates = num_initial_candidates
        self.num_initial_random_choices = num_initial_random_choices
        self.map_reward = map_reward
        self.local_minimizer_class = local_minimizer_class
        self.acquisition_class = acquisition_class
        self.debug_log = debug_log
        self.initial_scoring = check_initial_candidates_scorer(initial_scoring)
        # Create state transformer
        # Initial state is empty (note that the state is mutable)
        if init_state is None:
            init_state = TuningJobState(
                hp_ranges=hp_ranges,
                candidate_evaluations=[],
                failed_candidates=[],
                pending_evaluations=[])
        else:
            assert hp_ranges is init_state.hp_ranges, \
                "hp_ranges and init_state.hp_ranges must be same object"
        self.state_transformer = GPMXNetPendingCandidateStateTransformer(
            gpmodel=gpmodel,
            init_state=init_state,
            model_args=model_args,
            skip_optimization=skip_optimization,
            profiler=profiler,
            debug_log=debug_log)
        self.random_state = np.random.RandomState(random_seed)
        self.random_generator = RandomStatefulCandidateGenerator(
            hp_ranges, random_state=self.random_state)
        self.profiler = profiler
        self.do_profile = (profiler is not None)
        self.first_is_default = first_is_default
        if first_is_default:
            assert isinstance(hp_ranges, HyperparameterRanges_CS), \
                "If first_is_default, must have hp_ranges of HyperparameterRanges_CS type"
        if debug_log is not None:
            assert isinstance(hp_ranges, HyperparameterRanges_CS), \
                "If debug_log is given, must have hp_ranges of HyperparameterRanges_CS type"
        # Sums up profiling records across all get_config calls
        self._profile_record = dict()
        if debug_log is not None:
            deb_msg = "[GPFIFOSearcher.__init__]\n"
            deb_msg += ("- acquisition_class = {}\n".format(acquisition_class))
            deb_msg += ("- local_minimizer_class = {}\n".format(local_minimizer_class))
            deb_msg += ("- num_initial_candidates = {}\n".format(num_initial_candidates))
            deb_msg += ("- num_initial_random_choices = {}\n".format(num_initial_random_choices))
            deb_msg += ("- initial_scoring = {}\n".format(self.initial_scoring))
            deb_msg += ("- first_is_default = {}".format(first_is_default))
            logger.info(deb_msg)

    def update(self, config: Candidate, reward: float):
        """
        Registers new datapoint at config, with reward reward.
        Note that in general, config should previously have been registered as
        pending (register_pending). If so, it is switched from pending
        to labeled. If not, it is considered directly labeled.

        :param config:
        :param reward:
        """
        crit_val = self.map_reward(reward)
        self.state_transformer.label_candidate(CandidateEvaluation(
            candidate=copy.deepcopy(config),
            metrics=dictionarize_objective(crit_val)))
        if self.debug_log is not None:
            config_id = self.debug_log.config_id(config)
            msg = "Update for config_id {}: reward = {}, crit_val = {}".format(
                config_id, reward, crit_val)
            logger.info(msg)

    def register_pending(self, config: Candidate):
        """
        Registers config as pending. This means the corresponding evaluation
        task is running. Once it finishes, update is called for config.

        """
        # It is OK for the candidate already to be registered as pending, in
        # which case we do nothing
        state = self.state_transformer.state
        if config not in state.pending_candidates:
            if config in (x.candidate for x in state.candidate_evaluations):
                evals = state.candidate_evaluations
                num_labeled = len(evals)
                pos_cand = next(
                    i for i, x in enumerate(evals) if x.candidate == config)
                error_msg = """
                This configuration is already registered as labeled:
                   Position of labeled candidate: {} of {}
                   Label value: {}
                """.format(
                    pos_cand, num_labeled,
                    evals[pos_cand].metrics[DEFAULT_METRIC])
                assert False, error_msg
            self.state_transformer.append_candidate(config)

    def get_config(self) -> Candidate:
        """
        Runs Bayesian optimization in order to suggest the next config to evaluate.

        :return: Next config to evaluate at
        """
        state = self.state_transformer.state
        if self.do_profile:
            fit_hyperparams = not self.state_transformer.skip_optimization(
                state)
            self.profiler.set_state(state, fit_hyperparams)
        blacklisted_candidates = compute_blacklisted_candidates(state)
        pick_random = (len(blacklisted_candidates) < self.num_initial_random_choices) or \
            (not state.candidate_evaluations)
        if self.debug_log is not None:
            self.debug_log.start_get_config('random' if pick_random else 'BO')
        if pick_random:
            config = None
            if self.first_is_default and (not blacklisted_candidates):
                # Use default configuration if there is one specified
                default_config = self.hp_ranges.config_space.get_default_configuration()
                if default_config and len(default_config.get_dictionary()) > 0:
                    config = default_config
                    if self.debug_log is not None:
                        logger.info("Start with default config:\n{}".format(
                            candidate_for_print(config)))
            if config is None:
                if self.do_profile:
                    self.profiler.start('random')
                for _ in range(GET_CONFIG_RANDOM_RETRIES):
                    _config = self.hp_ranges.random_candidate(self.random_state)
                    if _config not in blacklisted_candidates:
                        config = _config
                        break
                if config is None:
                    raise AssertionError(
                        "Failed to sample a configuration not already chosen "
                        "before. Maybe there are no free configurations left? "
                        "The blacklist size is {}".format(len(blacklisted_candidates)))
                if self.do_profile:
                    self.profiler.stop('random')
        else:
            # Obtain current GPMXNetModel from state transformer. Based on
            # this, the BO algorithm components can be constructed
            state = self.state_transformer.state
            if self.do_profile:
                self.profiler.start('total_all')
                self.profiler.start('total_update')
            # Note: Asking for the model triggers the posterior computation
            model = self.state_transformer.model()
            if self.do_profile:
                self.profiler.stop('total_update')
            # Create BO algorithm
            initial_candidates_scorer = create_initial_candidates_scorer(
                self.initial_scoring, model, self.acquisition_class,
                self.random_state)
            local_optimizer = self.local_minimizer_class(
                state, model, self.acquisition_class)
            # Make sure not to use the same random seed for each call:
            #random_seed = compute_random_seed({'0': state}, self.random_seed)
            bo_algorithm = BayesianOptimizationAlgorithm(
                initial_candidates_generator=self.random_generator,
                initial_candidates_scorer=initial_candidates_scorer,
                num_initial_candidates=self.num_initial_candidates,
                local_optimizer=local_optimizer,
                pending_candidate_state_transformer=None,
                blacklisted_candidates=blacklisted_candidates,
                num_requested_candidates=1,
                greedy_batch_selection=False,
                duplicate_detector=DuplicateDetectorIdentical(),
                profiler=self.profiler,
                sample_unique_candidates=False,
                debug_log=self.debug_log)
            # Next candidate decision
            if self.do_profile:
                self.profiler.start('total_nextcand')
            _config = bo_algorithm.next_candidates()
            if len(_config) == 0:
                raise AssertionError(
                    "Failed to find a configuration not already chosen "
                    "before. Maybe there are no free configurations left? "
                    "The blacklist size is {}".format(len(blacklisted_candidates)))
            config = _config[0]
            if self.do_profile:
                self.profiler.stop('total_nextcand')
            if self.do_profile:
                self.profiler.stop('total_all')

        if self.debug_log is not None:
            self.debug_log.set_final_config(config)
            # All get_config debug log info is only written here
            self.debug_log.write_block()
        if self.do_profile:
            self.profiler.clear()
            # Pull out profiling data from this block, add to _profile_record
            accumulate_profiling_record(
                self._profile_record, self.profiler, pick_random)

        return config

    def evaluation_failed(self, config: Candidate):
        # Remove pending candidate
        self.state_transformer.drop_candidate(config)
        # Mark config as failed (which means it will be blacklisted in
        # future get_config calls)
        self.state_transformer.mark_candidate_failed(config)

    def dataset_size(self):
        return len(self.state_transformer.state.candidate_evaluations)

    def cumulative_profile_record(self):
        """
        If profiling is activated, we sum up the profiling blocks for each
        call of get_config and return it as dict. See get_config for what
        is recorded:
        - num_random: Number of get_config calls with random selection
        - num_model: Number of get_config calls with model-based selection
        - total_all: Sum of total times for all get_config calls
        """
        return self._profile_record

    def get_params(self):
        """
        Note: Once MCMC is supported, this method will have to be refactored.

        :return: Dictionary with current hyperparameter values
        """
        return self.state_transformer.get_params()

    def set_params(self, param_dict):
        self.state_transformer.set_params(param_dict)

    def get_state(self):
        """
        The mutable state consists of the GP model parameters, the
        TuningJobState, and the skip_optimization predicate (which can have a
        mutable state).
        We assume that skip_optimization can be pickled.

        """
        state = {
            'model_params': self.get_params(),
            'state': encode_state(self.state_transformer.state),
            'skip_optimization': self.state_transformer.skip_optimization,
            'random_state': self.random_state}
        if self.debug_log is not None:
            state['debug_log'] = self.debug_log.get_mutable_state()
        return state

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        model_args = self.state_transformer._model_args
        init_state = decode_state(state['state'], self.hp_ranges)
        skip_optimization = state['skip_optimization']
        new_searcher = GPFIFOSearcher(
            hp_ranges=self.hp_ranges,
            random_seed=self.random_seed,
            gpmodel=self.state_transformer._gpmodel,
            model_args=model_args,
            map_reward=self.map_reward,
            acquisition_class=self.acquisition_class,
            init_state=init_state,
            local_minimizer_class=self.local_minimizer_class,
            skip_optimization=skip_optimization,
            num_initial_candidates=self.num_initial_candidates,
            num_initial_random_choices=self.num_initial_random_choices,
            initial_scoring=self.initial_scoring,
            profiler=self.profiler,
            first_is_default=self.first_is_default,
            debug_log=self.debug_log)
        new_searcher.state_transformer.set_params(state['model_params'])
        new_searcher.random_state = state['random_state']
        new_searcher.random_generator.random_state = \
            state['random_state']
        if self.debug_log and 'debug_log' in state:
            new_searcher.debug_log.set_mutable_state(state['debug_log'])
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher


def encode_state(state: TuningJobState) -> dict:
    assert isinstance(state.hp_ranges, HyperparameterRanges_CS), \
        "Must have hp_ranges of HyperparameterRanges_CS type"
    candidate_evaluations = [
        {'candidate': eval.candidate.get_dictionary(),
         'metrics': eval.metrics}
        for eval in state.candidate_evaluations]
    failed_candidates = [x.get_dictionary() for x in state.failed_candidates]
    pending_evaluations = [
        eval.candidate.get_dictionary() for eval in state.pending_evaluations]
    return {
        'candidate_evaluations': candidate_evaluations,
        'failed_candidates': failed_candidates,
        'pending_evaluations': pending_evaluations}


def decode_state(enc_state: dict, hp_ranges: HyperparameterRanges_CS) \
        -> TuningJobState:
    assert isinstance(hp_ranges, HyperparameterRanges_CS), \
        "Must have hp_ranges of HyperparameterRanges_CS type"
    config_space = hp_ranges.config_space

    def to_cs(x):
        return CS.Configuration(config_space, values=x)

    candidate_evaluations = [
        CandidateEvaluation(to_cs(x['candidate']), x['metrics'])
        for x in enc_state['candidate_evaluations']]
    failed_candidates = [to_cs(x) for x in enc_state['failed_candidates']]
    pending_evaluations = [
        PendingEvaluation(to_cs(x)) for x in enc_state['pending_evaluations']]
    return TuningJobState(
        hp_ranges=hp_ranges,
        candidate_evaluations=candidate_evaluations,
        failed_candidates=failed_candidates,
        pending_evaluations=pending_evaluations)


def map_reward(const=1.0) -> MapReward:
    """
    Factory for map_reward argument in GPMultiFidelitySearcher.
    """
    def const_minus_x(x):
        return const - x

    return MapReward(forward=const_minus_x, reverse=const_minus_x)

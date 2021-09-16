import numpy as np
from typing import Callable, Type, Tuple, Set, Iterable, Optional, Any
import ConfigSpace as CS
from collections import Counter
import logging

from .gp_fifo_searcher import GET_CONFIG_RANDOM_RETRIES, GPSearcher, MapReward, \
    create_initial_candidates_scorer, decode_state
from ..datatypes.common import PendingEvaluation, candidate_for_print
from ..datatypes.config_ext import ExtendedConfiguration
from ..datatypes.hp_ranges_cs import HyperparameterRanges_CS
from ..datatypes.tuning_job_state import TuningJobState
from ..models.gp_model import GPModel
from ..models.gpmodel_skipopt import SkipOptimizationPredicate
from ..models.gpmodel_transformers import GPModelArgs
from ..tuning_algorithms.base_classes import LocalOptimizer, \
    AcquisitionFunction, DEFAULT_METRIC
from ..tuning_algorithms.bo_algorithm import BayesianOptimizationAlgorithm
from ..tuning_algorithms.common import compute_blacklisted_candidates
from ..tuning_algorithms.defaults import DEFAULT_LOCAL_OPTIMIZER_CLASS, DEFAULT_NUM_INITIAL_CANDIDATES, \
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS
from ..utils.debug_log import DebugLogPrinter
from ..utils.duplicate_detector import DuplicateDetectorIdentical
from ..utils.simple_profiler import SimpleProfiler

logger = logging.getLogger(__name__)


class GPMultiFidelitySearcher(GPSearcher):
    """
    Supports asynchronous multi-fidelity hyperparameter optimization, in the
    style of Hyperband or BOHB. Here, a joint GP surrogate model is fit to
    observations made at all resource levels.

    """
    def __init__(
            self, hp_ranges: HyperparameterRanges_CS,
            resource_attr_key: str,
            resource_attr_range: Tuple[int, int],
            random_seed: int,
            gpmodel: GPModel, model_args: GPModelArgs,
            map_reward: MapReward,
            acquisition_class: Type[AcquisitionFunction],
            resource_for_acquisition: Callable[..., int],
            init_state: TuningJobState = None,
            local_minimizer_class: Type[LocalOptimizer] = DEFAULT_LOCAL_OPTIMIZER_CLASS,
            skip_optimization: SkipOptimizationPredicate = None,
            num_initial_candidates: int = DEFAULT_NUM_INITIAL_CANDIDATES,
            num_initial_random_choices: int = DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
            initial_scoring: Optional[str] = None,
            first_is_default: bool = True,
            debug_log: Optional[DebugLogPrinter] = None,
            cost_metric_name: Optional[str] = None,
            profiler: Optional[SimpleProfiler] = None,
            getconfig_callback: Optional[Callable[[dict], Any]] = None):
        """
        Note that the GPMXNetModel is created on demand (by the state
        transformer) in get_config, along with components needed for the BO
        algorithm.

        The configuration space is hp_ranges. This does not include the resource
        attribute, which is passed as result component instead, with key
        resource_attr_key. The GP model is over configuration and resource
        attribute, its configuration space is maintained in configspace_ext.

        The search for a next candidate in get_config is fixing the resource
        level, meaning that extended configs for which the acquisition function
        is evaluated, all have the same resource level. This level may depend on
        the current state, the function can be passed as
        resource_for_acquisition. Its signature is
            resource_for_acquisition(state, resource_attr_name, **kwargs) -> int,
        where state is TuningJobState.
        Example: resource_for_acquisition may count the number of labeled data
        at each resource level, and choose the largest level supported by
        enough data. Or information about bracket and first milestone may be
        passed from the scheduler via **kwargs.

        The searcher is supposed to maximize reward, while internally, the
        criterion is minimized. map_reward maps reward to internal criterion, it
        must be strictly decreasing.

        :param hp_ranges: Configuration space without resource attribute
        :param resource_attr_key: Key for resource attribute.
            Note: The resource attribute must be int valued
        :param resource_attr_range: Range (lower, upper) for resource
            attribute( (must be int valued)
        :param random_seed:
        :param gpmodel: GP regression model
        :param model_args: Arguments for GPMXNet model creation
        :param map_reward: Function mapping reward to criterion to be minimized
        :param acquisition_class: Type for acquisition function
        :param resource_for_acquisition: See above
        :param init_state: TuningJobState to start from (default is empty).
            Here, init_state.hp_ranges must be equal to
            self.configspace_ext.hp_ranges_ext here (this is not checked)
        :param local_minimizer_class: Type for local minimizer
        :param skip_optimization: Predicate, see
            GPMXNetPendingCandidateStateTransformer
        :param num_initial_candidates: See BayesianOptimizationAlgorithm
        :param num_initial_random_choices: Configs are sampled at random until
            this many candidates received label feedback
        :param initial_scoring: Scoring function to rank initial candidates.
            Default: thompson_indep (independent Thompson sampling)
        :param first_is_default: If true, the first result of get_config is the
            default config of hp_ranges
        :param debug_log: DebugLogPrinter for debug logging (optional)
        :param cost_metric_name: If an `elapsed_time` record is passed to
            `update`, the value is entered in the state using this key
        :param profiler: If given, HPO computations are profiled
        :param getconfig_callback: If given, this callback function is called
            at the end of each get_config. It receives the chosen config
            (as dict) as single argument. One use case for this callback is
            to store profiler records.

        """
        assert isinstance(hp_ranges, HyperparameterRanges_CS), \
            "hp_ranges must be of type HyperparameterRanges_CS"
        # Extended configuration space including resource attribute
        self.configspace_ext = ExtendedConfiguration(
            hp_ranges, resource_attr_key, resource_attr_range)
        super().__init__(
            hp_ranges, random_seed, gpmodel, model_args, map_reward,
            acquisition_class, init_state, local_minimizer_class,
            skip_optimization, num_initial_candidates,
            num_initial_random_choices, initial_scoring, first_is_default,
            debug_log, cost_metric_name, profiler, getconfig_callback)

        self.resource_for_acquisition = resource_for_acquisition
        self._gpmodel = gpmodel
        if debug_log is not None:
            # Configure DebugLogPrinter
            debug_log.set_configspace_ext(self.configspace_ext)

    def _hp_ranges_ext(self):
        return self.configspace_ext.hp_ranges_ext

    def _config_ext_update(self, config, **kwargs):
        return self.configspace_ext.get(config, kwargs['resource'])

    def register_pending(self, config: CS.Configuration, milestone: int):
        """
        Registers config as pending for resource level milestone. This means
        the corresponding evaluation task is running and should reach that
        level later, when update is called for it.

        :param config:
        :param milestone:
        """
        # It is OK for the candidate already to be registered as pending, in
        # which case we do nothing
        state = self.state_transformer.state
        config_ext = self.configspace_ext.get(config, milestone)
        if config_ext not in state.pending_candidates:
            if config_ext in (x.candidate for x in state.candidate_evaluations):
                evals = state.candidate_evaluations
                num_labeled = len(evals)
                pos_cand = next(
                    i for i, x in enumerate(evals) if x.candidate == config_ext)
                error_msg = """
                This configuration is already registered as labeled:
                   Position of labeled candidate: {} of {}
                   Label value: {}
                   Resource level: {}
                """.format(
                    pos_cand, num_labeled,
                    evals[pos_cand].metrics[DEFAULT_METRIC], milestone)
                assert False, error_msg
            self.state_transformer.append_candidate(config_ext)

    def _get_unique_candidates(
            self, candidate_list: Iterable[CS.Configuration],
            target_resource: int) -> Set[CS.Configuration]:
        return set(self.configspace_ext.remap_resource(x, target_resource)
                   for x in candidate_list)

    def _get_blacklisted_candidates(self, target_resource: int) -> Set[CS.Configuration]:
        """
        We want to blacklist all configurations which are labeled or pending at
        any resource level. As the search affected by the blacklist happens at
        resource level target_resource, the candidates have to be modified
        accordingly

        """
        return self._get_unique_candidates(
            compute_blacklisted_candidates(self.state_transformer.state),
            target_resource)

    def _fix_resource_attribute(self, resource_attr_value: int):
        self.configspace_ext.hp_ranges_ext.value_for_last_pos = \
            resource_attr_value

    def get_config(self, **kwargs) -> CS.Configuration:
        """
        Runs Bayesian optimization in order to suggest the next config to evaluate.

        :return: Next config to evaluate at

        """
        state = self.state_transformer.state
        if self.do_profile:
            # Start new profiler block
            meta = {
                'fit_hyperparams': not self.state_transformer.skip_optimization(
                    state),
                'num_observed': len(state.candidate_evaluations),
                'num_pending': len(state.pending_evaluations)
            }
            self.profiler.begin_block(meta)
            self.profiler.start('all')
        # Fix resource attribute during the search to the value obtained from
        # self.resource_for_acquisition. Compute blacklisted_candidates.
        if state.candidate_evaluations:
            target_resource = self.resource_for_acquisition(
                state, self.configspace_ext.resource_attr_name, **kwargs)
        else:
            # Any valid value works here:
            target_resource = self.configspace_ext.resource_attr_range[0]
        blacklisted_candidates = self._get_blacklisted_candidates(target_resource)
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
                config, _ = draw_random_candidate(
                    blacklisted_candidates, self.configspace_ext,
                    self.random_state, target_resource)
                if self.do_profile:
                    self.profiler.stop('random')
        else:
            # Obtain current SurrogateModel from state transformer. Based on
            # this, the BO algorithm components can be constructed
            state = self.state_transformer.state
            if self.do_profile:
                self.profiler.push_prefix('getconfig')
                self.profiler.start('all')
                self.profiler.start('gpmodel')
            # Note: Asking for the model triggers the posterior computation
            model = self.state_transformer.model()
            if self.do_profile:
                self.profiler.stop('gpmodel')
            # BO should only search over configs at resource level
            # target_resource
            self._fix_resource_attribute(target_resource)
            if self.debug_log is not None:
                self.debug_log.append_extra(
                    "Score values computed at target_resource = {}".format(
                        target_resource))
            # Create BO algorithm
            initial_candidates_scorer = create_initial_candidates_scorer(
                self.initial_scoring, model, self.acquisition_class,
                self.random_state)
            local_optimizer = self.local_minimizer_class(
                state, model, self.acquisition_class)
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
            _config = bo_algorithm.next_candidates()
            assert len(_config) > 0, \
                ("Failed to find a configuration not already chosen "
                 "before. Maybe there are no free configurations left? "
                 "The blacklist size is {}".format(len(blacklisted_candidates)))
            next_config = _config[0]
            # Remove resource attribute
            config = self.configspace_ext.remove_resource(next_config)
            if self.do_profile:
                self.profiler.stop('all')
                self.profiler.pop_prefix()  # getconfig

        if self.debug_log is not None:
            self.debug_log.set_final_config(config)
            # All get_config debug log info is only written here
            self.debug_log.write_block()
        if self.do_profile:
            self.profiler.stop('all')
            self.profiler.clear()
        if self.getconfig_callback is not None:
            self.getconfig_callback(config.get_dictionary())

        return config

    def evaluation_failed(self, config: CS.Configuration):
        # Remove all pending evaluations for config
        self.cleanup_pending(config)
        # Mark config as failed (which means it will be blacklisted in
        # future get_config calls)
        # We need to create an extended config by appending a resource
        # attribute. Its value does not matter, because of how the blacklist
        # is created
        lowest_attr_value = self.configspace_ext.resource_attr_range[0]
        config_ext = self.configspace_ext.get(config, lowest_attr_value)
        self.state_transformer.mark_candidate_failed(config_ext)

    def cleanup_pending(self, config: CS.Configuration):
        """
        Removes all pending candidates whose configuration (i.e., lacking the
        resource attribute) is equal to config.
        This should be called after an evaluation terminates. For various
        reasons (e.g., termination due to convergence), pending candidates
        for this evaluation may still be present.
        It is also called for a failed evaluation.

        :param config: See above
        """
        config_dct = config.get_dictionary()

        def filter_pred(x: PendingEvaluation) -> bool:
            x_dct = self.configspace_ext.remove_resource(
                x.candidate, as_dict=True)
            return x_dct != config_dct

        self.state_transformer.filter_pending_evaluations(filter_pred)

    def remove_case(self, config: CS.Configuration, resource: int):
        config_ext = self.configspace_ext.get(config, resource)
        self.state_transformer.drop_candidate(config_ext)

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        model_args = self.state_transformer._model_args
        init_state = decode_state(
            state['state'], self.configspace_ext.hp_ranges_ext)
        skip_optimization = state['skip_optimization']
        new_searcher = GPMultiFidelitySearcher(
            hp_ranges=self.hp_ranges,
            resource_attr_key=self.configspace_ext.resource_attr_key,
            resource_attr_range=self.configspace_ext.resource_attr_range,
            random_seed=self.random_seed,
            gpmodel=self.state_transformer._gpmodel,
            model_args=model_args,
            map_reward=self.map_reward,
            acquisition_class=self.acquisition_class,
            resource_for_acquisition=self.resource_for_acquisition,
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


def draw_random_candidate(
        blacklisted_candidates: Set[CS.Configuration],
        configspace_ext: ExtendedConfiguration,
        random_state: np.random.RandomState,
        target_resource: int) -> (CS.Configuration, CS.Configuration):
    config, config_ext = None, None
    for _ in range(GET_CONFIG_RANDOM_RETRIES):
        _config = configspace_ext.hp_ranges.random_candidate(random_state)
        # Test whether this config has already been considered
        _config_ext = configspace_ext.remap_resource(
            _config, target_resource)
        if _config_ext not in blacklisted_candidates:
            config = _config
            config_ext = _config_ext
            break
    if config is None:
        raise AssertionError(
            "Failed to sample a configuration not already chosen "
            "before. Maybe there are no free configurations left? "
            "The blacklist size is {}".format(len(blacklisted_candidates)))
    return config, config_ext


def _max_at_least_k(k, counter=None):
    """
    Get largest key of `counter` whose value is at least `k`.

    :param counter: dict with keys that support comparison operators
    :param k: lower bound
    :return: largest key of `counter`
    """
    assert counter, "counter must be non-empty and not `None`."
    return max(filter(lambda r: counter[r] >= k, counter.keys()),
               default=min(counter.keys()))


def resource_for_acquisition_bohb(threshold: int) -> Callable[..., int]:
    """
    Factory for resource_for_acquisition argument in GPMultiFidelitySearcher.
    Mirrors what is done in the BOHB algorithm. An integer threshold is given
    at construction.
    We return the largest resource level whose number of labeled candidates
    is >= the threshold. If none of the levels passes the threshold, the
    lowest level is returned.

    """
    def rfa_map(state: TuningJobState, resource_attr_name: str, **kwargs) -> int:
        assert state.candidate_evaluations, "state must not be empty"
        histogram = Counter([int(x.candidate.get_dictionary()[resource_attr_name])
                             for x in state.candidate_evaluations])
        return _max_at_least_k(threshold, histogram)

    return rfa_map


def resource_for_acquisition_first_milestone(
        state: TuningJobState, resource_attr_name: str, **kwargs) -> int:
    """
    Implementation for resource_for_acquisition argument in
    GPMultiFidelitySearcher. We assume that the scheduler passes the first
    milestone to be attained by the new config as kwargs['milestone']. This is
    returned as resource level.

    """
    assert 'milestone' in kwargs, \
        "Need the first milestone to be attained by the new config passed as "\
        "kwargs['milestone']. Use a scheduler which does that (in particular, "\
        "Hyperband_Scheduler)"
    return kwargs['milestone']

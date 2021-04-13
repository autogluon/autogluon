from typing import Callable, Dict, Type, Optional, Any
import copy
import logging

from .gp_fifo_searcher import GPFIFOSearcher, decode_state, \
    MapReward, GET_CONFIG_RANDOM_RETRIES, create_initial_candidates_scorer
from ..datatypes.common import CandidateEvaluation, Candidate, \
    candidate_for_print
from ..datatypes.hp_ranges import HyperparameterRanges
from ..datatypes.tuning_job_state import TuningJobState
from ..models.gp_model import GPModel
from ..models.gpmodel_skipopt import SkipOptimizationPredicate
from ..models.gpmodel_transformers import GPModelPendingCandidateStateTransformer, GPModelArgs
from ..tuning_algorithms.base_classes import DEFAULT_METRIC, DEFAULT_CONSTRAINT_METRIC, \
    LocalOptimizer, AcquisitionFunction
from ..tuning_algorithms.bo_algorithm import BayesianOptimizationAlgorithm
from ..tuning_algorithms.common import compute_blacklisted_candidates
from ..tuning_algorithms.defaults import DEFAULT_LOCAL_OPTIMIZER_CLASS, DEFAULT_NUM_INITIAL_CANDIDATES, \
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS
from ..utils.debug_log import DebugLogPrinter
from ..utils.duplicate_detector import DuplicateDetectorIdentical
from ..utils.simple_profiler import SimpleProfiler

logger = logging.getLogger(__name__)


class ConstrainedGPFIFOSearcher(GPFIFOSearcher):
    """
    Supports constrained GP-based hyperparameter optimization (to be used with a FIFO scheduler).

    """
    def __init__(self, hp_ranges: HyperparameterRanges, random_seed: int,
                 output_gpmodels: Dict[str, GPModel],
                 output_models_args: Dict[str, GPModelArgs],
                 map_reward: MapReward,
                 acquisition_class: Type[AcquisitionFunction],
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
        Note that the SurrogateModel is created on demand (by the state
        transformer) in get_config, along with components needed for the BO
        algorithm.

        The searcher is supposed to maximize reward, while internally, the
        criterion is minimized. map_reward maps reward to internal criterion, it
        must be strictly decreasing.

        :param hp_ranges: Configuration space without resource attribute
        :param random_seed:
        :param output_gpmodels: Dict with the GP regression model for the active metric and constraint metric
        :param output_models_args: Dict with arguments for GPMXNet model creation for the active and constraint metric
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
        :param first_is_default: If true, the first config to be evaluated is
            the default of the search space. Otherwise, the first is sampled
            at random
        :param debug_log: DebugLogPrinter for debug logging (optional)
        :param cost_metric_name: If an `elapsed_time` record is passed to
            `update`, the value is entered in the state using this key
        :param profiler: If given, HPO computations are profiled
        :param getconfig_callback: If given, this callback function is called
            at the end of each get_config. It receives the chosen config
            (as dict) as single argument. One use case for this callback is
            to store profiler records.
        """
        super().__init__(hp_ranges, random_seed, output_gpmodels[DEFAULT_METRIC],
                         output_models_args[DEFAULT_METRIC], map_reward, acquisition_class, init_state,
                         local_minimizer_class, skip_optimization, num_initial_candidates, num_initial_random_choices,
                         initial_scoring, first_is_default, debug_log, cost_metric_name, profiler, getconfig_callback)

        self.skip_optimization = skip_optimization
        self.state_transformer = GPModelPendingCandidateStateTransformer(
            gpmodel=output_gpmodels,
            init_state=self._init_state,
            model_args=output_models_args,
            skip_optimization=skip_optimization,
            profiler=profiler,
            debug_log=debug_log)

        if debug_log is not None:
            deb_msg = "[ConstrainedGPFIFOSearcher.__init__]\n"
            deb_msg += ("- acquisition_class = {}\n".format(acquisition_class))
            deb_msg += ("- local_minimizer_class = {}\n".format(local_minimizer_class))
            deb_msg += ("- num_initial_candidates = {}\n".format(num_initial_candidates))
            deb_msg += ("- num_initial_random_choices = {}\n".format(num_initial_random_choices))
            deb_msg += ("- initial_scoring = {}\n".format(self.initial_scoring))
            deb_msg += ("- first_is_default = {}".format(first_is_default))
            logger.info(deb_msg)

    def update(self, config: Candidate, reward: float, constraint: float, **kwargs):
        """
        Registers new datapoint at config, with reward reward.
        Note that in general, config should previously have been registered as
        pending (register_pending). If so, it is switched from pending
        to labeled. If not, it is considered directly labeled.
        """
        crit_val = self.map_reward(reward)
        constr_val = constraint
        metrics = {DEFAULT_METRIC: crit_val,
                   DEFAULT_CONSTRAINT_METRIC: constr_val}
        if 'elapsed_time' in kwargs:
            metrics[self.cost_metric_name] = kwargs['elapsed_time']
        self.state_transformer.label_candidate(CandidateEvaluation(
            candidate=copy.deepcopy(config), metrics=metrics))
        if self.debug_log is not None:
            config_id = self.debug_log.config_id(config)
            msg = "Update for config_id {}: reward = {}, crit_val = {}, constr_val = {}".format(
                config_id, reward, crit_val, constr_val)
            logger.info(msg)

    def get_config(self) -> Candidate:
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
                if self.do_profile:
                    self.profiler.stop('random')
                if config is None:
                    raise AssertionError(
                        "Failed to sample a configuration not already chosen "
                        "before. Maybe there are no free configurations left? "
                        "The blacklist size is {}".format(len(blacklisted_candidates)))
        else:
            # Obtain current SurrogateModel from state transformer. Based on
            # this, the BO algorithm components can be constructed
            if self.do_profile:
                self.profiler.push_prefix('getconfig')
                self.profiler.start('all')
                self.profiler.start('gpmodel')
            # Create the active and constraint models
            output_models = self.state_transformer.model()  # Asking for the model triggers the posterior computation
            if self.do_profile:
                self.profiler.stop('total_update')
            # Create BO algorithm
            initial_candidates_scorer = create_initial_candidates_scorer(initial_scoring=self.initial_scoring,
                                                                         model=output_models,
                                                                         acquisition_class=self.acquisition_class,
                                                                         random_state=self.random_state,
                                                                         active_output=DEFAULT_METRIC)
            local_optimizer = self.local_minimizer_class(state=state,
                                                         model=output_models,
                                                         acquisition_function_class=self.acquisition_class,
                                                         active_metric=DEFAULT_METRIC)
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
            if len(_config) == 0:
                raise AssertionError(
                    "Failed to find a configuration not already chosen "
                    "before. Maybe there are no free configurations left? "
                    "The blacklist size is {}".format(len(blacklisted_candidates)))
            config = _config[0]
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

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        output_models_args = self.state_transformer._model_args
        output_gpmodels = self.state_transformer._gpmodel
        init_state = decode_state(state['state'], self.hp_ranges)
        skip_optimization = state['skip_optimization']
        new_searcher = ConstrainedGPFIFOSearcher(
            hp_ranges=self.hp_ranges,
            random_seed=self.random_seed,
            output_gpmodels=output_gpmodels,
            output_models_args=output_gpmodels,
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

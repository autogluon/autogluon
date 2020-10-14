from typing import Set
import numpy as np

from .debug_log import DebugLogPrinter
from .gp_fifo_searcher import GPFIFOSearcher, map_reward, MapReward, DEFAULT_INITIAL_SCORING, SUPPORTED_INITIAL_SCORING
from .gp_multifidelity_searcher import GPMultiFidelitySearcher, resource_for_acquisition_bohb, resource_for_acquisition_first_milestone
from .gp_profiling import GPMXNetSimpleProfiler
from .hp_ranges import HyperparameterRanges_CS
from .model_factories import resource_kernel_factory
from ..datatypes.hp_ranges import HyperparameterRanges
from ..gpmxnet.constants import OptimizationConfig, DEFAULT_OPTIMIZATION_CONFIG
from ..gpmxnet.debug_gp_regression import DebugGPRegression
from ..gpmxnet.gp_regression import GaussianProcessRegression
from ..gpmxnet.kernel import Matern52
from ..gpmxnet.mean import ScalarMeanFunction
from ..models.gpmxnet_skipopt import SkipNoMaxResourcePredicate, SkipPeriodicallyPredicate
from ..models.gpmxnet_transformers import GPMXNetModelArgs
from ..models.nphead_acqfunc import EIAcquisitionFunction
from ..tuning_algorithms.default_algorithm import DEFAULT_METRIC, DEFAULT_NUM_INITIAL_CANDIDATES, DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS
from ....utils.default_arguments import Integer, Categorical, Boolean

__all__ = ['gp_fifo_searcher_factory',
           'gp_multifidelity_searcher_factory',
           'gp_fifo_searcher_defaults',
           'gp_multifidelity_searcher_defaults']


def _create_common_objects(**kwargs):
    # TODO: Validity checks on kwargs arguments
    scheduler = kwargs['scheduler']
    config_space = kwargs['configspace']
    is_hyperband = scheduler.startswith('hyperband')
    if kwargs.get('debug_use_hyperparameter_ranges', False):
        assert isinstance(config_space, HyperparameterRanges)
        assert not is_hyperband, \
            "Cannot use debug_use_hyperparameter_ranges with Hyperband scheduling"
        hp_ranges_cs = config_space
    else:
        import ConfigSpace as CS
        assert isinstance(config_space, CS.ConfigurationSpace)
        hp_ranges_cs = HyperparameterRanges_CS(config_space)
    # Note: This base random seed is used to create different random seeds for
    # each BO get_config call internally
    random_seed = kwargs.get('random_seed', 31415927)
    # Skip optimization predicate for GP surrogate model
    if kwargs.get('opt_skip_num_max_resource', False) and is_hyperband:
        skip_optimization = SkipNoMaxResourcePredicate(
            init_length=kwargs['opt_skip_init_length'],
            resource_attr_name=kwargs['resource_attribute'],
            max_resource=kwargs['max_epochs'])
    elif kwargs.get('opt_skip_period', 1) > 1:
        skip_optimization = SkipPeriodicallyPredicate(
            init_length=kwargs['opt_skip_init_length'],
            period=kwargs['opt_skip_period'])
    else:
        skip_optimization = None
    # Profiler
    if kwargs.get('profiler', False):
        profiler = GPMXNetSimpleProfiler()
    else:
        profiler = None
    # Conversion from reward to metric (strictly decreasing) and back
    _map_reward = kwargs.get('map_reward', '1_minus_x')
    if isinstance(_map_reward, str):
        _map_reward_name = _map_reward
        supp_map_reward = {'1_minus_x', 'minus_x'}
        assert _map_reward_name in supp_map_reward, \
            "This factory needs map_reward in {}".format(supp_map_reward)
        _map_reward: MapReward = map_reward(
            const=1.0 if _map_reward_name == '1_minus_x' else 0.0)
    else:
        assert isinstance(_map_reward, MapReward), \
            "map_reward must either be string or of MapReward type"
    if is_hyperband:
        # Note: 'min_reward' is needed only to support the exp-decay
        # surrogate model. If not given, it is assumed to be 0.
        min_reward = kwargs.get('min_reward', 0)
        max_metric_value = _map_reward(min_reward)
    else:
        max_metric_value = None
    opt_warmstart = kwargs.get('opt_warmstart', False)

    # Underlying GP regression model
    kernel = Matern52(dimension=hp_ranges_cs.ndarray_size(), ARD=True)
    mean = ScalarMeanFunction()
    if is_hyperband:
        kernel, mean = resource_kernel_factory(
            kwargs['gp_resource_kernel'],
            kernel_x=kernel, mean_x=mean,
            max_metric_value=max_metric_value)
    optimization_config = OptimizationConfig(
        lbfgs_tol=DEFAULT_OPTIMIZATION_CONFIG.lbfgs_tol,
        lbfgs_maxiter=kwargs['opt_maxiter'],
        verbose=kwargs['opt_verbose'],
        n_starts=kwargs['opt_nstarts'])
    debug_writer = None
    if kwargs.get('opt_debug_writer', False):
        fname_msk = kwargs.get('opt_debug_writer_fmask', 'debug_gpr_{}')
        debug_writer = DebugGPRegression(
            fname_msk=fname_msk, rolling_size=5)
    gpmodel = GaussianProcessRegression(
        kernel=kernel, mean=mean,
        optimization_config=optimization_config,
        fit_reset_params=not opt_warmstart,
        debug_writer=debug_writer)
    model_args = GPMXNetModelArgs(
        num_fantasy_samples=kwargs['num_fantasy_samples'],
        random_seed=random_seed,
        active_metric=DEFAULT_METRIC,
        normalize_targets=True)
    debug_log = DebugLogPrinter() if kwargs.get('debug_log', False) else None

    return hp_ranges_cs, random_seed, gpmodel, model_args, profiler, \
           _map_reward, skip_optimization, debug_log


def gp_fifo_searcher_factory(**kwargs) -> GPFIFOSearcher:
    """
    Creates GPFIFOSearcher object, based on kwargs equal to search_options
    passed to and extended by scheduler (see FIFOScheduler).

    Extensions of kwargs by the scheduler:
    - scheduler: Name of scheduler ('fifo', 'hyperband_*')
    - configspace: CS.ConfigurationSpace (or HyperparameterRanges if
      debug_use_hyperparameter_ranges is true)
    Only Hyperband schedulers:
    - resource_attribute: Name of resource (or time) attribute
    - min_epochs: Smallest resource value being rung level
    - max_epochs: Maximum resource value

    :param kwargs: search_options coming from scheduler
    :return: GPFIFOSearcher object

    """
    assert kwargs['scheduler'] == 'fifo', \
        "This factory needs scheduler = 'fifo' (instead of '{}')".format(
            kwargs['scheduler'])
    # Common objects
    hp_ranges_cs, random_seed, gpmodel, model_args, profiler, _map_reward, \
    skip_optimization, debug_log = \
        _create_common_objects(**kwargs)

    gp_searcher = GPFIFOSearcher(
        hp_ranges=hp_ranges_cs,
        random_seed=random_seed,
        gpmodel=gpmodel,
        model_args=model_args,
        map_reward=_map_reward,
        acquisition_class=EIAcquisitionFunction,
        skip_optimization=skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        profiler=profiler,
        first_is_default=kwargs['first_is_default'],
        debug_log=debug_log)
    return gp_searcher


def gp_multifidelity_searcher_factory(**kwargs) -> GPMultiFidelitySearcher:
    """
    Creates GPMultiFidelitySearcher object, based on kwargs equal to search_options
    passed to and extended by scheduler (see HyperbandScheduler).

    :param kwargs: search_options coming from scheduler
    :return: GPMultiFidelitySearcher object

    """
    supp_schedulers = {'hyperband_stopping', 'hyperband_promotion'}
    assert kwargs['scheduler'] in supp_schedulers, \
        "This factory needs scheduler in {} (instead of '{}')".format(
            supp_schedulers, kwargs['scheduler'])
    # Common objects
    hp_ranges_cs, random_seed, gpmodel, model_args, profiler, _map_reward,\
    skip_optimization, debug_log = \
        _create_common_objects(**kwargs)

    _resource_acq = kwargs.get('resource_acq', 'bohb')
    if _resource_acq == 'bohb':
        resource_for_acquisition = resource_for_acquisition_bohb(
            threshold=hp_ranges_cs.ndarray_size())
    else:
        assert _resource_acq == 'first', \
            "resource_acq must be 'bohb' or 'first'"
        resource_for_acquisition = resource_for_acquisition_first_milestone
    epoch_range = (kwargs['min_epochs'], kwargs['max_epochs'])
    gp_searcher = GPMultiFidelitySearcher(
        hp_ranges=hp_ranges_cs,
        resource_attr_key=kwargs['resource_attribute'],
        resource_attr_range=epoch_range,
        random_seed=random_seed,
        gpmodel=gpmodel,
        model_args=model_args,
        map_reward=_map_reward,
        acquisition_class=EIAcquisitionFunction,
        resource_for_acquisition=resource_for_acquisition,
        skip_optimization=skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        profiler=profiler,
        first_is_default=kwargs['first_is_default'],
        debug_log=debug_log)
    return gp_searcher


def _common_defaults(is_hyperband: bool) -> (Set[str], dict, dict):
    mandatory = set()

    default_options = {
        'random_seed': np.random.randint(10000),
        'opt_skip_init_length': 150,
        'opt_skip_period': 1,
        'profiler': False,
        'opt_maxiter': 50,
        'opt_nstarts': 2,
        'opt_warmstart': False,
        'opt_verbose': False,
        'opt_debug_writer': False,
        'num_fantasy_samples': 20,
        'scheduler': 'fifo',
        'num_init_random': DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
        'num_init_candidates': DEFAULT_NUM_INITIAL_CANDIDATES,
        'initial_scoring': DEFAULT_INITIAL_SCORING,
        'first_is_default': True,
        'debug_log': False}
    if is_hyperband:
        default_options['opt_skip_num_max_resource'] = False
        default_options['gp_resource_kernel'] = 'matern52'
        default_options['resource_acq'] = 'bohb'
        default_options['num_init_random'] = 10

    constraints = {
        'random_seed': Integer(),
        'opt_skip_init_length': Integer(0, None),
        'opt_skip_period': Integer(1, None),
        'profiler': Boolean(),
        'opt_maxiter': Integer(1, None),
        'opt_nstarts': Integer(1, None),
        'opt_warmstart': Boolean(),
        'opt_verbose': Boolean(),
        'opt_debug_writer': Boolean(),
        'num_fantasy_samples': Integer(1, None),
        'num_init_random': Integer(1, None),
        'num_init_candidates': Integer(5, None),
        'initial_scoring': Categorical(
            choices=tuple(SUPPORTED_INITIAL_SCORING)),
        'first_is_default': Boolean(),
        'debug_log': Boolean()}
    if is_hyperband:
        constraints['opt_skip_num_max_resource'] = Boolean()
        constraints['gp_resource_kernel'] = Categorical(choices=(
            'exp-decay-sum', 'exp-decay-combined', 'exp-decay-delta1',
            'matern52', 'matern52-res-warp'))
        constraints['resource_acq'] = Categorical(
            choices=('bohb', 'first'))

    return mandatory, default_options, constraints


def gp_fifo_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    GPFIFOSearcher.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=False)


def gp_multifidelity_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    GPMultiFidelitySearcher.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=True)

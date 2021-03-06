from .gp_searcher import GPFIFOSearcher, GPMultiFidelitySearcher
from .grid_searcher import GridSearcher
from .searcher import RandomSearcher
from .skopt_searcher import SKoptSearcher

__all__ = ['searcher_factory']

SEARCHER_CONFIGS = dict(
    random=dict(
        searcher_cls=RandomSearcher,
    ),
    skopt=dict(
        searcher_cls=SKoptSearcher,
        supported_schedulers={'fifo'},
    ),
    grid=dict(
        searcher_cls=GridSearcher,
    ),
    bayesopt=dict(
        # Gaussian process based Bayesian optimization
        # The searchers and their kwargs differ depending on the scheduler
        # type (fifo, hyperband_*)
        searcher_cls=lambda scheduler: GPFIFOSearcher if scheduler in ['fifo', 'local'] else GPMultiFidelitySearcher,
        supported_schedulers={'fifo', 'hyperband_stopping', 'hyperband_promotion', 'local'},
    ),
)


def searcher_factory(searcher_name, **kwargs):
    """Factory for searcher objects

    This function creates searcher objects from string argument name and
    additional kwargs. It is typically called in the constructor of a
    scheduler (see FIFOScheduler), which provides most of the required kwargs.

    Note: RLSearcher is not supported here, because its **kwargs are different
    from other searchers (no 'configspace').

    The 'bayesopt' searchers depend on which scheduler is used (GPFIFOSearcher
    for FIFOScheduler, GPMultiFidelitySearcher for HyperbandScheduler). They
    have many additional parameters in kwargs (see docstrings for
    GPFIFOSearcher, GPMultiFidelitySearcher).

    Parameters
    ----------
    searcher_name : str
        Searcher type. Supported are 'random' (RandomSearcher), 'skopt'
        (SKoptSearcher), 'grid' (GridSearcher), 'bayesopt' (GPFIFOSearcher,
        GPMultiFidelitySearcher)
    configspace : ConfigSpace.ConfigurationSpace
        Config space of train_fn, equal to train_fn.cs
    scheduler : str
        Scheduler type the searcher is used in. Supported are 'fifo'
        (FIFOScheduler), 'hyperband_stopping', 'hyperband_promotion'
        (HyperbandScheduler: type = 'stopping' or 'promotion')
    reward_attribute : str
        Name of reward attribute reported by train_fn, equal to
        reward_attr
    resource_attribute : str [only for HyperbandScheduler]
        Name of resource (or time) attribute reported by train_fn,
        equal to time_attr
    min_epochs : int [only for HyperbandScheduler]
        Minimum value of resource attribute, equal to grace_period
    max_epochs : int [only for HyperbandScheduler]
        Maximum value of resource attribute, equal to max_t
    debug_log : bool (default: False)
        Supported by 'random', 'bayesopt'. If True, both searcher and
        scheduler output an informative log, from which the configs chosen
        and decisions being made can be traced.
    first_is_default : bool (default: True)
        Supported by 'random', 'skopt', 'bayesopt'. If True, the first config
        to be evaluated is the default one of the config space. Otherwise, this
        first config is drawn at random.
    random_seed : int
        Seed for pseudo-random number generator used.

    See Also
    --------
    GPFIFOSearcher
    GPMultiFidelitySearcher
    """
    if searcher_name in SEARCHER_CONFIGS:
        searcher_config = SEARCHER_CONFIGS[searcher_name]
        searcher_cls = searcher_config['searcher_cls']
        scheduler = kwargs.get('scheduler')

        # Check if searcher_cls is a lambda - evaluate then
        if isinstance(searcher_cls, type(lambda: 0)):
            searcher_cls = searcher_cls(scheduler)

        if 'supported_schedulers' in searcher_config:
            supported_schedulers = searcher_config['supported_schedulers']
            assert scheduler is not None, "Scheduler must set search_options['scheduler']"
            assert scheduler in supported_schedulers, \
                f"Searcher '{searcher_name}' only works with schedulers {supported_schedulers} (not with '{scheduler}')"

        searcher = searcher_cls(**kwargs)
        return searcher
    else:
        raise AssertionError(f'searcher \'{searcher_name}\' is not supported')

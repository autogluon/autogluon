from .searcher import RandomSearcher
from .skopt_searcher import SKoptSearcher
from .grid_searcher import GridSearcher
from .gp_searcher import GPFIFOSearcher, GPMultiFidelitySearcher

__all__ = ['searcher_factory']


def searcher_factory(name, **kwargs):
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
    name : str
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
    if name == 'random':
        return RandomSearcher(**kwargs)
    elif name == 'skopt':
        _check_supported_scheduler(
            name, kwargs.get('scheduler'), {'fifo'})
        return SKoptSearcher(**kwargs)
    elif name == 'grid':
        return GridSearcher(**kwargs)
    elif name == 'bayesopt':
        # Gaussian process based Bayesian optimization
        # The searchers and their kwargs differ depending on the scheduler
        # type (fifo, hyperband_*)
        scheduler = _check_supported_scheduler(
            name, kwargs.get('scheduler'),
            {'fifo', 'hyperband_stopping', 'hyperband_promotion'})
        if scheduler == 'fifo':
            return GPFIFOSearcher(**kwargs)
        else:
            return GPMultiFidelitySearcher(**kwargs)
    else:
        raise AssertionError("name = '{}' not supported".format(name))


def _check_supported_scheduler(name, scheduler, supp_schedulers):
    assert scheduler is not None, \
        "Scheduler must set search_options['scheduler']"
    assert scheduler in supp_schedulers, \
        "Searcher '{}' only works with schedulers {} (not with '{}')".format(
            name, supp_schedulers, scheduler)
    return scheduler

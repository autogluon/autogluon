from .searcher import RandomSearcher
from .skopt_searcher import SKoptSearcher
from .grid_searcher import GridSearcher


def searcher_factory(name, **kwargs):
    """
    This function creates searcher objects from string argument name and
    additional kwargs. It is typically called in the constructor of a
    scheduler.
    kwargs['scheduler'] contains the name of the scheduler (e.g. 'fifo' for
    FIFOScheduler, or 'hyperband' for HyperbandScheduler). Some searchers
    do different things (and need different other kwargs) depending on the
    scheduler.

    Note: RLSearcher is not supported here, because its **kwargs are different
    from other searchers (no 'configspace').

    :param name: Supported are 'random', 'skopt', 'grid', 'bayesopt'
    :param kwargs: Arguments to constructor of BaseSearcher subclass
    :return: Searcher object (subclass of BaseSearcher)
    """
    if name == 'random':
        return RandomSearcher(**kwargs)
    elif name == 'skopt':
        _check_supported_scheduler(
            name, kwargs.get('scheduler'), {'fifo'})
        return SKoptSearcher(**kwargs)
    elif name == 'grid':
        return GridSearcher(**kwargs)
    else:
        raise AssertionError("name = '{}' not supported".format(name))

def _check_supported_scheduler(name, scheduler, supp_schedulers):
    assert scheduler is not None, \
        "Scheduler must set search_options['scheduler']"
    assert scheduler in supp_schedulers, \
        "Searcher '{}' only works with schedulers {} (not with '{}')".format(
            name, supp_schedulers, scheduler)
    return scheduler

from .local_grid_searcher import LocalGridSearcher
from .local_random_searcher import LocalRandomSearcher

__all__ = ["searcher_factory"]

SEARCHER_CONFIGS = dict(
    local_random=dict(searcher_cls=LocalRandomSearcher),
    local_grid=dict(
        searcher_cls=LocalGridSearcher,
    ),
    # Fall back to random search since Bayes searcher is not supported
    bayes=dict(searcher_cls=LocalRandomSearcher),
)


def searcher_factory(searcher_name, **kwargs):
    """Factory for searcher objects

    This function creates searcher objects from string argument name and
    additional kwargs. It is typically called in the constructor of a
    scheduler (see LocalSequentialScheduler), which provides most of the required kwargs.

    Parameters
    ----------
    searcher_name : str
        Searcher type. Supported are 'random' (RandomSearcher), 'grid' (GridSearcher)
    configspace : ConfigSpace.ConfigurationSpace
        Config space of train_fn, equal to train_fn.cs
    scheduler : str [Currently not used]
        Scheduler type the searcher is used in.
    reward_attribute : str
        Name of reward attribute reported by train_fn, equal to
        reward_attr
    debug_log : bool (default: False)
        Supported by 'random'. If True, both searcher and
        scheduler output an informative log, from which the configs chosen
        and decisions being made can be traced.
    first_is_default : bool (default: True)
        Supported by 'random'. If True, the first config
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
        searcher_cls = searcher_config["searcher_cls"]
        scheduler = kwargs.get("scheduler")

        # Check if searcher_cls is a lambda - evaluate then
        if isinstance(searcher_cls, type(lambda: 0)):
            searcher_cls = searcher_cls(scheduler)

        if "supported_schedulers" in searcher_config:
            supported_schedulers = searcher_config["supported_schedulers"]
            assert scheduler is not None, "Scheduler must set search_options['scheduler']"
            assert scheduler in supported_schedulers, (
                f"Searcher '{searcher_name}' only works with schedulers {supported_schedulers} (not with '{scheduler}')"
            )

        searcher = searcher_cls(**kwargs)
        return searcher
    else:
        raise AssertionError(f"searcher '{searcher_name}' is not supported")

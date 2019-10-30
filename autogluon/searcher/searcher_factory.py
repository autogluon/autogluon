from .searcher import RandomSearcher
from .skopt_searcher import SKoptSearcher
#from .gp_searcher import GPFIFOSearcher, GPMultiFidelitySearcher


def searcher_factory(name, **kwargs):
    # Note: To support GP searchers, more work has to be done:
    # - Different searchers, depending on FIFO or Hyperband scheduler
    # - Quite a lot more options
    if name == 'random':
        return RandomSearcher(**kwargs)
    elif name == 'skopt':
        return SKoptSearcher(**kwargs)
    else:
        raise AssertionError("name = '{}' not supported".format(name))

from .searcher import RandomSearcher
from .skopt_searcher import SKoptSearcher
from .grid_searcher import GridSearcher


def searcher_factory(name, **kwargs):
    if name == 'random':
        return RandomSearcher(**kwargs)
    elif name == 'skopt':
        return SKoptSearcher(**kwargs)
    elif name == 'grid':
        return GridSearcher(**kwargs)
    else:
        raise AssertionError("name = '{}' not supported".format(name))

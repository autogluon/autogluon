import ConfigSpace.hyperparameters as CSH

__all__ = ['Space', 'ListSpace', 'LinearSpace', 'LogLinearSpace', 'get_config_space']

class Space(object):
    pass

class ListSpace(Space):
    def __init__(self, *args):
        self.data = [*args]

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __call__(self, *args, **kwargs):
        self._instance(*args, **kwargs)

class LinearSpace(Space):
    def __init__(self, low, high):
        self.low = low
        self.high = high

class LogLinearSpace(Space):
    def __init__(self, low, high):
        self.low = low
        self.high = high

def get_config_space(name, space):
    assert isinstance(space, Space)
    if isinstance(space, ListSpace):
        return CSH.CategoricalHyperparameter(name=name, choices=space.data)
    elif isinstance(space, LinearSpace):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.low, upper=space.high)
    elif isinstance(space, LogLinearSpace):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.low, upper=space.high, log=True)
    else:
        raise NotImplemented

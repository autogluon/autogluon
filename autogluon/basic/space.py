import ConfigSpace.hyperparameters as CSH

__all__ = ['Space', 'ListSpace', 'LinearSpace', 'LogLinearSpace', 'IntSpace',
           'BoolSpace', 'get_config_space']

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
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class LogLinearSpace(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class IntSpace(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class BoolSpace(IntSpace):
    def __init__(self):
        super(BoolSpace, self).__init__(0, 1)

class ConstantSpace(Space):
    def __init__(self, val):
        self.value = val

def get_config_space(name, space):
    assert isinstance(space, Space)
    if isinstance(space, ListSpace):
        return CSH.CategoricalHyperparameter(name=name, choices=space.data)
    elif isinstance(space, LinearSpace):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, LogLinearSpace):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper, log=True)
    elif isinstance(space, IntSpace):
        return CSH.UniformIntegerHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, ConstantSpace):
        return CSH.Constant(name=name, value=space.value)
    else:
        raise NotImplemented

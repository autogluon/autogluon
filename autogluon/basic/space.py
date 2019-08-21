import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

__all__ = ['Space', 'ListSpace', 'LinearSpace', 'LogLinearSpace', 'IntSpace',
           'BoolSpace', 'strip_cofing_space', 'AutoGluonObject']

class AutoGluonObject:
    pass

class Space(object):
    def get_config_space(self, name):
        return _get_hp(name, self)

class ListSpace(Space):
    def __init__(self, *args):
        self.data = [*args]
        if len(self.data) > 0 and isinstance(self.data[0], AutoGluonObject):
            for x in self.data:
                assert(isinstance(x, AutoGluonObject))

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def get_config_space(self, name):
        cs = CS.ConfigurationSpace()
        if len(self.data) == 0: 
            return CS.ConfigurationSpace()
        if not isinstance(self.data[0], AutoGluonObject):
            hp = CSH.CategoricalHyperparameter(name=name, choices=self.data)
            cs.add_hyperparameter(hp)
        else:
            choices = []
            for i, x in enumerate(self.data):
                choices.append(i)
                cs.add_configuration_space(str(i), x.cs, '.')
            hp = CSH.CategoricalHyperparameter(name=name, choices=choices)
            cs.add_hyperparameter(hp)
        return cs

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

def _get_hp(name, space):
    assert isinstance(space, Space)
    if isinstance(space, LinearSpace):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, LogLinearSpace):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper, log=True)
    elif isinstance(space, IntSpace):
        return CSH.UniformIntegerHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, ConstantSpace):
        return CSH.Constant(name=name, value=space.value)
    else:
        raise NotImplemented

def strip_cofing_space(config, prefix):
    # filter out the config with the corresponding prefix
    new_config = {}
    for k, v in config.items():
        if k.startswith(prefix):
            new_config[k[len(prefix)+1:]] = v
            #config.pop(k)
    return new_config

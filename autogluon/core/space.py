import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

__all__ = ['Space', 'List', 'Linear', 'LogLinear', 'Int',
           'Bool', 'strip_cofing_space', 'AutoGluonObject', 'Sequence']

class AutoGluonObject:
    pass

class Space(object):
    def get_config_space(self, name):
        return _get_hp(name, self)

<<<<<<< HEAD
    def __repr__(self):
        reprstr = self.__class__.__name__
        if hasattr(self, 'lower') and hasattr(self, 'upper'):
            reprstr += ': lower={}, upper={}'.format(self.lower, self.upper)
        if hasattr(self, 'value'):
            
            reprstr += ': value={}'.format(self.value)
        return reprstr

=======
>>>>>>> c8b325866201574caeb688c623d02b23799a65fc
class Sequence(object):
    """A Sequence of AutoGluon Objects
    """
    def __init__(self, *args):
        self.data = [*args]

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def __len__(self):
        return len(self.data)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        if len(self.data) == 0: 
            return CS.ConfigurationSpace()
        for i, x in enumerate(self.data):
            if isinstance(x, AutoGluonObject):
                cs.add_configuration_space(str(i), x.cs, '.')
        return cs

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr

class List(Space):
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

    def __len__(self):
        return len(self.data)

    def get_config_space(self, name):
        cs = CS.ConfigurationSpace()
        if len(self.data) == 0: 
            return CS.ConfigurationSpace()
        hp = CSH.CategoricalHyperparameter(name=name, choices=range(len(self.data)))
        cs.add_hyperparameter(hp)
        for i, x in enumerate(self.data):
            if isinstance(x, AutoGluonObject):
                cs.add_configuration_space(str(i), x.cs, '.')
        return cs

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr

class Linear(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class LogLinear(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class Int(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class Bool(Int):
    def __init__(self):
        super(Bool, self).__init__(0, 1)

class Constant(Space):
    def __init__(self, val):
        self.value = val

def _get_hp(name, space):
    assert isinstance(space, Space)
    if isinstance(space, Linear):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, LogLinear):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper, log=True)
    elif isinstance(space, Int):
        return CSH.UniformIntegerHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, Constant):
        return CSH.Constant(name=name, value=space.value)
    else:
        raise NotImplemented

def strip_cofing_space(config, prefix):
    # filter out the config with the corresponding prefix
    new_config = {}
    for k, v in config.items():
        if k.startswith(prefix):
            new_config[k[len(prefix)+1:]] = v
    return new_config

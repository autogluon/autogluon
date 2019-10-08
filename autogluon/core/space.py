import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ..utils import DeprecationHelper

__all__ = ['Space', 'List', 'Choice', 'Linear', 'LogLinear', 'Int',
           'Bool', 'strip_config_space', 'AutoGluonObject', 'Sequence']

class AutoGluonObject:
    pass

class Space(object):
    def get_config_space(self, name):
        return _get_hp(name, self)

    def __repr__(self):
        reprstr = self.__class__.__name__
        if hasattr(self, 'lower') and hasattr(self, 'upper'):
            reprstr += ': lower={}, upper={}'.format(self.lower, self.upper)
        if hasattr(self, 'value'):
            
            reprstr += ': value={}'.format(self.value)
        return reprstr

class Sequence(Space):
    """A Sequence of AutoGluon Objects

    Args:
        args: a list of search spaces.

    Example:
        >>> sequence = ag.Sequence(
        >>>     ag.Choice('conv3x3', 'conv5x5', 'conv7x7'),
        >>>     ag.Choice('BatchNorm', 'InstanceNorm'),
        >>>     ag.Choice('relu', 'sigmoid'),
        >>> )
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

class Choice(Space):
    """List Search Space

    Args:
        data: the choice candidates

    Example:
        >>> net = ag.Choice('resnet50', 'resnet101')
    """
    def __init__(self, *data):
        self.data = [*data]

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

List = DeprecationHelper(Choice, 'List')

class Linear(Space):
    """linear search space.

    Args:
        lower: the lower bound of the search space
        upper: the upper bound of the search space

    Example:
        >>> learning_rate = ag.Linear(0.01, 0.1)
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class LogLinear(Space):
    """log linear search space.

    Args:
        lower: the lower bound of the search space
        upper: the upper bound of the search space

    Example:
        >>> learning_rate = ag.LogLinear(0.01, 0.1)
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class Int(Space):
    """integer search space.

    Args:
        lower: the lower bound of the search space
        upper: the upper bound of the search space

    Example:
        >>> learning_rate = ag.Int(0, 100)
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class Bool(Int):
    """Bool Search Space

    Example:
        >>> pretrained = ag.Bool()
    """
    def __init__(self):
        super(Bool, self).__init__(0, 1)

def _get_hp(name, space):
    assert isinstance(space, Space)
    if isinstance(space, Linear):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper)
    elif isinstance(space, LogLinear):
        return CSH.UniformFloatHyperparameter(name=name, lower=space.lower, upper=space.upper, log=True)
    elif isinstance(space, Int):
        return CSH.UniformIntegerHyperparameter(name=name, lower=space.lower, upper=space.upper)
    else:
        raise NotImplemented

def strip_config_space(config, prefix):
    # filter out the config with the corresponding prefix
    new_config = {}
    for k, v in config.items():
        if k.startswith(prefix):
            new_config[k[len(prefix)+1:]] = v
    return new_config

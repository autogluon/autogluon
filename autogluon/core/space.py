import copy
import collections
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ..utils import DeprecationHelper, EasyDict

__all__ = ['Space', 'List', 'Dict', 'Categorical', 'Choice', 'Linear', 'LogLinear', 'Int',
           'Bool', 'strip_config_space', 'AutoGluonObject', 'Sequence']

class AutoGluonObject(object):
    def __call__(self, *args, **kwargs):
        if not self._inited:
            self._inited = True
            self._instance = self.init()
        return self._instance.__call__(*args, **kwargs)

    def init(self):
        config = self.cs.get_default_configuration().get_dictionary()
        return self.sample(**config)

    @property
    def cs(self):
        cs = CS.ConfigurationSpace()
        for k, v in self.kwvars.items():
            if isinstance(v, Categorical):
                sub_cs = v.get_config_space()
                _add_cs(cs, sub_cs, k)
            elif isinstance(v, Dict):
                sub_cs = v.get_config_space()
                _add_cs(cs, sub_cs, k)
            elif isinstance(v, Space):
                hp = v.get_config_space(name=k)
                _add_hp(cs, hp)
            else:
                _rm_hp(cs, k)
        return cs

    def sample(self):
        raise NotImplemented

    def __repr__(self):
        return 'AutoGluonObject'

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
        >>>     ag.Categorical('conv3x3', 'conv5x5', 'conv7x7'),
        >>>     ag.Categorical('BatchNorm', 'InstanceNorm'),
        >>>     ag.Categorical('relu', 'sigmoid'),
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

    def sample(self, **config):
        ret = []
        for idx, obj in enumerate(self.data):
            assert isinstance(obj, AutoGluonObject)
            min_config = strip_config_space(config, prefix=str(idx))
            ret.append(obj.sample(**min_config))
        return ret

    @property
    def cs(self):
        return self.get_config_space()

    @property
    def kwspaces(self):
        kw_spaces = collections.OrderedDict()
        for idx, obj in enumerate(self.data):
            if isinstance(obj, AutoGluonObject):
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}.{}'.format(idx, sub_k)
                    kw_spaces[new_k] = sub_v
        return kw_spaces

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr


class Dict(EasyDict):
    """A Dict of Search Spaces
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for k, v in self.items():
            if isinstance(v, AutoGluonObject):
                cs.add_configuration_space(k, v.cs, '.')
        return cs

    @property
    def cs(self):
        return self.get_config_space()

    @property
    def kwspaces(self):
        kw_spaces = collections.OrderedDict()
        for k, obj in self.items():
            if isinstance(v, AutoGluonObject):
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}.{}'.format(k, sub_k)
                    kw_spaces[new_k] = sub_v
        return kw_spaces

    def sample(self, **config):
        raise NotImplemented


class Categorical(Space):
    """Categorical Search Space

    Args:
        data: the choice candidates

    Example:
        >>> net = ag.Categorical('resnet50', 'resnet101')
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

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        if len(self.data) == 0: 
            return CS.ConfigurationSpace()
        hp = CSH.CategoricalHyperparameter(name='choice', choices=range(len(self.data)))
        cs.add_hyperparameter(hp)
        for i, v in enumerate(self.data):
            if isinstance(v, AutoGluonObject):
                cs.add_configuration_space(str(i), v.cs, '.')
        return cs

    @property
    def cs(self):
        return self.get_config_space()

    def sample(self, **config):
        choice = config.pop('choice')
        if isinstance(self.data[choice], AutoGluonObject):
            # nested space: Categorical of AutoGluonobjects
            min_config = strip_config_space(config, prefix=str(choice))
            return self.data[choice].sample(**min_config)
        else:
            return self.data[choice]

    @property
    def kwspaces(self):
        kw_spaces = collections.OrderedDict()
        for idx, obj in enumerate(self.data):
            if isinstance(obj, AutoGluonObject):
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}.{}'.format(idx, sub_k)
                    kw_spaces[new_k] = sub_v
        return kw_spaces

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr

List = DeprecationHelper(Categorical, 'List')
Choice = DeprecationHelper(Categorical, 'Choice')

class Real(Space):
    """linear search space.

    Args:
        lower: the lower bound of the search space
        upper: the upper bound of the search space

    Example:
        >>> learning_rate = ag.Real(0.01, 0.1)
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

Linear = DeprecationHelper(Real, 'Linear')

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
    if isinstance(space, Real):
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

def _add_hp(cs, hp):
    if hp.name in cs._hyperparameters:
        cs._hyperparameters[hp.name] = hp
    else:
        cs.add_hyperparameter(hp)

def _add_cs(master_cs, sub_cs, prefix, delimiter='.', parent_hp=None):
    new_parameters = []
    for hp in sub_cs.get_hyperparameters():
        new_parameter = copy.deepcopy(hp)
        # Allow for an empty top-level parameter
        if new_parameter.name == '':
            new_parameter.name = prefix
        elif not prefix == '':
            new_parameter.name = "%s%s%s" % (prefix, '.', new_parameter.name)
        new_parameters.append(new_parameter)
    for hp in new_parameters:
        _add_hp(master_cs, hp)

def _rm_hp(cs, k):
    if k in cs._hyperparameters:
        cs._hyperparameters.pop(k)
    for hp in cs.get_hyperparameters():
        if  hp.name.startswith("%s."%(k)):
            cs._hyperparameters.pop(hp.name)

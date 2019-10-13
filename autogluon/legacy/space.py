from ..core.space import *
from ..utils import DeprecationHelper
import ConfigSpace.hyperparameters as CSH

__all__ = ['List', 'Linear', 'Log', 'ListSpace',
           'IntSpace', 'LogLinearSpace']

class List(Categorical):
    def __init__(self, objs):
        super().__init__(*objs)

    def __repr__(self):
        reprstr = 'Categorical' + str(self.data)
        return reprstr

class Linear(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def get_config_space(self, name):
        if isinstance(lower, int) and isinstance(upper, int):
            self.hyper_param = CSH.UniformIntegerHyperparameter(
                name=name, lower=self.lower, upper=self.upper, log=False)
        else:
            self.hyper_param = CSH.UniformFloatHyperparameter(
                name=name, lower=self.lower, upper=self.upper, log=False)
        return self.hyper_param

class Log(Space):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def get_config_space(self, name):
        if isinstance(lower, int) and isinstance(upper, int):
            self.hyper_param = CSH.UniformIntegerHyperparameter(
                name=name, lower=self.lower, upper=self.upper, log=True)
        else:
            self.hyper_param = CSH.UniformFloatHyperparameter(
                name=name, lower=self.lower, upper=self.upper, log=True)
        return self.hyper_param

ListSpace = DeprecationHelper(Categorical, 'List')
IntSpace = DeprecationHelper(Int, 'Int')
LogLinearSpace = DeprecationHelper(LogLinear, 'LogLinear')

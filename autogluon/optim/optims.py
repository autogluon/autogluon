import ConfigSpace as CS

from ..space import *
from .utils import autogluon_optims

__all__ = ['Optimizers']


class Optimizers(object):
    def __init__(self, optim_list):
        assert isinstance(optim_list, list), type(optim_list)
        self.optim_list = optim_list
        self.search_space = None
        self.add_search_space()

    def _set_search_space(self, cs):
        self.search_space = cs

    def add_search_space(self):
        cs = CS.ConfigurationSpace()
        optim_list_hyper_param = List('autooptims', choices=self.optim_list).get_hyper_param()
        cs.add_hyperparameter(optim_list_hyper_param)
        for optim in self.optim_list:
            # TODO: add more optims
            optim_hyper_params = optim.get_hyper_params()
            cs.add_hyperparameters(optim_hyper_params)
            conds = []
            for optim_hyper_param in optim_hyper_params:
                cond = CS.InCondition(optim_hyper_param, optim_list_hyper_param,
                                      ['sgd', 'adam'])
                conds.append(cond)
            cs.add_conditions(conds)
        self._set_search_space(cs)

    def get_search_space(self):
        return self.search_space


optims = ['sgd',
          'nag',
          'rmsprop',
          'adam',
          'adagrad',
          'adadelta',
          'adamax',
          'nadam',
          'dcasgd',
          'sgld',
          'signum',
          'ftml',
          'lbsgd',
          'ftrl']


@autogluon_optims
def get_optim(name):
    name = name.lower()
    if name not in optims:
        err_str = '"%s" is not among the following optim list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(optims)))
        raise ValueError(err_str)
    optim = name
    return optim

#TODO: add more method based optim auto decorator

@autogluon_optims
def SGD(**kwargs):
    pass


@autogluon_optims
def NAG(**kwargs):
    pass


@autogluon_optims
def RMSPROP(**kwargs):
    pass


@autogluon_optims
def ADAM(**kwargs):
    pass


@autogluon_optims
def ADAGRAD(**kwargs):
    pass


@autogluon_optims
def ADADELTA(**kwargs):
    pass


@autogluon_optims
def ADAMAX(**kwargs):
    pass


@autogluon_optims
def NADAM(**kwargs):
    pass


@autogluon_optims
def DCASGD(**kwargs):
    pass


@autogluon_optims
def SGLD(**kwargs):
    pass


@autogluon_optims
def SIGNUM(**kwargs):
    pass


@autogluon_optims
def FTML(**kwargs):
    pass


@autogluon_optims
def LBSGD(**kwargs):
    pass


@autogluon_optims
def FTRL(**kwargs):
    pass

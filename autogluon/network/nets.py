import ConfigSpace as CS

from ..space import *
from .utils import Net

__all__ = ['Nets']


class Nets(object):
    def __init__(self, net_list):
        #TODO(cgraywang): add model instance, for now, use a list of model names
        assert isinstance(net_list, list), type(net_list)
        self.net_list = net_list
        self.search_space = None
        self.add_search_space()

    def _set_search_space(self, cs):
        self.search_space = cs

    def add_search_space(self):
        cs = CS.ConfigurationSpace()
        net_list_hyper_param = List('model',
                                    choices=self.get_net_strs())\
            .get_hyper_param()
        cs.add_hyperparameter(net_list_hyper_param)
        for net in self.net_list:
            #TODO(cgraywang): distinguish between different nets, only support resnet for now
            net_hyper_params = net.get_hyper_params()
            conds = []
            for net_hyper_param in net_hyper_params:
                if net_hyper_param not in cs.get_hyperparameters():
                    cs.add_hyperparameter(net_hyper_param)
                #TODO(cgraywang): put condition in presets? split task settings out
                cond = CS.InCondition(net_hyper_param, net_list_hyper_param,
                                      self.get_net_strs())
                conds.append(cond)
            cs.add_conditions(conds)
        self._set_search_space(cs)

    def get_search_space(self):
        return self.search_space

    def get_net_strs(self):
        net_strs = []
        for net in self.net_list:
            if isinstance(net, Net):
                net_strs.append(net.name)
        return net_strs

    def __repr__(self):
        return "AutoGluon Nets %s with %s" % (str(self.get_net_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Nets %s with %s" % (str(self.get_net_strs()), str(self.search_space))

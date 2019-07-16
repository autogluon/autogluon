import ConfigSpace as CS

import autogluon as ag
from ..core import *
from ..space import *
from .utils import Net

__all__ = ['Nets']


class Nets(BaseAutoObject):
    def __init__(self, net_list):
        #TODO (cgraywang): add net user config
        #TODO(cgraywang): add model instance, for now, use a list of model names
        assert isinstance(net_list, list), type(net_list)
        super(Nets, self).__init__()
        self.net_list = net_list
        self._add_search_space()

    def _add_search_space(self):
        cs = CS.ConfigurationSpace()
        net_list_hyper_param = List('model',
                                    choices=self._get_search_space_strs()).get_hyper_param()
        cs.add_hyperparameter(net_list_hyper_param)
        for net in self.net_list:
            #TODO(cgraywang): distinguish between different nets, only support resnet for now
            if isinstance(net, str):
                net = self._get_task_model(net)
            net_hyper_params = net.get_hyper_params()
            conds = []
            for net_hyper_param in net_hyper_params:
                if net_hyper_param not in cs.get_hyperparameters():
                    cs.add_hyperparameter(net_hyper_param)
                #TODO(cgraywang): put condition in presets? split task settings out
                cond = CS.InCondition(net_hyper_param, net_list_hyper_param,
                                      self._get_search_space_strs())
                conds.append(cond)
            cs.add_conditions(conds)
        self.search_space = cs

    def _get_search_space_strs(self):
        net_strs = []
        for net in self.net_list:
            if isinstance(net, Net):
                net_strs.append(net.name)
            elif isinstance(net, str):
                net_strs.append(net)
            else:
                raise NotImplementedError
        return net_strs

    def _get_task_model(self, net):
        #TODO: temporal solution, we now have limit application support
        net = net.lower()
        if net in ag.task.image_classification.models:
            net = ag.task.image_classification.get_model(net)
        elif net in ag.task.object_detection.models:
            net = ag.task.object_detection.get_model(net)
        else:
            raise NotImplementedError
        return net

    def __repr__(self):
        return "AutoGluon Nets %s with %s" % \
               (str(self._get_search_space_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Nets %s with %s" % \
               (str(self._get_search_space_strs()), str(self.search_space))

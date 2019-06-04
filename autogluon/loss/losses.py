import ConfigSpace as CS

import autogluon as ag
from ..space import *
from .utils import Loss

__all__ = ['Losses']


class Losses(object):
    def __init__(self, loss_list):
        # TODO(cgraywang): add model instance, for now, use a list of model names
        assert isinstance(loss_list, list), type(loss_list)
        self.loss_list = loss_list
        self.search_space = None
        self.add_search_space()

    def _set_search_space(self, cs):
        self.search_space = cs

    def add_search_space(self):
        cs = CS.ConfigurationSpace()
        loss_list_hyper_param = List('loss',
                                     choices=self.get_loss_strs()) \
            .get_hyper_param()
        cs.add_hyperparameter(loss_list_hyper_param)
        # TODO (cgraywang): do not add hyper-params for loss
        self._set_search_space(cs)

    def get_search_space(self):
        return self.search_space

    def get_loss_strs(self):
        loss_strs = []
        for loss in self.loss_list:
            if isinstance(loss, Loss):
                loss_strs.append(loss.name)
            elif isinstance(loss, str):
                loss_strs.append(loss)
            else:
                pass
        return loss_strs

    def __repr__(self):
        return "AutoGluon Losses %s with %s" % (str(self.get_loss_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Losses %s with %s" % (str(self.get_loss_strs()), str(self.search_space))

import ConfigSpace as CS

import autogluon as ag
from ..space import *
from .utils import Metric

__all__ = ['Metrics']


class Metrics(object):
    def __init__(self, metric_list):
        # TODO(cgraywang): add model instance, for now, use a list of model names
        assert isinstance(metric_list, list), type(metric_list)
        self.metric_list = metric_list
        self.search_space = None
        self.add_search_space()

    def _set_search_space(self, cs):
        self.search_space = cs

    def add_search_space(self):
        cs = CS.ConfigurationSpace()
        metric_list_hyper_param = List('metric',
                                       choices=self.get_metric_strs()) \
            .get_hyper_param()
        cs.add_hyperparameter(metric_list_hyper_param)
        # TODO (cgraywang): do not add hyper-params for metric
        self._set_search_space(cs)

    def get_search_space(self):
        return self.search_space

    def get_metric_strs(self):
        metric_strs = []
        for metric in self.metric_list:
            if isinstance(metric, Metric):
                metric_strs.append(metric.name)
            elif isinstance(metric, str):
                metric_strs.append(metric)
            else:
                pass
        return metric_strs

    def __repr__(self):
        return "AutoGluon Metrics %s with %s" % (
        str(self.get_metric_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Metrics %s with %s" % (
        str(self.get_metric_strs()), str(self.search_space))

import ConfigSpace as CS

from ..core import *
from ..space import *
from .utils import Metric

__all__ = ['Metrics']


class Metrics(BaseAutoObject):
    def __init__(self, metric_list):
        # TODO(cgraywang): add instance, for now, use a list
        # TODO(cgraywang): use all option
        assert isinstance(metric_list, list), type(metric_list)
        super(Metrics, self).__init__()
        self.metric_list = metric_list
        self._add_search_space()

    def _add_search_space(self):
        cs = CS.ConfigurationSpace()
        # TODO (cgraywang): add more hparams for metric, e.g., weight
        metric_list_hyper_param = List('metric',
                                       choices=self._get_search_space_strs()).get_hyper_param()
        cs.add_hyperparameter(metric_list_hyper_param)
        self.search_space = cs

    def _get_search_space_strs(self):
        metric_strs = []
        for metric in self.metric_list:
            if isinstance(metric, Metric):
                metric_strs.append(metric.name)
            elif isinstance(metric, str):
                metric_strs.append(metric)
            else:
                raise NotImplementedError
        return metric_strs

    def __repr__(self):
        return "AutoGluon Metrics %s with %s" % (
            str(self._get_search_space_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Metrics %s with %s" % (
            str(self._get_search_space_strs()), str(self.search_space))

import ConfigSpace as CS

from ..core import *
from ..space import *
from .utils import Loss

__all__ = ['Losses']


class Losses(BaseAutoObject):
    """The auto loss.

    Args:
        loss_list: a list of losses.

    Example:
        >>> losses = Losses(['SoftmaxCrossEntropyLoss'])
    """
    def __init__(self, loss_list):
        # TODO(cgraywang): add instance, for now, use a list
        assert isinstance(loss_list, list), type(loss_list)
        super(Losses, self).__init__()
        self.loss_list = loss_list
        self._add_search_space()

    def _add_search_space(self):
        cs = CS.ConfigurationSpace()
        # TODO (cgraywang): add more hparams for loss, e.g., weight
        loss_list_hyper_param = List('loss',
                                     choices=self._get_search_space_strs()).get_hyper_param()
        cs.add_hyperparameter(loss_list_hyper_param)
        self.search_space = cs

    def _get_search_space_strs(self):
        loss_strs = []
        for loss in self.loss_list:
            if isinstance(loss, Loss):
                loss_strs.append(loss.name)
            elif isinstance(loss, str):
                loss_strs.append(loss)
            else:
                raise NotImplementedError
        return loss_strs

    def __repr__(self):
        return "AutoGluon Losses %s with %s" % (
            str(self._get_search_space_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Losses %s with %s" % (
            str(self._get_search_space_strs()), str(self.search_space))

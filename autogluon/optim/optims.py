import ConfigSpace as CS

from .utils import autogluon_optims, Optimizer
from ..core import *
from ..space import *

__all__ = ['Optimizers', 'get_optim', 'SGD', 'NAG', 'RMSProp', 'Adam', 'AdaGrad', 'AdaDelta',
           'Adamax', 'Nadam', 'DCASGD', 'SGLD', 'Signum', 'FTML', 'LBSGD', 'Ftrl', 'BertAdam']


class Optimizers(BaseAutoObject):
    # TODO (cgraywang): add optimizer hparams config
    def __init__(self, optim_list):
        assert isinstance(optim_list, list), type(optim_list)
        super(Optimizers, self).__init__()
        self.optim_list = optim_list
        self._add_search_space()

    def _add_search_space(self):
        cs = CS.ConfigurationSpace()
        optim_list_hyper_param = List('optimizer', choices=self._get_search_space_strs()) \
            .get_hyper_param()
        cs.add_hyperparameter(optim_list_hyper_param)
        for optim in self.optim_list:
            # TODO: add more optims
            if isinstance(optim, str):
                optim = get_optim(optim)
            optim_hyper_params = optim.get_hyper_params()
            conds = []
            for optim_hyper_param in optim_hyper_params:
                if optim_hyper_param not in cs.get_hyperparameters():
                    cs.add_hyperparameter(optim_hyper_param)
                cond = CS.InCondition(optim_hyper_param, optim_list_hyper_param,
                                      self._get_search_space_strs())
                conds.append(cond)
            cs.add_conditions(conds)
        self.search_space = cs

    def _get_search_space_strs(self):
        optim_strs = []
        for optim in self.optim_list:
            if isinstance(optim, Optimizer):
                optim_strs.append(optim.name)
            elif isinstance(optim, str):
                optim_strs.append(optim)
            else:
                raise NotImplementedError
        return optim_strs

    def __repr__(self):
        return "AutoGluon Optimizers %s with %s" % (
            str(self._get_search_space_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Optimizers %s with %s" % (
            str(self._get_search_space_strs()), str(self.search_space))


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
          'ftrl',
          'bertadam']


@autogluon_optims
def get_optim(name):
    """Returns a optimizer with search space by name

    Parameters
    ----------
    name : str
        Name of the model.
    rescale_grad : float, optional
        Multiply the gradient with `rescale_grad` before updating. Often
        choose to be ``1.0/batch_size``.

    param_idx2name : dict from int to string, optional
        A dictionary that maps int index to string name.

    clip_gradient : float, optional
        Clip the gradient by projecting onto the box ``[-clip_gradient, clip_gradient]``.

    learning_rate : float, optional
        The initial learning rate.

    lr_scheduler : LRScheduler, optional
        The learning rate scheduler.

    wd : float, optional
        The weight decay (or L2 regularization) coefficient. Modifies objective
        by adding a penalty for having large weights.

    sym: Symbol, optional
        The Symbol this optimizer is applying to.

    begin_num_update : int, optional
        The initial number of updates.

    multi_precision : bool, optional
       Flag to control the internal precision of the optimizer.::

           False: results in using the same precision as the weights (default),
           True: makes internal 32-bit copy of the weights and applies gradients
           in 32-bit precision even if actual weights used in the model have lower precision.
           Turning this on can improve convergence and accuracy when training with float16.

    Returns
    -------
    optim
        The optimizer with search space.
    """
    name = name.lower()
    if name not in optims:
        err_str = '"%s" is not among the following optim list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(optims)))
        raise ValueError(err_str)
    optim = Optimizer(name)
    return optim


@autogluon_optims
def SGD(**kwargs):
    return Optimizer('sgd')


@autogluon_optims
def NAG(**kwargs):
    return Optimizer('nag')


@autogluon_optims
def RMSProp(**kwargs):
    return Optimizer('rmsprop')


@autogluon_optims
def Adam(**kwargs):
    return Optimizer('adam')


@autogluon_optims
def AdaGrad(**kwargs):
    return Optimizer('adagrad')


@autogluon_optims
def AdaDelta(**kwargs):
    return Optimizer('adadelta')


@autogluon_optims
def Adamax(**kwargs):
    return Optimizer('adamax')


@autogluon_optims
def Nadam(**kwargs):
    return Optimizer('nadam')


@autogluon_optims
def DCASGD(**kwargs):
    return Optimizer('dcasgd')


@autogluon_optims
def SGLD(**kwargs):
    return Optimizer('sgld')


@autogluon_optims
def Signum(**kwargs):
    return Optimizer('signum')


@autogluon_optims
def FTML(**kwargs):
    return Optimizer('ftml')


@autogluon_optims
def LBSGD(**kwargs):
    return Optimizer('lbsgd')


@autogluon_optims
def Ftrl(**kwargs):
    return Optimizer('ftrl')


@autogluon_optims
def BertAdam(**kwargs):
    return Optimizer('bertadam')

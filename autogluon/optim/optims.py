import ConfigSpace as CS

from ..space import *
from .utils import autogluon_optims, Optimizer

__all__ = ['Optimizers', 'get_optim']


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
        optim_list_hyper_param = List('optimizer', choices=self.get_optim_strs()).\
            get_hyper_param()
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
                                      self.get_optim_strs())
                conds.append(cond)
            cs.add_conditions(conds)
        self._set_search_space(cs)

    def get_search_space(self):
        return self.search_space

    def get_optim_strs(self):
        optim_strs = []
        for optim in self.optim_list:
            if isinstance(optim, Optimizer):
                optim_strs.append(optim.name)
            elif isinstance(optim, str):
                optim_strs.append(optim)
            else:
                pass
        return optim_strs

    def __repr__(self):
        return "AutoGluon Optimizers %s with %s" % (str(self.get_optim_strs()), str(self.search_space))

    def __str__(self):
        return "AutoGluon Optimizers %s with %s" % (str(self.get_optim_strs()), str(self.search_space))
    

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


@autogluon_optims
def BERTADAM(**kwargs):
    pass

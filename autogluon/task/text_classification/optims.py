from autogluon.optim import optims
from autogluon.optim.utils import autogluon_optims, Optimizer

__all__ = ['BertAdam', 'get_optim']

optims = ['bertadam'] + optims.optims  # TODO segregate the task specific optimizers


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
def BertAdam(**kwargs):
    pass

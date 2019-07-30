from mxnet import gluon
import gluoncv

from ...loss import autogluon_losses, Loss

__all__ = ['get_loss', 'get_loss_instance']

#TODO(cgraywang): abstract general loss shared across tasks
losses = {'SSDMultiBoxLoss': gluoncv.loss.SSDMultiBoxLoss}


@autogluon_losses
def get_loss(name, **kwargs):
    """Returns a loss with search space by name

    Parameters
    ----------
    name : str
        Name of the loss.

    Returns
    -------
    Loss
        The loss with search space.
    """
    if name not in losses and name.lower() not in losses:
        err_str = '"%s" is not among the following loss list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(losses.keys())))
        raise ValueError(err_str)
    loss = Loss(name)
    return loss


def get_loss_instance(name, **kwargs):
    """Returns a loss instance by name

    Parameters
    ----------
    name : str
        Name of the loss.

    Returns
    -------
    Loss
        The loss instance.
    """
    if name not in losses and name.lower() not in losses:
        err_str = '"%s" is not among the following loss list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(losses.keys())))
        raise ValueError(err_str)
    loss = losses[name](*kwargs)
    return loss


# TODO (cgraywang): add more models using method
@autogluon_losses
def SSDMultiBoxLoss(**kwargs):
    pass

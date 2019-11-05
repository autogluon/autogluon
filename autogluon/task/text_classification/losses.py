from mxnet import gluon
__all__ = ['get_loss_instance']

losses = {'SoftmaxCrossEntropyLoss': gluon.loss.SoftmaxCrossEntropyLoss}


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
    loss = losses[name]()
    return loss

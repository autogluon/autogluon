from mxnet import gluon
import gluoncv

from autogluon.loss import autogluon_losses, Loss

__all__ = ['get_loss', 'get_loss_instance']

losses = {'SoftmaxCrossEntropyLoss': gluon.loss.SoftmaxCrossEntropyLoss,
          'L2Loss': gluon.loss.L2Loss,
          'L1Loss': gluon.loss.L1Loss,
          'SigmoidBinaryCrossEntropyLoss': gluon.loss.SigmoidBinaryCrossEntropyLoss,
          'KLDivLoss': gluon.loss.KLDivLoss,
          'HuberLoss': gluon.loss.HuberLoss,
          'HingeLoss': gluon.loss.HingeLoss,
          'SquaredHingeLoss': gluon.loss.SquaredHingeLoss,
          'LogisticLoss': gluon.loss.LogisticLoss,
          'TripletLoss': gluon.loss.TripletLoss,
          'CTCLoss': gluon.loss.CTCLoss,
          'CosineEmbeddingLoss': gluon.loss.CosineEmbeddingLoss,
          'PoissonNLLLoss': gluon.loss.SoftmaxCrossEntropyLoss,
          'DistillationSoftmaxCrossEntropyLoss': gluoncv.loss.SoftmaxCrossEntropyLoss,
          'MixSoftmaxCrossEntropyLoss': gluoncv.loss.SoftmaxCrossEntropyLoss}


@autogluon_losses
def get_loss(name, **kwargs):
    """Returns a loss with search space by name

    Parameters
    ----------
    name : str
        Name of the model.

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
    """Returns a loss with search space by name

    Parameters
    ----------
    name : str
        Name of the model.

    Returns
    -------
    Loss
        The loss with search space.
    """
    if name not in losses and name.lower() not in losses:
        err_str = '"%s" is not among the following loss list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(losses.keys())))
        raise ValueError(err_str)
    loss = losses[name]()
    return loss


# TODO (cgraywang): add more models using method
@autogluon_losses
def L2Loss(**kwargs):
    pass


@autogluon_losses
def L1Loss(**kwargs):
    pass


@autogluon_losses
def SigmoidBinaryCrossEntropyLoss(**kwargs):
    pass


@autogluon_losses
def SoftmaxCrossEntropyLoss(**kwargs):
    pass


@autogluon_losses
def KLDivLoss(**kwargs):
    pass


@autogluon_losses
def HuberLoss(**kwargs):
    pass


@autogluon_losses
def HingeLoss(**kwargs):
    pass


@autogluon_losses
def SquaredHingeLoss(**kwargs):
    pass


@autogluon_losses
def LogisticLoss(**kwargs):
    pass


@autogluon_losses
def TripletLoss(**kwargs):
    pass


@autogluon_losses
def CTCLoss(**kwargs):
    pass


@autogluon_losses
def CosineEmbeddingLoss(**kwargs):
    pass


@autogluon_losses
def PoissonNLLLoss(**kwargs):
    pass


@autogluon_losses
def DistillationSoftmaxCrossEntropyLoss(**kwargs):
    pass


@autogluon_losses
def MixSoftmaxCrossEntropyLoss(**kwargs):
    pass

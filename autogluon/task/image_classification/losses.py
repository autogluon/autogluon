from mxnet import gluon
import gluoncv

__all__ = ['get_loss_instance']

#TODO: abstract general loss shared across tasks
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

def get_loss_instance(name, **kwargs):
    """Returns a loss instance by name

    Args:
        name : str
            Name of the loss.
    """
    if name not in losses and name.lower() not in losses:
        err_str = '"%s" is not among the following loss list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(losses.keys())))
        raise ValueError(err_str)
    loss = losses[name]()
    return loss


from mxnet.gluon import loss
from . import obj

__all__ = ['SoftmaxCrossEntropyLoss']

@obj()
class SoftmaxCrossEntropyLoss(loss.SoftmaxCrossEntropyLoss):
    pass

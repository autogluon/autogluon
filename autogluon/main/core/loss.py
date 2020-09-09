from mxnet.gluon import loss
from ..core import obj

__all__ = ['SoftmaxCrossEntropyLoss']

@obj()
class SoftmaxCrossEntropyLoss(loss.SoftmaxCrossEntropyLoss):
    pass

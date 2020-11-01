from mxnet.gluon import loss

from autogluon.core import obj

__all__ = ['SoftmaxCrossEntropyLoss']

@obj()
class SoftmaxCrossEntropyLoss(loss.SoftmaxCrossEntropyLoss):
    pass

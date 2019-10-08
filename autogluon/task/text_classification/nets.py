import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn

class Nets(HybridBlock):
    """Model for classification task with pretrained and head models.
    """

    def __init__(self, pretrained_net, head_net, num_classes=2, dropout=0.0,
                 prefix=None, params=None):
        super(Nets, self).__init__(prefix=prefix, params=params)
        self.pretrained_net = pretrained_net
        self.head_net = head_net

    def __call__(self, inputs, token_types, valid_length=None):
        # pylint: disable=dangerous-default-value, arguments-differ
        """
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        valid_length = [] if valid_length is None else valid_length
        return super(Nets, self).__call__(inputs, token_types, valid_length)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=arguments-differ
        """
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None
        if isinstance(valid_length, list) and len(valid_length) == 0:
            valid_length = None
        _, pooler_out = self.pretrained_net(inputs, token_types, valid_length)
        try:
            return self.head_net(pooler_out)
        except ValueError:
            raise ValueError

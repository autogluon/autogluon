import mxnet as mx
from mxnet import gluon
from ...model_zoo.models.utils import *
from ...model_zoo.models.mbconv import MBConvBlock
from ...core.space import *
from .enas import enas_unit, ENAS_Sequential

__all__ = ['ENAS_MbBlock', 'ENAS_MBNet']

@enas_unit()
class ENAS_MbBlock(MBConvBlock):
    pass

class ENAS_MBNet(gluon.HybridBlock):
    def __init__(self, blocks_args=[], dropout_rate=0.2, num_classes=1000, input_size=224,
                 activation='swish', blocks=None, **kwargs):
        r"""ENAS model with MobileNet search space

        Args:
            blocks_args (list of autogluon.Dict)
        """
        super(ENAS_MBNet, self).__init__(**kwargs)
        if len(blocks_args)==0 and blocks is None:
            blocks_args = get_enas_blockargs()
        #assert isinstance(blocks_args, (tuple, list)), 'blocks_args should be a list'
        self.input_size = input_size
        with self.name_scope():
            self._features = gluon.nn.HybridSequential()
            with self._features.name_scope():
                # stem conv
                out_channels = 32
                _add_conv(self._features, out_channels, kernel=3,
                    stride=2, activation=activation, batchnorm=True,
                    input_size=input_size, in_channels=3)

            input_size = _update_input_size(input_size, 2)
            _blocks = []
            for block_arg in blocks_args:
                block_arg.update(in_channels=out_channels, input_size=input_size, activation=activation)
                out_channels=block_arg.channels
                _blocks.append(ENAS_MbBlock(**block_arg))
                input_size = _update_input_size(input_size, block_arg.stride)
                if block_arg.num_repeat > 1:
                    block_arg.update(in_channels=out_channels, stride=1,
                                     input_size=input_size)

                for _ in range(block_arg.num_repeat - 1):
                    _blocks.append(ENAS_MbBlock(**block_arg))

            if blocks is not None:
                self._blocks = ENAS_Sequential(blocks)
            else:
                self._blocks = ENAS_Sequential(_blocks)
            # Head
            self._conv_head = gluon.nn.HybridSequential()
            out_channels = 320
            hidden_channels = 1280
            with self._conv_head.name_scope():
                _add_conv(self._conv_head, hidden_channels, activation=activation,
                          batchnorm=True, input_size=input_size,
                          in_channels=out_channels)
            out_channels = hidden_channels
            # Final linear layer
            self._dropout = dropout_rate
            self._fc = gluon.nn.Dense(num_classes, use_bias=True, in_units=out_channels)

    def sample(self, **config):
        self._blocks.sample(**config)

    @property
    def kwspaces(self):
        return self._blocks.kwspaces

    @property
    def latency(self):
        return self._blocks.latency

    @property
    def avg_latency(self):
        return self._blocks.avg_latency

    @property
    def nparams(self):
        return self._blocks.nparams

    def evaluate_latency(self, x):
        self._blocks.evaluate_latency(x)

    def hybrid_forward(self, F, x):
        x = self._features(x)
        x = self._blocks(x)
        x = self._conv_head(x)
        x = F.contrib.AdaptiveAvgPooling2D(x, 1)
        if self._dropout:
            x = F.Dropout(x, self._dropout)
        x = self._fc(x)
        return x

def get_enas_blockargs():
    """ Creates a predefined efficientnet model, which searched by original paper. """
    blocks_args = [
        Dict(kernel=3, num_repeat=1, channels=16, expand_ratio=1, stride=1, se_ratio=0.25, in_channels=32),
        Dict(kernel=3, num_repeat=1, channels=16, expand_ratio=1, stride=1, se_ratio=0.25, in_channels=16, with_zero=True),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=1, channels=24, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=16),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=3, channels=24, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, in_channels=24, with_zero=True),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=1, channels=40, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=24),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=3, channels=40, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, in_channels=40, with_zero=True),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=1, channels=80, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=40),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=4, channels=80, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, in_channels=80, with_zero=True),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=1, channels=112, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, in_channels=80),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=4, channels=112, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, in_channels=112, with_zero=True),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=1, channels=192, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=112),
        Dict(kernel=Categorical(3, 5, 7), num_repeat=5, channels=192, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, in_channels=192, with_zero=True),
        Dict(kernel=3, num_repeat=1, channels=320, expand_ratio=6, stride=1, se_ratio=0.25, in_channels=192),
    ]
    return blocks_args

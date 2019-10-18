import collections
import mxnet as mx
from mxnet import gluon
from .utils import *
from .mbconv import MBConvBlock
from ..core.space import *

__all__ = ['autogluon_enas_unit', 'ENAS_Unit', 'ENAS_Sequential', 'ENAS_Net']

class ENAS_Sequential(gluon.HybridBlock):
    def __init__(self, *modules_list):
        """
        Args:
            modules_list(list of ENAS_Unit)
        """
        super().__init__()
        if len(modules_list) == 1 and isinstance(modules_list, (list, tuple)):
            modules_list = modules_list[0]
        self._modules = {}
        self._blocks = gluon.nn.HybridSequential()
        self._kwspaces = collections.OrderedDict()
        for i, op in enumerate(modules_list):
            self._modules[str(i)] = op
            with self._blocks.name_scope():
                self._blocks.add(op)
            if isinstance(op, ENAS_Unit):
                self._kwspaces[str(i)] = Categorical(*list(range(len(op))))
     
    @property
    def kwspaces(self):
        return self._kwspaces

    def hybrid_forward(self, F, x):
        for k, op in self._modules.items():
            x = op(x)
        return x

    def sample(self, **configs):
        for k, v in configs.items():
            self._modules[k].sample(v)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        for i, op in self._modules.items():
            reprstr += '\n\t{}: {}'.format(i, op)
        reprstr += ')\n'
        return reprstr

class ENAS_Unit(gluon.HybridBlock):
    def __init__(self, *ops):
        super().__init__()
        self.module_list = gluon.nn.HybridSequential()
        for op in ops:
            self.module_list.add(op)
        self.index = 0

    def hybrid_forward(self, F, x):
        return self.module_list[self.index](x)

    def sample(self, ind):
        self.index = ind

    def __len__(self):
        return len(self.module_list)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(num of choices: {}), current architecture:\n\t {}' \
            .format(len(self.module_list), self.module_list[self.index])
        return reprstr

def autogluon_enas_unit(**kwvars):
    def registered_class(Cls):
        class enas_unit(ENAS_Unit):
            def __init__(self, *args, **kwargs):
                kwvars.update(kwargs)
                self._args = args
                self._kwargs = kwvars
                blocks = []
                for arg in self.get_config_grid(kwvars):
                    blocks.append(Cls(**arg))
                super().__init__(*blocks)

            @staticmethod
            def get_config_grid(dict_space):
                param_grid = {}
                constants = {}
                for k, v in dict_space.items():
                    if isinstance(v, Categorical):
                        param_grid[k] = v.data
                    elif isinstance(v, Space):
                        raise NotImplemented
                    else:
                        constants[k] = v
                from sklearn.model_selection import ParameterGrid
                configs = list(ParameterGrid(param_grid))
                for config in configs:
                    config.update(constants)
                return configs

        return enas_unit

    return registered_class

@autogluon_enas_unit(
    in_channels=32,
    channels=32,
    expand_ratio=Categorical(3, 6),
    kernel=Categorical(3, 5),
    input_size=32,
)
class ENAS_Block(MBConvBlock):
    pass

class ENAS_Net(gluon.HybridBlock):
    def __init__(self, blocks_args=None, dropout_rate=0.2, num_classes=1000, input_size=224,
                 activation='swish', **kwargs):
        r"""ENAS model with MobileNet search space

        Args:
            blocks_args (list of autogluon.Dict)
        """
        super(ENAS_Net, self).__init__(**kwargs)
        if blocks_args is None:
            blocks_args = get_enas_blockargs()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
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
                block_arg.update(in_channels=out_channels, input_size=input_size)
                out_channels=block_arg.channels
                _blocks.append(ENAS_Block(**block_arg))
                input_size = _update_input_size(input_size, block_arg.stride)
                if block_arg.num_repeat > 1:
                    block_arg.update(in_channels=out_channels, stride=1,
                                     input_size=input_size)

                for _ in range(block_arg.num_repeat - 1):
                    _blocks.append(ENAS_Block(**block_arg))

            self._blocks = ENAS_Sequential(_blocks)
            # Head
            self._conv_head = gluon.nn.HybridSequential()
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
        Dict(kernel=Categorical(3, 5), num_repeat=2, channels=24, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=16),
        Dict(kernel=Categorical(3, 5), num_repeat=2, channels=40, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=24),
        Dict(kernel=Categorical(3, 5), num_repeat=3, channels=80, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=40),
        Dict(kernel=Categorical(3, 5), num_repeat=3, channels=112, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, in_channels=80),
        Dict(kernel=Categorical(3, 5), num_repeat=4, channels=192, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, in_channels=112),
        Dict(kernel=3, num_repeat=1, channels=320, expand_ratio=6, stride=1, se_ratio=0.25, in_channels=192),
    ]
    return blocks_args

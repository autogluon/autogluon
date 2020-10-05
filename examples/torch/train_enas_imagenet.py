import torch
from torch import nn

from autogluon.core import Dict, Categorical
from autogluon.extra import *
from autogluon.extra.contrib.enas import *
from autogluon.extra.model_zoo.models.utils import _update_input_size


@enas_unit()
class ENAS_MBConv(MBConvBlock):
    pass

blocks_args = [
    Dict(kernel=3, num_repeat=1, output_filters=16, expand_ratio=1, stride=1, se_ratio=0.25, input_filters=32),
    Dict(kernel=3, num_repeat=1, output_filters=16, expand_ratio=1, stride=1, se_ratio=0.25, input_filters=16, with_zero=True),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=1, output_filters=24, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, input_filters=16),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=3, output_filters=24, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, input_filters=24, with_zero=True),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=1, output_filters=40, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, input_filters=24),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=3, output_filters=40, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, input_filters=40, with_zero=True),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=1, output_filters=80, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, input_filters=40),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=4, output_filters=80, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, input_filters=80, with_zero=True),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=1, output_filters=112, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, input_filters=80),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=4, output_filters=112, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, input_filters=112, with_zero=True),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=1, output_filters=192, expand_ratio=Categorical(3, 6), stride=2, se_ratio=0.25, input_filters=112),
    Dict(kernel=Categorical(3, 5, 7), num_repeat=5, output_filters=192, expand_ratio=Categorical(3, 6), stride=1, se_ratio=0.25, input_filters=192, with_zero=True),
    Dict(kernel=3, num_repeat=1, output_filters=320, expand_ratio=6, stride=1, se_ratio=0.25, input_filters=192),
]

input_size = 224
Conv2D = get_same_padding_conv2d(input_size)
features = nn.Sequential(
        Conv2D(3, 32, kernel_size=3, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
    )

_blocks = []
input_size = 112
out_filters = 32
for block_arg in blocks_args:
    block_arg.update(input_filters=out_filters, input_size=input_size)
    out_filters=block_arg.output_filters
    _blocks.append(ENAS_MBConv(**block_arg))
    input_size = _update_input_size(input_size, block_arg.stride)
    if block_arg.num_repeat > 1:
        block_arg.update(input_filters=out_filters, stride=1,
                         input_size=input_size)

    for _ in range(block_arg.num_repeat - 1):
        _blocks.append(ENAS_MBConv(**block_arg))

@enas_net(
    features = features,
    blocks = ENAS_Sequential(_blocks),
)
class ENAS_MBNet(nn.Module):
    def __init__(self, features, blocks, dropout_rate=0.2, num_classes=1000, input_size=224):
        super().__init__()
        self.features = features
        # blocks
        self.blocks = blocks
        # head
        Conv2D = get_same_padding_conv2d(input_size//32)
        self.conv_head = nn.Sequential(
            Conv2D(320, 1280, kernel_size=3, stride=2),
            nn.BatchNorm2d(1280),
            nn.ReLU(True),
        )
        # pool + fc
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self._dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.pool(x)
        x = self.flatten(x)
        if self._dropout:
            x = self._dropout(x)
        x = self.fc(x)
        return x

mbnet = ENAS_MBNet()
mbnet.cuda()
x = torch.rand(8, 3, 224, 224).cuda()
mbnet.evaluate_latency(x)

print('average latency is ', mbnet.avg_latency)

reward_fn = lambda metric, net: metric * ((net.avg_latency / net.latency) ** 0.1)

scheduler = Torch_ENAS_Scheduler(mbnet, train_set='imagenet', num_cpus=32, num_gpus=8,
                                 reward_fn=reward_fn, 
                                 warmup_epochs=5, epochs=120, controller_lr=1e-3,
                                 plot_frequency=0, update_arch_frequency=5)

scheduler.run()

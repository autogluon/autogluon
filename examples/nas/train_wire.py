import mxnet as mx
from mxnet import gluon

import autogluon as ag
from autogluon.core.space import *

from autogluon.contrib.wire import *

blocks_args = [
    Dict(kernel=3, num_repeat=3, channels=16, expand_ratio=3, stride=1, se_ratio=0.25, in_channels=32),
    Dict(kernel=3, num_repeat=4, channels=24, expand_ratio=3, stride=2, se_ratio=0.25, in_channels=16),
    Dict(kernel=3, num_repeat=4, channels=40, expand_ratio=3, stride=2, se_ratio=0.25, in_channels=24),
]

net = WireEfficientNet(blocks_args, num_classes=10)
net.initialize()

print(net)

scheduler = Wire_Scheduler(net, train_set='cifar', num_gpus=1, num_cpu=4, warmup_epochs=5,
                           controller_type='atten', checkname='./wire_atten_pen2/checkpoint.ag')
scheduler.run()


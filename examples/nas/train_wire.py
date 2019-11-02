import mxnet as mx
from mxnet import gluon

import autogluon as ag
from autogluon.core.space import *

from autogluon.contrib.wire import *

blocks_args = [
    Dict(num_repeat=6, kernel=3, channels=16, stride=1, in_channels=32),
    Dict(num_repeat=6, kernel=3, channels=24, stride=2, in_channels=16),
    Dict(num_repeat=6, kernel=3, channels=40, stride=2, in_channels=24),
]

net = WireNet(blocks_args, num_classes=10)
net.initialize()

print(net)

scheduler = Wire_Scheduler(net, train_set='cifar', num_gpus=1, num_cpu=4, warmup_epochs=5,
                           controller_type='atten', checkname='./wire_atten_Nov1/checkpoint.ag')
scheduler.run()


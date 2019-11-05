# Reproducing ProxylessNAS in 10 mins
:label:`sec_proxyless`

## What is the key idea of ENAS and ProxylessNAS

Traditional reinforcement learning based neural architecture search learns a architecture controller
by teratively sampling the architecture and training the model to get final reward to update the controller.
It is extremely expensive process due to training CNN.

![ProxylessNAS](https://autogluon.s3.amazonaws.com/_images/proxyless.png)

Recent work of ENAS and ProxylessNAS construct an over-parameterized network (supernet) and share the weights
across different architecutre to speed up the search speed. The reward is calculated every few iters instead
of every entire training period.

import MXNet and AutoGluon:

```{.python .input}
import autogluon as ag
import mxnet as mx
import mxnet.gluon.nn as nn
```

## How to construct a SuperNet?

Basic NN blocks for CNN.

```{.python .input}
class Identity(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return x
    
class ConvBNReLU(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, kernel, stride):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2D(channels, kernel, stride, padding, in_channels=in_channels)
        self.bn = nn.BatchNorm(in_channels=channels)
        self.relu = nn.Activation('relu')
    def hybrid_forward(self, F, x):
        return self.relu(self.bn(self.conv(x)))
```

### AutoGluon ENAS Unit

```{.python .input}
from autogluon.contrib.enas import *

@autogluon_enas_unit()
class ResUnit(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, hidden_channels, kernel, stride):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, kernel, stride)
        self.conv2 = ConvBNReLU(hidden_channels, channels, kernel, 1)
        if in_channels == channels and stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = nn.Conv2D(channels, 1, stride, in_channels=in_channels)
    def hybrid_forward(self, F, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)
```

### AutoGluon Sequntial

Creating a ENAS network using Sequential Block

```{.python .input}
mynet = ENAS_Sequential(
    ResUnit(1, 8, hidden_channels=ag.space.Categorical(4, 8), kernel=ag.space.Categorical(3, 5), stride=2),
    ResUnit(8, 8, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=2),
    ResUnit(8, 16, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=2),
    ResUnit(16, 16, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=1, with_zero=True),
    ResUnit(16, 16, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=1, with_zero=True),
    nn.GlobalAvgPool2D(),
    nn.Flatten(),
    nn.Activation('relu'),
    nn.Dense(10, in_units=16),
)

mynet.initialize()

print(mynet)
```

### Evaluate Network Latency and Define Reward Function

```{.python .input}
x = mx.nd.random.uniform(shape=(1, 1, 28, 28))
y = mynet.evaluate_latency(x)
```

Show the latencies:

```{.python .input}
print('Average latency is {}, latency of the current architecutre is {}'.format(mynet.avg_latency, mynet.latency))
```

We also provide number of params
```{.python .input}
mynet.nparams
```

```{.python .input}
reward_fn = lambda metric, net: metric * ((net.avg_latency / net.latency) ** 0.1)
```

## Start the Training

Construct experiment scheduler, which automatically cretes a RL controller based on user defined search space.

```{.python .input}
scheduler = ENAS_Scheduler(mynet, train_set='mnist',
                           reward_fn=reward_fn,
                           warmup_epochs=0, epochs=1, controller_lr=3e-3,
                           plot_frequency=1, update_arch_frequency=5)
```

Start the training:

```{.python .input}
scheduler.run()
```

The resulting architecture is:
```{.python .input}
mynet.graph
```

**Change to a different reward trade-off:**

```{.python .input}
reward_fn = lambda metric, net: metric * ((net.avg_latency / net.latency) ** 0.5)
scheduler = ENAS_Scheduler(mynet, train_set='mnist',
                           reward_fn=reward_fn,
                           warmup_epochs=0, epochs=1, controller_lr=3e-3,
                           plot_frequency=1, update_arch_frequency=5)
scheduler.run()
```

The resulting architecture is:
```{.python .input}
mynet.graph
```

## Defining a Complicated Network

Can we define a more complicated network than just sequential?

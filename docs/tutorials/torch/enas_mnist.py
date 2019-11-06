# Reproducing ENAS/ProxylessNAS in 10 mins
:label:`sec_torch_enas`

## What is the key idea of ENAS and ProxylessNAS

Traditional reinforcement learning based neural architecture search learns a architecture controller
by teratively sampling the architecture and training the model to get final reward to update the controller.
It is extremely expensive process due to training CNN.

![ProxylessNAS](https://autogluon.s3.amazonaws.com/_images/proxyless.png)

Recent work of ENAS and ProxylessNAS construct an over-parameterized network (supernet) and share the weights
across different architecutre to speed up the search speed. The reward is calculated every few iters instead
of every entire training period.

import PyTorch and AutoGluon:

```{.python .input}
import torch
import torch.nn as nn
import autogluon as ag
```

## How to construct a SuperNet?

Basic NN modules for CNN.

```{.python .input}
class Identity(torch.nn.Module):
    def forward(self, x):
        return x
    
class ConvBNReLU(torch.nn.Module):
    def __init__(self, in_channels, channels, kernel, stride):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(in_channels, channels, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

### AutoGluon ENAS Unit

```{.python .input}
from autogluon.contrib.torch.enas import *

@enas_unit()
class ResUnit(torch.nn.Module):
    def __init__(self, in_channels, channels, hidden_channels, kernel, stride):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, kernel, stride)
        self.conv2 = ConvBNReLU(hidden_channels, channels, kernel, 1)
        if in_channels == channels and stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, channels, 1, stride)
    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)
```

### AutoGluon Sequntial

Creating a ENAS network using Sequential Block

```{.python .input}
mynet = ENAS_Sequential(
    ResUnit(1, 8, hidden_channels=ag.Categorical(4, 8), kernel=ag.Categorical(3, 5), stride=2),
    ResUnit(8, 8, hidden_channels=8, kernel=ag.Categorical(3, 5), stride=2),
    ResUnit(8, 16, hidden_channels=8, kernel=ag.Categorical(3, 5), stride=2),
    ResUnit(16, 16, hidden_channels=8, kernel=ag.Categorical(3, 5), stride=1, with_zero=True),
    ResUnit(16, 16, hidden_channels=8, kernel=ag.Categorical(3, 5), stride=1, with_zero=True),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(16, 10),
)

mynet.initialize()

mynet.graph
```

### Evaluate Network Latency and Define Reward Function

```{.python .input}
x = torch.rand(1, 1, 28, 28)
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
from autogluon.contrib.torch.enas_scheduler import Torch_ENAS_Scheduler
scheduler = Torch_ENAS_Scheduler(mynet, train_set='mnist',
                                 reward_fn=reward_fn, batch_size=128,
                                 warmup_epochs=0, epochs=2, controller_lr=3e-3,
                                 plot_frequency=2, update_arch_frequency=5)
```

Start the training:

```{.python .input}
scheduler.run()
```

The resulting architecture is:
```{.python .input}
mynet.graph
```

**Change the reward trade-off:**

```{.python .input}
reward_fn = lambda metric, net: metric * ((net.avg_latency / net.latency) ** 0.8)
```

Reinitialize weights:

```{.python .input}
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
mynet.apply(weight_init)
```

```{.python .input}
scheduler = ENAS_Scheduler(mynet, train_set='mnist',
                           reward_fn=reward_fn, batch_size=128,
                           warmup_epochs=0, epochs=2, controller_lr=3e-3,
                           plot_frequency=2, update_arch_frequency=5)
scheduler.run()
```

The resulting architecture is:
```{.python .input}
mynet.graph
```

## Defining a Complicated Network

Can we define a more complicated network than just sequential?

```{.python .input}
@enas_net(
    unit1 = ResUnit(1, 8, hidden_channels=8, kernel=ag.Categorical(3, 5), stride=1),
    unit2 = ResUnit(8, 1, hidden_channels=8, kernel=3, stride=1),
    body = mynet,
)
class MNIST_Net(nn.Module):
    def __init__(self, unit1, unit2, body):
        super().__init__()
        self.unit1 = unit1
        self.unit2 = unit2
        self.body = body
        
    def forward(self, x):
        x = self.unit1(x)
        x = self.unit2(x)
        return self.body(x)
```

Evaluate the Latency:

```{.python .input}
mnist_net = MNIST_Net()
mnist_net.enas_modules.keys()
```

Start the training:

```{.python .input}
scheduler = Torch_ENAS_Scheduler(mnist_net, train_set='mnist',
                                 reward_fn=reward_fn, batch_size=128,
                                 warmup_epochs=0, epochs=1, controller_lr=3e-3,
                                 plot_frequency=2, update_arch_frequency=5)
# scheduler.run()
```

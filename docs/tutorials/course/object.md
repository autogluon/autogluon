# Searchable Objects
:label:`sec_customobj`

When defining custom Python objects such as network architectures,
or specialized optimizers, it may be hard to decide what values to set for all of their attributes. AutoGluon provides an API that allows you to instead specify  a search space of possible values to consider for such attributes, within which the optimal value will be automatically searched for at runtime. This tutorial demonstrates how easy this is to do, without having to modify your existing code at all!  

## Example for Constructing a Network

This tutorial covers an example of selecting a neural network's architecture as a hyperparameter optimization (HPO) task. If you are interested in efficient neural architecture search (NAS), please refer to this other tutorial instead: `sec_proxyless`_ .

### CIFAR ResNet in GluonCV

GluonCV provides [CIFARResNet](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/cifarresnet.py#L167-L183), which allow user to specify how many layers at each stage. For example, we can construct a CIFAR ResNet with only 1 layer per stage:

```{.python .input}
from gluoncv.model_zoo.cifarresnet import CIFARResNetV1, CIFARBasicBlockV1

layers = [1, 1, 1]
channels = [16, 16, 32, 64]
net = CIFARResNetV1(CIFARBasicBlockV1, layers, channels)
```

We can visualize the network:

```{.python .input}
import autogluon.core as ag
from autogluon.vision.utils import plot_network

plot_network(net, (1, 3, 32, 32))
```

### Searchable Network Architecture Using AutoGluon Object

:func:`autogluon.obj` enables customized search space to any user defined class. It can also be used within `autogluon.Categorical()` if you have multiple networks to choose from.


```{.python .input}
@ag.obj(
    nstage1=ag.space.Int(2, 4),
    nstage2=ag.space.Int(2, 4),
)
class MyCifarResNet(CIFARResNetV1):
    def __init__(self, nstage1, nstage2):
        nstage3 = 9 - nstage1 - nstage2
        layers = [nstage1, nstage2, nstage3]
        channels = [16, 16, 32, 64]
        super().__init__(CIFARBasicBlockV1, layers=layers, channels=channels)
```

Create one network instance and print the configuration space:

```{.python .input}
mynet=MyCifarResNet()
print(mynet.cs)
```

We can also overwrite existing search spaces:

```{.python .input}
mynet1 = MyCifarResNet(nstage1=1,
                       nstage2=ag.space.Int(5, 10))
print(mynet1.cs)
```

### Decorate Existing Class

We can also use :func:`autogluon.obj` to easily decorate any existing classes.
For example, if we want to search learning rate and weight decay for Adam optimizer, we only
need to add a decorator:

```{.python .input}
from mxnet import optimizer as optim
@ag.obj()
class Adam(optim.Adam):
    pass
```

Then we can create an instance:

```{.python .input}
myoptim = Adam(learning_rate=ag.Real(1e-2, 1e-1, log=True), wd=ag.Real(1e-5, 1e-3, log=True))
print(myoptim.cs)
```

### Launch Experiments Using AutoGluon Object

AutoGluon Object is compatible with Fit API in AutoGluon tasks, and also works with user-defined training
scripts using :func:`autogluon.autogluon_register_args`. We can start fitting:

```{.python .input}
from autogluon.vision import ImagePredictor as Task
classifier = Task()
classifier.fit('cifar10', ngpus_per_trial=1, hyperparameters={'net': mynet, 'optimizer': myoptim, 'epochs': 1})
```

```{.python .input}
print(classifier.fit_summary())
```

# Tune PyTorch Model on MNIST
:label:`sec_customstorch`

In this tutorial, we demonstrate how to do Hyperparameter Optimization (HPO) using AutoGluon with PyTorch.
AutoGluon is a framework agnostic HPO toolkit, which is compatible with any training code written in python. The PyTorch code used in this tutorial is adapted from this [git repo](https://github.com/kuangliu/pytorch-cifar). In your applications, this code can be replaced with your own PyTorch code.

Import the packages:

```{.python .input}
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
```

## Start with an MNIST Example

### Data Transforms

We first apply standard image transforms to our training and validation data:

```{.python .input}
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

# the datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

### Main Training Loop

The following `train_mnist` function represents normal training code a user would write for
training on MNIST dataset. Python users typically use an argparser to conveniently
change default values. The only additional argument you need to add to your existing python function is a reporter object that is used to store performance achieved under different hyperparameter settings.

```{.python .input}
def train_mnist(args, reporter):
    # get variables from args
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    net = args.net
    print('lr: {}, wd: {}'.format(lr, wd))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model
    net = net.to(device)

    if device == 'cuda':
        net = nn.DataParallel(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=wd)

    # datasets and dataloaders
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Training
    def train(epoch):
        net.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def test(epoch):
        net.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        # 'epoch' reports the number of epochs done
        reporter(epoch=epoch+1, accuracy=acc)

    for epoch in tqdm(range(0, epochs)):
        train(epoch)
        test(epoch)
```
## AutoGluon HPO

In this section, we cover how to define a searchable network architecture, convert the training function to be searchable, create the scheduler, and then launch the experiment.

### Define a Searchable Network Achitecture

Let's define a 'dynamic' network with searchable configurations by simply adding a decorator :func:`autogluon.obj`. In this example, we only search two arguments `hidden_conv` and
`hidden_fc`, which represent the hidden channels in convolutional layer and fully connected layer. 
More info about searchable space is available at :meth:`autogluon.core.space`.

```{.python .input}
import autogluon.core as ag

@ag.obj(
    hidden_conv=ag.space.Int(6, 12),
    hidden_fc=ag.space.Categorical(80, 120, 160),
)
class Net(nn.Module):
    def __init__(self, hidden_conv, hidden_fc):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_conv, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hidden_conv, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Convert the Training Function to Be Searchable

We can simply add a decorator :func:`autogluon.args` to convert the `train_mnist` function argument values to be tuned by AutoGluon's hyperparameter optimizer. In the example below, we specify that the lr argument is a real-value that should be searched on a log-scale in the range 0.01 - 0.2. Before passing lr to your train function, AutoGluon always selects an actual floating point value to assign to lr so you do not need to make any special modifications to your existing code to accommodate the hyperparameter search.

```{.python .input}
@ag.args(
    lr = ag.space.Real(0.01, 0.2, log=True),
    wd = ag.space.Real(1e-4, 5e-4, log=True),
    net = Net(),
    epochs=5,
)
def ag_train_mnist(args, reporter):
    return train_mnist(args, reporter)
```


### Create the Scheduler and Launch the Experiment

For hyperparameter tuning, AutoGluon provides a number of different schedulers:

- `FIFOScheduler`: Each training jobs runs for the full number of epochs
- `HyperbandScheduler`: Uses successive halving and Hyperband scheduling in
   order to stop unpromising jobs early, so that the available budget is allocated
   more efficiently

Each scheduler is internally configured by a searcher, which determines the choice
of hyperparameter configurations to be run. The default searcher is `random`:
configurations are drawn uniformly at random from the search space.

```{.python .input}
myscheduler = ag.scheduler.FIFOScheduler(
    ag_train_mnist,
    resource={'num_cpus': 4, 'num_gpus': 1},
    num_trials=2,
    time_attr='epoch',
    reward_attr='accuracy')
print(myscheduler)
```

```{.python .input}
myscheduler.run()
myscheduler.join_jobs()
```

We plot the test accuracy achieved over the course of training under each hyperparameter configuration that AutoGluon tried out (represented as different colors).

```{.python .input}
myscheduler.get_training_curves(plot=True,use_legend=False)
print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                               myscheduler.get_best_reward()))
```

### Search by Bayesian Optimization

While simple to implement, random search is usually not an efficient way to
propose configurations for evaluation. AutoGluon provides a number of
model-based searchers:

- Gaussian process based Bayesian optimization (`bayesopt`)
- SkOpt Bayesian optimization (`skopt`; only with FIFO scheduler)

Here, `skopt` maps to [scikit.optimize](https://scikit-optimize.github.io/stable/),
whereas `bayesopt` is an own implementation. While `skopt` is currently somewhat
more versatile (choice of acquisition function, surrogate model), `bayesopt`
is directly optimized to asynchronous parallel scheduling. Importantly, `bayesopt`
runs both with FIFO and Hyperband scheduler (while `skopt` is restricted to the
FIFO scheduler).

When running the following examples, comparing the different schedulers and
searchers, you need to increase `num_trials` (or use `time_out` instead, which
specifies the search budget in terms of wall-clock time) in order to see
differences in performance.

```{.python .input}
myscheduler = ag.scheduler.FIFOScheduler(
    ag_train_mnist,
    resource={'num_cpus': 4, 'num_gpus': 1},
    searcher='bayesopt',
    num_trials=2,
    time_attr='epoch',
    reward_attr='accuracy')
print(myscheduler)
```

```{.python .input}
myscheduler.run()
myscheduler.join_jobs()
```

### Search by Asynchronous BOHB

When training neural networks, it is often more efficient to use early stopping,
and in particular Hyperband scheduling can save a lot of wall-clock time. AutoGluon
provides a combination of Hyperband scheduling with asynchronous Bayesian
optimization (more details can be found [here](https://arxiv.org/abs/2003.10865)):

```{.python .input}
myscheduler = ag.scheduler.HyperbandScheduler(
    ag_train_mnist,
    resource={'num_cpus': 4, 'num_gpus': 1},
    searcher='bayesopt',
    num_trials=2,
    time_attr='epoch',
    reward_attr='accuracy',
    grace_period=1,
    reduction_factor=3,
    brackets=1)
print(myscheduler)
```

```{.python .input}
myscheduler.run()
myscheduler.join_jobs()
```

**Tip**: If you like to learn more about HPO algorithms in AutoGluon, please
have a look at :ref:`sec_custom_advancedhpo`.

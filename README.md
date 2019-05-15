AutoGluon: AutoML Toolkit with MXNet Gluon
===
![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![build](https://img.shields.io/appveyor/ci/:user/:repo.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

## Table of Contents

[TOC]

## Installation
    python setup.py install

## Beginners Guide
```python
import logging
import autogluon as ag
import autogluon.image_classification as task

train_dataset, valid_dataset = task.Dataset('./CIFAR10/train', 
                                            './CIFAR10/valid')

models = task.fit(train_dataset)

logging.info('trials results:')
logging.info(models[0])
```

## Advanced User Guide
```python
import logging

import autogluon as ag
import autogluon.image_classification as task

train_dataset, valid_dataset = task.Dataset('./CIFAR10/train', './CIFAR10/valid')

models, best_result, search_space = task.fit(train_dataset,
                                             nets=ag.Nets([task.model_zoo.get_model('resnet18_v1'),
                                                   task.model_zoo.get_model('resnet34_v1'),
                                                   task.model_zoo.get_model('resnet50_v1'),
                                                   task.model_zoo.get_model('resnet101_v1'),
                                                   task.model_zoo.get_model('resnet152_v1')]),
                                             optimizers=ag.Optimizers([ag.optims.get_optim('sgd'),
                                                         ag.optims.get_optim('adam')]))
logging.info('trials results:')
logging.info(models)
logging.info('=========================')
logging.info('best results:')
logging.info(best_result)
logging.info('=========================')
logging.info('print search space')
logging.info(search_space)

```

## Auto Fit Usage
```python
def fit(data,
        nets,
        optimizers=None,
        metrics=None,
        losses=None,
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='./outputdir/',
        visualizer='tensorboard',
        stop_criterion={'time_limits': 1 * 60 * 60,
                        'max_metric': 0.80,
                        'max_trial_count': 100},
        resources_per_trial={'max_num_gpus': 1,
                             'max_num_cpus': 4,
                             'max_training_epochs': 2},
        *args):
    cs = CS.ConfigurationSpace()
    assert data is not None
    assert nets is not None
    if data.search_space is not None:
        cs.add_configuration_space(data.search_space)
    if nets.search_space is not None:
        cs.add_configuration_space(nets.search_space)
    if optimizers.search_space is not None:
        cs.add_configuration_space(optimizers.search_space)
    if metrics.search_space is not None:
        cs.add_configuration_space(metrics.search_space)
    if losses.search_space is not None:
        cs.add_configuration_space(losses.search_space)
    import json
    with open('config_space.json', 'w') as f:
        f.write(json.write(cs))
    with open('config_space.json') as f:
        search_space = json.load(f)

    if searcher is None:
        searcher = tune.automl.search_policy.RandomSearch(search_space,
                                                          stop_criterion['max_metric'],
                                                          stop_criterion['max_trial_count'])
    if trial_scheduler is None:
        trial_scheduler = tune.schedulers.FIFOScheduler()

    tune.register_trainable(
        "TRAIN_FN", lambda config, reporter: pipeline.train_image_classification(
            args, config, reporter))
    trials = tune.run(
        "TRAIN_FN",
        name=args.expname,
        verbose=2,
        scheduler=trial_scheduler,
        **{
            "stop": {
                "mean_accuracy": stop_criterion['max_metric'],
                "training_iteration": resources_per_trial['max_training_epochs']
            },
            "resources_per_trial": {
                "cpu": int(resources_per_trial['max_num_cpus']),
                "gpu": int(resources_per_trial['max_num_gpus'])
            },
            "num_samples": resources_per_trial['max_trial_count'],
            "config": {
                "lr": tune.sample_from(lambda spec: np.power(
                    10.0, np.random.uniform(-4, -1))),
                "momentum": tune.sample_from(lambda spec: np.random.uniform(
                    0.85, 0.95)),
            }
        })
    best_result = max([trial.best_result for trial in trials])
    return trials, best_result, cs
```

## Auto Nets Usage
```python
nets = ag.Nets([ag.task.model_zoo.get_model('resnet18_v1'),
                ag.task.model_zoo.get_model('resnet34_v1'),
                ag.task.model_zoo.get_model('resnet50_v1'),
                ag.task.model_zoo.get_model('resnet101_v1'),
                ag.task.model_zoo.get_model('resnet152_v1')])
logging.info(nets)
```

Some implementaion details:
```python
def add_search_space(self):
    cs = CS.ConfigurationSpace()
    net_list_hyper_param = List('autonets', choices=self.net_list).get_hyper_param()
    cs.add_hyperparameter(net_list_hyper_param)
    for net in self.net_list:
        #TODO(cgraywang): distinguish between different nets, only support resnet for now
        net_hyper_params = net.get_hyper_params()
        cs.add_hyperparameters(net_hyper_params)
        conds = []
        for net_hyper_param in net_hyper_params:
            #TODO(cgraywang): put condition in presets? split task settings out
            cond = CS.InCondition(net_hyper_param, net_list_hyper_param,
                                  ['resnet18_v1', 'resnet34_v1',
                                   'resnet50_v1', 'resnet101_v1',
                                   'resnet152_v1'])
            conds.append(cond)
        cs.add_conditions(conds)
    self._set_search_space(cs)

@autogluon_nets
def get_model(name):
    name = name.lower()
    if name not in models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(models)))
        raise ValueError(err_str)
    net = name
    return net
```

## Auto Optimizers Usage
```python
optims = ag.Optimizers([ag.Optimizers([ag.optims.get_optim('sgd'),
                                       ag.optims.get_optim('adam')])])
logging.info(optims)
```

Some implementation details:
```python
@autogluon_optims
def get_optim(name):
    name = name.lower()
    if name not in optims:
        err_str = '"%s" is not among the following optim list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(optims)))
        raise ValueError(err_str)
    optim = name
    return optim
```

## Auto Space Usage
- Categorical space
```python
list_space = ag.space.List('listspace', ['0',
                                         '1',
                                         '2'])
logging.info(list_space)
```
- Linear space
```python
linear_space = ag.space.Linear('linspace', 0, 10)
logging.info(linear_space)
```
- Log space
```python
log_space = ag.space.Log('logspace', 10**-10, 10**-1)
logging.info(log_space)
```

## Understand Task, Resource and Scheduler

`Resources` contains number of cpus and gpus. `Task` includes execute
function, args and its resource.
`TaskScheduler` schedule the tasks right away, but `my_task` is executed
only when its resource is ready.

```python
from  autogluon.scheduler import Task, Resources, TaskScheduler
import time

def my_task():
    print('running, my_task')
    time.sleep(3.0)

scheduler = TaskScheduler()

for i in range(10):
    resource = Resources(num_cpus=2, num_gpus=0)
    task = Task(my_task, {}, resource)
    scheduler.add_task(task)
```

See [`test_scheduler.py`](./tests/unitests/test_schedulter.py)

## Hyper-parameter Tuning

### Schedule Different Configs

Adding `@autogluon` decorator to the original training func, and
define the training config.

```python
import autogluon as ag
from autogluon import autogluon_method
from  autogluon.scheduler import Task, Resources, TaskScheduler

@autogluon_method
def train_fn(args):
    # original code
    pass

args = parser.parse_args()
config = {'lr': ag.distribution.sample_from(
          lambda: np.power(10.0, np.random.uniform(-4, -1)))}
myscheduler = TaskScheduler()

for i in range(5):
    resource = Resources(num_cpus=2, num_gpus=0)
    task = Task(train_fn, {'args': args, 'config': config}, resource)
    myscheduler.add_task(task)
```

See [`mnist_autogluon_numpy_config.py`](./examples/mnist_autogluon_numpy_config.py) for details.

### ConfigSpace and Random Searcher with FIFO Scheduler

```python
@autogluon_method
def train_mnist(args, reporter):
    for e in range(epochs):
        # forward, backward, optimizer step and evaluation metric
        reporter(accuracy=top1_accuracy)

args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

# create hyperparameters configuration
cs = CS.ConfigurationSpace()
lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, log=True)
cs.add_hyperparameter(lr)

# construct searcher from the configuration space
searcher = ag.searcher.RandomSampling(cs)

# use FIFO scheduler
myscheduler = ag.scheduler.FIFO_Scheduler(train_mnist, args,
                                          {'num_cpus': 2, 'num_gpus': 2}, searcher)
# run tasks
myscheduler.run(num_trials=10)
```

See [`mnist_autogluon.py`](./examples/mnist_autogluon.py) for details.

## Schedule Bash Tasks

Schedule the tasks using bash based on the resource management:

```python
import subprocess
import numpy as np
import autogluon as ag

def my_task(lr):
    subprocess.run(['python', 'examples/mnist_native.py', '--lr', str(lr)])

scheduler = ag.scheduler.TaskScheduler()
for i in range(10):
    lr = np.power(10.0, np.random.uniform(-4, -1))
    resource = ag.scheduler.Resources(num_cpus=2, num_gpus=1)
    task = ag.scheduler.Task(my_task, {'lr': lr}, resource)
    scheduler.add_task(task)
```

See [`bash_scheduler.py`](./examples/bash_scheduler.py).

## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::

###### tags: `Templates` `Documentation`
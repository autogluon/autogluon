AutoGluon: AutoML Toolkit with MXNet Gluon
===
![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![build](https://img.shields.io/appveyor/ci/:user/:repo.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

## Table of Contents

[TOC]

## Installation
    git clone ssh://git.amazon.com/pkg/AutoGluon
    cd AutoGluon
    python setup.py install


## Contribution Guide

Step 1: Install the ![mwinit](https://midway.amazon.com/nextgen/ssh)

Step 2: Install toolbox, brazil and cr following this ![wiki](https://w.amazon.com/index.php/BuilderToolbox/GettingStarted#Install_Toolbox)

Step 3: Use git commit to add and commit any changes you wish to have in your code review, but do not push them

Step 4 (one time): make sure using curl from /usr/bin. update ~/.bash_profile with the below command to be the last line in the file 
    
    export PATH="/usr/bin:$PATH"

Step 5: Type cr to begin (CRUX will generate a URL of the form: https://code.amazon.com/reviews/CR-######.)
In the generated link, you could select reviewers. Once everything is set. Please do not forget `Publish` top-right corner of the CR URL.

For more detailed information on how to submit PR and revisions, please refer to ![cr link](https://builderhub.corp.amazon.com/docs/getting-started-cr.html)


## Beginners Guide
```python
import logging

from autogluon import image_classification as task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

dataset = task.Dataset(name='CIFAR10')
results = task.fit(dataset)

logger.debug('Best result:')
logger.debug(results.val_accuracy)
logger.debug('=========================')
logger.debug('Best search space:')
logger.debug(results.config)
logger.debug('=========================')
logger.debug('Total time cost:')
logger.debug(results.time)
```

## Advanced User Guide
```python
import logging

import autogluon as ag
from autogluon import image_classification as task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

dataset = task.Dataset(name='CIFAR10')

results = task.fit(dataset,
                   nets=ag.Nets(['cifar_resnet20_v1',
                                 'cifar_resnet56_v1',
                                 'cifar_resnet110_v1']),
                   optimizers=ag.Optimizers(['sgd', 'adam']))

logger.debug('Best result:')
logger.debug(results.val_accuracy)
logger.debug('=========================')
logger.debug('Best search space:')
logger.debug(results.config)
logger.debug('=========================')
logger.debug('Total time cost:')
logger.debug(results.time)

```

## Auto Fit Usage
```python
def fit(data,
        nets=Nets([
            get_model('cifar_resnet20_v1'),
            get_model('cifar_resnet56_v1'),
            get_model('cifar_resnet110_v1')]),
        optimizers=Optimizers(
            [get_optim('sgd'),
             get_optim('adam')]),
        metrics=None,
        losses=None,
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='checkpoint/exp1.ag',
        visualizer='tensorboard',
        stop_criterion={
            'time_limits': 1*60*60,
            'max_metric': 1.0,
            'max_trial_count': 2
        },
        resources_per_trial={
            'max_num_gpus': 0,
            'max_num_cpus': 4,
            'max_training_epochs': 3
        },
        backend='default',
        demo=False,
        **kwargs):
    r"""
    Fit networks on dataset

    Parameters
    ----------
    data: Input data. It could be:
        autogluon.Datasets
        task.Datasets
    nets: autogluon.Nets
    optimizers: autogluon.Optimizers
    metrics: autogluon.Metrics
    losses: autogluon.Losses
    stop_criterion (dict): The stopping criteria. The keys may be any field in
        the return result of 'train()', whichever is reached first.
        Defaults to empty dict.
    resources_per_trial (dict): Machine resources to allocate per trial,
        e.g. ``{"max_num_cpus": 64, "max_num_gpus": 8}``. Note that GPUs will not be
        assigned unless you specify them here.
    savedir (str): Local dir to save training results to.
    searcher: Search Algorithm.
    trial_scheduler: Scheduler for executing
        the experiment. Choose among FIFO (default) and HyperBand.
    resume (bool): If checkpoint exists, the experiment will
        resume from there.
    backend: support autogluon default backend, ray. (Will support SageMaker)
    **kwargs: Used for backwards compatibility.

    Returns
    ----------
    results:
        model: the parameters associated with the best model. (TODO:)
        val_accuracy: validation set accuracy
        config: best configuration
        time: total time cost
    """
    logger.info('Start fitting')
    start_fit_time = time.time()

    def _construct_search_space(objs, obj_names):
        def _set_range(obj, name):
            pass
        def _assert_fit_error(obj, name):
            pass
        def _init_args(cs):
            pass
        return cs, args
    logger.info('Start constructing search space')
    search_objs = [data, nets, optimizers, losses, metrics]
    # TODO (cgraywang) : replace with autogluon*.name
    search_obj_names = ['data', 'net', 'optimizer', 'loss', 'metric']
    cs, args = _construct_search_space(search_objs, search_obj_names)
    logger.info('Finished.')

    def _reset_checkpoint(dir, resume):
        pass
    _reset_checkpoint(savedir, resume)

    def _run_backend(searcher, trial_scheduler):
        logger.info('Start using default backend.')

        if searcher is None or searcher == 'random':
            searcher = ag.searcher.RandomSampling(cs)
        if trial_scheduler == 'hyperband':
            trial_scheduler = ag.scheduler.Hyperband_Scheduler()
            # TODO (cgraywang): use empiral val now
        else:
            trial_scheduler = ag.scheduler.FIFO_Scheduler()
        trial_scheduler.run()
        # TODO (cgraywang)
        trials = None
        best_result = trial_scheduler.get_best_reward()
        best_config = trial_scheduler.get_best_config()
        results = Results(trials, best_result, best_config, time.time() - start_fit_time)
        logger.info('Finished.')
        return results
    results = _run_backend(searcher, trial_scheduler)
    logger.info('Finished.')
    return results
```

## Auto Nets Usage
```python
nets = ag.Nets(['resnet18_v1',
                'resnet34_v1',
                'resnet50_v1',
                'resnet101_v1',
                'resnet152_v1'])
logging.info(nets)
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
# AutoGluon

## Install
```bash
python setup.py install
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

See [`mnist_autogluon_config_random.py`](./examples/mnist_autogluon_config_random.py) for details.

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

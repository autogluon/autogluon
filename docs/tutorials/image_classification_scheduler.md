# Image Classification - Configure Your Trial Scheduler (How to Stop Early)
:label:`sec_imgscheduler`

This tutorial dives into how to configure and create your own trial scheduler, such as stopping the running trials earlier, and use the customized trial scheduler in the fitting process.

We again begin by letting AutoGluon know that `image_classification` is the task of interest: 

```{.python .input}
from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)
```

## Use fifo trial scheduler

Let's first create the dataset using the same subset of the `Shopee-IET` dataset as before.
Recall that as we only specify the `train_path`, a 90/10 train/validation split is automatically performed.
We also set the time limits and training epochs to ensure the demo run quickly.

In AutoGluon, [autogluon.scheduler](../api/autogluon.scheduler.html) orchestrates how individual training jobs are scheduled.

AutoGluon currently supports scheduling trials in serial order and with early stopping (eg. if the performance of the model early within training already looks bad, the trial may be terminated early to free up resources).
We support a serial [FIFO scheduler](../api/autogluon.scheduler.html#autogluon.scheduler.FIFO_Scheduler) as default trial scheduler.
The simplest way to specify the serial scheduler to be used is via the string name `scheduler_fifo = 'fifo'`:

```{.python .input}
dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')

time_limits = 2*60
epochs = 10

results = task.fit(dataset,
                   time_limits=time_limits,
                   epochs=epochs)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % results.reward)
test_dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Use an early stopping trial scheduler - Hyperband

We could easily leverage the early stopping scheduler: [Hyperband](../api/autogluon.scheduler.html#autogluon.scheduler.Hyperband) make the fit procedure more efficient.
HyperBandScheduler early stops trials using the HyperBand optimization algorithm. It divides trials into brackets of varying sizes, and periodically early stops low-performing trials within each bracket.
We could simply specify Hyperband via string name and use it in the `fit` function:

```{.python .input}
scheduler = 'hyperband'

results = task.fit(dataset,
                   scheduler=scheduler,
                   time_limits=time_limits,
                   epochs=epochs)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % results.reward)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Create your own trial scheduler

We could also create our own trial scheduler. Here is an example of creating the Median Stopping scheduler. It is a simple stopping rule, which stops the trials with the rewards less than the median of the rewards at the same number of iters in the history.


```{.python .input}
import collections
import numpy as np
import multiprocessing as mp
from autogluon.scheduler import Hyperband


class MedianStopping_Scheduler(Hyperband):
    def __init__(self, train_fn, args, resource, searcher,
                 checkpoint='./exp/checkerpoint.ag', 
                 resume=False,
                 num_trials=None,
                 time_attr="training_epoch",
                 reward_attr="accuracy",
                 visualizer='tensorboard',
                 mode="max",
                 grace_period=1.0,
                 min_samples_required=3):
        super(MedianStopping_Scheduler, self).__init__(train_fn=train_fn, args=args, resource=resource,
                                                       searcher=searcher, checkpoint=checkpoint,
                                                       resume=resume, num_trials=num_trials,
                                                       time_attr=time_attr, reward_attr=reward_attr,
                                                       visualizer=visualizer)
        self.terminator = MediamStoppingRule(time_attr, reward_attr, mode, grace_period,
                                             min_samples_required)
    def state_dict(self, destination=None):
        pass

class MediamStoppingRule(object):
    LOCK = mp.Lock()
    def __init__(self,
                 time_attr="training_epoch",
                 reward_attr="accuracy",
                 mode="max",
                 grace_period=1,
                 min_samples_required=3):
        self._time_attr = time_attr
        self._reward_attr = reward_attr
        self._stopped_tasks = set()
        self._completed_tasks = set()
        self._results = collections.defaultdict(list)
        self._grace_period = grace_period
        self._min_samples_required = min_samples_required
        self._metric = reward_attr
        if mode == "max":
            self._metric_op = 1.
        elif mode == "min":
            self._metric_op = -1.
 
    def on_task_add(self, task):
        pass
 
    def on_task_report(self, task, result):
        # return True/False, which indicates whether we want to continue
        time = result[self._time_attr]
        self._results[task].append(result)
        median_result = self._get_median_result(time)
        best_result = self._best_result(task)
 
        if best_result < median_result and time > self._grace_period:
            self._stopped_tasks.add(task)
            return False
        else:
            return True

    def on_task_complete(self, task, result):
        self._results[task].append(result)
        self._completed_tasks.add(task)

    def on_task_remove(self, task):
        if task in self._results:
            self._completed_tasks.add(task)

    def _get_median_result(self, time):
        scores = []
        for task in self._completed_tasks:
            scores.append(self._running_result(task, time))
        if len(scores) >= self._min_samples_required:
            return np.median(scores)
        else:
            return float("-inf")

    def _running_result(self, task, t_max=float("inf")):
        results = self._results[task]
        return np.max([self._metric_op * r[self._metric] 
                       for r in results if r[self._time_attr] <= t_max])

    def _best_result(self, task):
        results = self._results[task]
        return max(self._metric_op * r[self._metric] for r in results)

    def __repr__(self):
        return "MedianStoppingRule: num_stopped={}.".format(
            len(self._stopped_tasks))
```

Then we can use our defined scheduler:

```{.python .input}
results = task.fit(dataset,
                   algorithm=MedianStopping_Scheduler,
                   time_limits=time_limits,
                   epochs=epochs)
```

Print the result:

```{.python .input}
print('Top-1 val acc: %.3f' % results.reward)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

For more complete usage of `fit` function, please refer to the [fit API](../api/autogluon.task.image_classification.html#autogluon.task.image_classification.ImageClassification.fit).

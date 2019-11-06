# Image Classification - Configure Your Trial Scheduler (How to Stop Early)
:label:`sec_imgscheduler`

This tutorial dives into how to configure and create your own trial scheduler, such as stopping the running trials earlier, and use the customized trial scheduler in the fitting process.

We again begin by letting AutoGluon know that `image_classification` is the task of interest: 

```{.python .input}
from autogluon import ImageClassification as task
```

## Use fifo trial scheduler

Let's first create the dataset using the same subset of the `Shopee-IET` dataset as before.
Recall that as we only specify the `train_path`, a 90/10 train/validation split is automatically performed.
We also set the time limits and training epochs to ensure the demo run quickly.

In AutoGluon, [autogluon.scheduler](../api/autogluon.scheduler.html) orchestrates how individual training jobs are scheduled.

AutoGluon currently supports scheduling trials in serial order and with early stopping (eg. if the performance of the model early within training already looks bad, the trial may be terminated early to free up resources).
We support a serial [FIFO scheduler](../api/autogluon.scheduler.html#autogluon.scheduler.FIFO_Scheduler) as default trial scheduler.

```{.python .input}
dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')

classifier = task.fit(dataset,
                      time_limits=2*60,
                      epochs=10,
                      ngpus_per_trial=1)
```

The validation and test top-1 accuracy are:

```{.python .input}
# print('Top-1 val acc: %.3f' % classifier.results[classifier.results['reward_attr']])
test_dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Use an early stopping trial scheduler - Hyperband

We could easily leverage the early stopping scheduler: [Hyperband](../api/autogluon.scheduler.html#autogluon.scheduler.Hyperband) make the fit procedure more efficient.
HyperBandScheduler early stops trials using the HyperBand optimization search_strategy. It divides trials into brackets of varying sizes, and periodically early stops low-performing trials within each bracket.
We could simply specify Hyperband via string name and use it in the `fit` function:

```{.python .input}
search_strategy = 'hyperband'

classifier = task.fit(dataset,
                      search_strategy=search_strategy,
                      time_limits=2*60,
                      epochs=10,
                      ngpus_per_trial=1)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % classifier.results[classifier.results['reward_attr']])
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

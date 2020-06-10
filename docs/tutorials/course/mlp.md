# Getting started with Advanced HPO Algorithms

## Loading libraries

```{.python .input  n=1}
# Basic utils for folder manipulations etc
import os
import shutil
import time
import re
import multiprocessing # to count the number of CPUs available

# External tools to load and process data
import openml
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# MXNet (NeuralNets)
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

# AutoGluon and HPO tools
import autogluon as ag
```

Check the version of MxNet, you should be fine with version >= 1.5

```{.python .input  n=2}
mx.__version__
```

You can also check the version of AutoGluon and the specific commit and check that it matches what you want.

```{.python .input  n=3}
ag.__version__
```

## Hyperparameter Optimization of a 2-layer MLP

### Setting up the context

Here
we declare a few "environment variables" setting the context for what we're
doing

```{.python .input  n=4}
OPENML_TASK_ID = 6                # describes the problem we will tackle
RATIO_TRAIN_VALID = 0.33          # split of the training data used for validation
RESOURCE_ATTR_NAME = 'epoch'      # how do we measure resources   (will become clearer further)
REWARD_ATTR_NAME = 'objective'    # how do we measure performance (will become clearer further)

NUM_CPUS = multiprocessing.cpu_count()
```

### Preparing the data

Set the openml directory to the current directory (next to the notebook) and download the task if the file is not already present.

```{.python .input  n=5}
openml.config.set_cache_directory("./")
task = openml.tasks.get_task(OPENML_TASK_ID)
```

It's a multiclass classification task with 26 classes:

```{.python .input  n=6}
n_classes = len(task.class_labels)
n_classes
```

OpenML provides a standard train/test split that we can use.

```{.python .input  n=7}
train_indices, test_indices = task.get_train_test_split_indices()

X, y = task.get_X_and_y()

X.shape
```

Impute any missing value with a basic strategy and recover the training data

```{.python .input  n=8}
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

X_train = X[train_indices]
y_train = y[train_indices]
```

Resplit the training data into training+validation using the specific fraction indicated earlier

```{.python .input  n=9}
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_train, y_train, random_state=1, test_size=RATIO_TRAIN_VALID)
```

Standardize the data

```{.python .input  n=10}
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / (std + 1e-10)
X_valid = (X_valid - mean) / (std + 1e-10)
```

### Declaring a model specifying a hyperparameter space with AutoGluon

Two layer MLP where we optimize over:

- the number of units on the first layer
- the number of units on the second layer
- the dropout rate after each layer
- the learning rate
- the scaling
- the `@ag.args` decorator allows us to specify the space we will optimize over, this matches the [ConfigSpace](https://automl.github.io/ConfigSpace/master/) syntax

The body of the function `run_mlp_openml` is pretty simple:

- it reads the hyperparameters given via the decorator
- it defines a 2 layer MLP with dropout
- it declares a trainer with the 'adam' loss function and a provided learning rate
- it trains the NN with a number of epochs (most of that is boilerplate code from `mxnet`)
- the `reporter` at the end is used to keep track of training history in the hyperparameter optimization

**Note**: The number of epochs and the hyperparameter space are reduced to make for a shorter experiment

```{.python .input  n=11}
@ag.args(n_units_1=ag.space.Int(lower=16, upper=128),
         n_units_2=ag.space.Int(lower=16, upper=128),
         dropout_1=ag.space.Real(lower=0, upper=.75),
         dropout_2=ag.space.Real(lower=0, upper=.75),
         learning_rate=ag.space.Real(lower=1e-6, upper=1, log=True),
         batch_size=ag.space.Int(lower=8, upper=128),
         scale_1=ag.space.Real(lower=0.001, upper=10, log=True),
         scale_2=ag.space.Real(lower=0.001, upper=10, log=True),
         epochs=9)
def run_mlp_openml(args, reporter, **kwargs):
    # Time stamp for elapsed_time
    ts_start = time.time()
    # Unwrap hyperparameters
    n_units_1 = args.n_units_1
    n_units_2 = args.n_units_2
    dropout_1 = args.dropout_1
    dropout_2 = args.dropout_2
    scale_1 = args.scale_1
    scale_2 = args.scale_2
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    ctx = mx.cpu()
    net = nn.Sequential()
    with net.name_scope():
        # Layer 1
        net.add(nn.Dense(n_units_1, activation='relu',
                         weight_initializer=mx.initializer.Uniform(scale=scale_1)))
        # Dropout
        net.add(gluon.nn.Dropout(dropout_1))
        # Layer 2
        net.add(nn.Dense(n_units_2, activation='relu',
                         weight_initializer=mx.initializer.Uniform(scale=scale_2)))
        # Dropout
        net.add(gluon.nn.Dropout(dropout_2))
        # Output
        net.add(nn.Dense(n_classes))
    net.initialize(ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate})

    for epoch in range(args.epochs):
        ts_epoch = time.time()

        train_iter = mx.io.NDArrayIter(
                        data={'data': X_train}, 
                        label={'label': y_train},
                        batch_size=batch_size, 
                        shuffle=True)
        valid_iter = mx.io.NDArrayIter(
                        data={'data': X_valid}, 
                        label={'label': y_valid},
                        batch_size=batch_size, 
                        shuffle=False)

        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        for batch in train_iter:
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
            L.backward()
            trainer.step(data.shape[0])
            metric.update([label], [output])

        name, train_acc = metric.get()

        metric = mx.metric.Accuracy()
        for batch in valid_iter:
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)
            output = net(data)
            metric.update([label], [output])

        name, val_acc = metric.get()

        print('Epoch %d ; Time: %f ; Training: %s=%f ; Validation: %s=%f' % (
            epoch + 1, time.time() - ts_start, name, train_acc, name, val_acc))

        ts_now = time.time()
        eval_time = ts_now - ts_epoch
        elapsed_time = ts_now - ts_start

        # The resource reported back (as 'epoch') is the number of epochs
        # done, starting at 1
        reporter(
            epoch=epoch + 1, 
            objective=float(val_acc), 
            eval_time=eval_time,
            time_step=ts_now, 
            elapsed_time=elapsed_time)
```

**Note**: The annotation `epochs=9` specifies the maximum number of epochs for
training. It becomes available as `args.epochs`. Importantly, it is also
processed by `HyperbandScheduler` below in order to set its `max_t` attribute.

**Recommendation**: Whenever writing training code to be passed as `train_fn` to
a scheduler, if this training code reports a resource (or time) attribute, the
corresponding maximum resource value should be included in `train_fn.args`:

- If the resource attribute (`time_attr` of scheduler) in `train_fn` is `epoch`,
  make sure to include `epochs=XYZ` in the annotation. This allows the scheduler
  to read `max_t` from `train_fn.args.epochs`. This case corresponds to our
  example here.
- If the resource attribute is something else than `epoch`, you can also include
  the annotation `max_t=XYZ`, which allows the scheduler to read `max_t` from
  `train_fn.args.max_t`.

Annotating the training function by the correct value for `max_t` simplifies
scheduler creation (since `max_t` does not have to be passed), and avoids
inconsistencies between `train_fn` and the scheduler.


### Running the Hyperparameter Optimization

You can use the following schedulers:

- FIFO (`fifo`)
- Hyperband (either the stopping (`hbs`) or promotion (`hbp`) variant)

And the following searchers:

- Random search (`random`)
- Gaussian process based Bayesian optimization (`bayesopt`)
- SkOpt Bayesian optimization (`skopt`; only with FIFO scheduler)

Note that the method known as (asynchronous) Hyperband is using random search.
Combining Hyperband scheduling with the `bayesopt` searcher uses a novel
method called asynchronous BOHB.

Pick the combination you're interested in (doing the full experiment takes around
120 seconds, see the `time_out` parameter), running everything with multiple runs
can take a fair bit of time. In real life, you will want to choose a larger
`time_out` in order to obtain good performance.

```{.python .input  n=37}
SCHEDULER = "hbs"
SEARCHER = "bayesopt"
```

```{.python .input  n=38}
def compute_error(df):
    return 1.0 - df["objective"]

def compute_runtime(df, start_timestamp):
        return df["time_step"] - start_timestamp

def process_training_history(task_dicts, start_timestamp, 
                             runtime_fn=compute_runtime,
                             error_fn=compute_error):
    task_dfs = []
    for task_id in task_dicts:
        task_df = pd.DataFrame(task_dicts[task_id])
        task_df = task_df.assign(task_id=task_id,
                                 runtime=runtime_fn(task_df, start_timestamp),
                                 error=error_fn(task_df),
                                 target_epoch=task_df["epoch"].iloc[-1])
        task_dfs.append(task_df)

    result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)
    # re-order by runtime
    result = result.sort_values(by="runtime")
    # calculate incumbent best -- the cumulative minimum of the error.
    result = result.assign(best=result["error"].cummin())
    return result

resources = dict(num_cpus=NUM_CPUS, num_gpus=0)
```

```{.python .input  n=39}
if SCHEDULER == 'fifo': 
    myscheduler = ag.scheduler.FIFOScheduler(
        run_mlp_openml,
        resource=resources,
        searcher=SEARCHER,
        time_out=120,
        time_attr=RESOURCE_ATTR_NAME,
        reward_attr=REWARD_ATTR_NAME)

else:
    # This setup uses rung levels at 1, 3, 9 epochs. We just use a single
    # bracket, so this is in fact successive halving (Hyperband would use
    # more than 1 bracket).
    # Also note that since we do not use the max_t argument of
    # HyperbandScheduler, this value is obtained from train_fn.args.epochs.
    sch_type = 'stopping' if SCHEDULER == 'hbs' else 'promotion'
    myscheduler = ag.scheduler.HyperbandScheduler(
        run_mlp_openml,
        resource=resources,
        searcher=SEARCHER,
        time_out=120,
        time_attr=RESOURCE_ATTR_NAME,
        reward_attr=REWARD_ATTR_NAME,
        type=sch_type,
        grace_period=1,
        reduction_factor=3,
        brackets=1)

# run tasks
myscheduler.run()
myscheduler.join_jobs()

results_df = process_training_history(
                myscheduler.training_history.copy(),
                start_timestamp=myscheduler._start_time)
```

### Analysing the results

The training history is stored in the `results_df`, the main fields are the
runtime and `'best'` (the objective).

**Note**: You will get slightly different curves for different pairs of scheduler/searcher, the `time_out` here is a bit
too short to really see the difference in a significant way (it would be better
to set it to >1000s). Generally speaking though, hyperband stopping / promotion
+ model will tend to significantly outperform other combinations given enough time.

```{.python .input  n=40}
results_df.head()
```

```{.python .input  n=42}
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

runtime = results_df['runtime'].values
objective = results_df['best'].values

plt.plot(runtime, objective, lw=2)
plt.xticks(fontsize=12)
plt.xlim(0, 120)
plt.ylim(0, 0.5)
plt.yticks(fontsize=12)
plt.xlabel("Runtime [s]", fontsize=14)
plt.ylabel("Objective", fontsize=14)
```

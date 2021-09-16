# Getting started with Advanced HPO Algorithms
:label:`sec_custom_advancedhpo`

This tutorial provides a complete example of how to use AutoGluon's state-of-the-art hyperparameter optimization (HPO) algorithms to tune a basic Multi-Layer Perceptron (MLP) model, which is the most basic type of neural network.

## Loading libraries

```{.python .input  n=1}
# Basic utils for folder manipulations etc
import time
import multiprocessing # to count the number of CPUs available

# External tools to load and process data
import numpy as np
import pandas as pd

# MXNet (NeuralNets)
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

# AutoGluon and HPO tools
import autogluon.core as ag
from autogluon.mxnet.utils import load_and_split_openml_data
```

Check the version of MxNet, you should be fine with version >= 1.5

```{.python .input  n=2}
mx.__version__
```

You can also check the version of AutoGluon and the specific commit and check that it matches what you want.

```{.python .input  n=3}
import autogluon.core.version
ag.version.__version__
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

We will use a multi-way classification task from OpenML. Data preparation
includes:

- Missing values are imputed, using the 'mean' strategy of
  `sklearn.impute.SimpleImputer`
- Split training set into training and validation
- Standardize inputs to mean 0, variance 1

```{.python .input  n=5}
X_train, X_valid, y_train, y_valid, n_classes = load_and_split_openml_data(
    OPENML_TASK_ID, RATIO_TRAIN_VALID, download_from_openml=False)
n_classes
```

The problem has 26 classes.

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
search_options = {
    'num_init_random': 2,
    'debug_log': True}
if SCHEDULER == 'fifo': 
    myscheduler = ag.scheduler.FIFOScheduler(
        run_mlp_openml,
        resource=resources,
        searcher=SEARCHER,
        search_options=search_options,
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
        search_options=search_options,
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

## Diving Deeper

Now, you are ready to try HPO on your own machine learning models (if you use
PyTorch, have a look at :ref:`sec_customstorch`). While AutoGluon comes with
well-chosen defaults, it can pay off to tune it to your specific needs. Here are
some tips which may come useful.

### Logging the Search Progress

First, it is a good idea in general to switch on `debug_log`, which outputs
useful information about the search progress. This is already done in the
example above.

The outputs show which configurations are chosen, stopped, or promoted. For
BO and BOHB, a range of information is displayed for every `get_config`
decision. This log output is very useful in order to figure out what is going
on during the search.

### Configuring `HyperbandScheduler`

The most important knobs to turn with `HyperbandScheduler` are `max_t`, `grace_period`,
`reduction_factor`, `brackets`, and `type`. The first three determine the rung
levels at which stopping or promotion decisions are being made.

- The maximum resource level `max_t` (usually, resource equates to epochs, so
   `max_t` is the maximum number of training epochs) is typically hardcoded in
   `train_fn` passed to the scheduler (this is `run_mlp_openml` in the example
   above). As already noted above, the value is best fixed in the `ag.args`
   decorator as `epochs=XYZ`, it can then be accessed as `args.epochs` in the
   `train_fn` code. If this is done, you do not have to pass `max_t` when creating
   the scheduler.
- `grace_period` and `reduction_factor` determine the rung levels, which are
   `grace_period`, `grace_period * reduction_factor`,
   `grace_period * (reduction_factor ** 2)`, etc. All rung levels must be less or
   equal than `max_t`. It is recommended to make `max_t` equal to the largest rung
   level. For example, if `grace_period = 1`, `reduction_factor = 3`, it is in
   general recommended to use `max_t = 9`, `max_t = 27`, or `max_t = 81`. Choosing
   a `max_t` value "off the grid" works against the successive halving principle
   that the total resources spent in a rung should be roughly equal between rungs. If in the
   example above, you set `max_t = 10`, about a third of configurations reaching
   9 epochs are allowed to proceed, but only for one more epoch.
- With `reduction_factor`, you tune the extent to which successive halving
   filtering is applied. The larger this integer, the fewer configurations make
   it to higher number of epochs. Values 2, 3, 4 are commonly used.
- Finally, `grace_period` should be set to the smallest resource (number of epochs)
   for which you expect any meaningful differentiation between configurations.
   While `grace_period = 1` should always be explored, it may be too low for any
   meaningful stopping decisions to be made at the first rung.
- `brackets` sets the maximum number of brackets in Hyperband (make sure to study
   the Hyperband paper or follow-ups for details). For `brackets = 1`, you are
   running successive halving (single bracket). Higher brackets have larger effective
   `grace_period` values (so runs are not stopped until later), yet are also chosen
   with less probability. We recommend to always consider successive halving
   (`brackets = 1`) in a comparison.
- Finally, with `type` (values `stopping`, `promotion`) you are choosing different
   ways of extending successive halving scheduling to the asynchronous
   case. The method for the default `stopping` is simpler and seems to perform well,
   but `promotion` is more careful promoting configurations to higher resource
   levels, which can work better in some cases.

### Asynchronous BOHB

Finally, here are some ideas for tuning asynchronous BOHB, apart from tuning its
`HyperbandScheduling` component. You need to pass these options in `search_options`.

- We support a range of different surrogate models over the criterion functions
   across resource levels. All of them are jointly dependent Gaussian process
   models, meaning that data collected at all resource levels are modelled
   together. The surrogate model is selected by `gp_resource_kernel`, values are
   `matern52`, `matern52-res-warp`, `exp-decay-sum`, `exp-decay-combined`,
   `exp-decay-delta1`. These are variants of either a joint Matern 5/2 kernel
   over configuration and resource, or the exponential decay model. Details about
   the latter can be found [here](https://arxiv.org/abs/2003.10865).
- Fitting a Gaussian process surrogate model to data encurs a cost which scales
   cubically with the number of datapoints. When applied to expensive deep learning
   workloads, even multi-fidelity asynchronous BOHB is rarely running up more than
   100 observations or so (across all rung levels and brackets), and the GP
   computations are subdominant. However, if you apply it to cheaper `train_fn`
   and find yourself beyond 2000 total evaluations, the cost of GP fitting can
   become painful. In such a situation, you can explore the options `opt_skip_period`
   and `opt_skip_num_max_resource`. The basic idea is as follows. By far the most
   expensive part of a `get_config` call (picking the next configuration) is the
   refitting of the GP model to past data (this entails re-optimizing hyperparameters
   of the surrogate model itself). The options allow you to skip this expensive
   step for most `get_config` calls, after some initial period. Check the docstrings
   for details about these options. If you find yourself in such a situation and
   gain experience with these skipping features, make sure to contact the AutoGluon
   developers -- we would love to learn about your use case.

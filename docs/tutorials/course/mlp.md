# Getting started

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
from autogluon.searcher.bayesopt.autogluon.gp_fifo_searcher import map_reward
from autogluon.scheduler.fifo import FIFOScheduler
from autogluon.scheduler.hyperband import HyperbandScheduler
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/ec2-user/anaconda3/envs/amazonei_mxnet_p36/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB\n  Optimizer.opt_registry[name].__name__))\n"
 }
]
```

Check the version of MxNet, you should be fine with version >= 1.5

```{.python .input  n=2}
mx.__version__
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "'1.6.0'"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can also check the version of AutoGluon and the specific commit and check
that it matches what you want.

```{.python .input  n=3}
ag.__version__
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "'0.0.6+ada1800'"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

Set the openml directory to the current directory (next
to the notebook) and download the task if the file is not already present.

```{.python .input  n=5}
openml.config.set_cache_directory("./")
task = openml.tasks.get_task(OPENML_TASK_ID)
```

```{.json .output n=5}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Data pickle file already exists and is up to date.\n"
 }
]
```

It's a multiclass classification task with 26 classes:

```{.python .input  n=6}
n_classes = len(task.class_labels)
n_classes
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "26"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

OpenML provides a standard train/test split that we can use.

```{.python .input  n=7}
train_indices, test_indices = task.get_train_test_split_indices()

X, y = task.get_X_and_y()

X.shape
```

```{.json .output n=7}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Data pickle file already exists and is up to date.\n"
 },
 {
  "data": {
   "text/plain": "(20000, 16)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Impute any missing value with a basic strategy and recover the training data

```{.python .input  n=8}
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

X_train = X[train_indices]
y_train = y[train_indices]
```

Resplit the training data into training+validation using the specific fraction
indicated earlier

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
- the `@ag.args` decorator allows us to specify
the space we will optimize over, this matches the
[ConfigSpace](https://automl.github.io/ConfigSpace/master/) syntax

The body of
the function `run_mlp_openml` is pretty simple:

- it reads the hyperparameters
given via the decorator
- it defines a 2 layer MLP with dropout
- it declares a
trainer with the 'adam' loss function and a provided learning rate
- it trains
the NN with a number of epochs (most of that is boilerplate code from `mxnet`)
- the `reporter` at the end is used to keep track of training history in the
hyperparameter optimization

**Note**: that the number of epochs and the
hyperparameter space are reduced on purpose to lead to shorter experiments

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

### Running the Hyperparameter Optimization

You can use the following
schedulers:

- FIFO (`fifo`)
- Hyperband (either the stopping (`hbs`) or promotion (`hbp`) variant)

And the following searchers:

- Random search (`random`)
- Gaussian process based Bayesian optimization (`bayesopt`)
- SkOpt Bayesian optimization (`skopt`; only with FIFO scheduler)

Note that the method known as (asynchronous) Hyperband is using random search.
Combining Hyperband scheduling with the `bayesopt` searcher uses a novel
method called asynchronous BOHB.

Pick the combination you're interested in (doing the full experiment takes around 120 seconds, see the
`time_out` parameter), running everything with multiple runs can take a fair bit
of time. In real life, you will want to choose a larger `time_out` in order to
obtain good performance.

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

# default for classification problems: map error to accuracy.
_map_reward = map_reward(const=1.0)
max_metric_value = 1.0
```

```{.python .input  n=39}
run_id = 1

if SCHEDULER == 'fifo': 
    myscheduler = FIFOScheduler(
        run_mlp_openml,
        resource=resources,
        searcher=searcher,
        search_options={'run_id': run_id},
        num_trials=100000, # if this is a very large number, just run until timeout 
        time_out=120,
        time_attr=RESOURCE_ATTR_NAME,
        reward_attr=REWARD_ATTR_NAME)

else:
    # This setup uses rung levels at 1, 3, 9 epochs. We just use a single
    # bracket, so this is in fact successive halving (Hyperband would use
    # more than 1 bracket)
    sch_type = 'stopping' if SCHEDULER == 'hbs' else 'promotion'
    myscheduler = HyperbandScheduler(
        run_mlp_openml,
        resource=resources,
        searcher=searcher,
        search_options={'run_id': run_id, 'min_reward': _map_reward.reverse(1.0)},
        num_trials=100000,
        time_out=120,
        time_attr=RESOURCE_ATTR_NAME,
        reward_attr=REWARD_ATTR_NAME,
        type=sch_type,
        max_t=9,
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

```{.json .output n=39}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "search_options: Key 'random_seed': Imputing default value 31415927\nsearch_options: Key 'opt_skip_init_length': Imputing default value 150\nsearch_options: Key 'opt_skip_period': Imputing default value 1\nsearch_options: Key 'profiler': Imputing default value False\nsearch_options: Key 'opt_maxiter': Imputing default value 50\nsearch_options: Key 'opt_nstarts': Imputing default value 2\nsearch_options: Key 'opt_warmstart': Imputing default value False\nsearch_options: Key 'opt_verbose': Imputing default value False\nsearch_options: Key 'opt_debug_writer': Imputing default value False\nsearch_options: Key 'num_fantasy_samples': Imputing default value 20\nsearch_options: Key 'num_init_random': Imputing default value 50\nsearch_options: Key 'num_init_candidates': Imputing default value 250\nsearch_options: Key 'initial_scoring': Imputing default value thompson_indep\nsearch_options: Key 'first_is_default': Imputing default value True\nsearch_options: Key 'debug_log': Imputing default value False\nsearch_options: Key 'opt_skip_num_max_resource': Imputing default value False\nsearch_options: Key 'gp_resource_kernel': Imputing default value exp-decay-combined\nsearch_options: Key 'resource_acq': Imputing default value bohb\n\nStarting Experiments\nNum of Finished Tasks is 0\nNum of Pending Tasks is 100000\nTime out (secs) is 120\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "da3181cff36640aca3c08c442880b4ff",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "HBox(children=(FloatProgress(value=0.0, max=100000.0), HTML(value='')))"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.473788 ; Training: accuracy=0.262475 ; Validation: accuracy=0.521557\nEpoch 2 ; Time: 0.939769 ; Training: accuracy=0.495043 ; Validation: accuracy=0.645221\nEpoch 3 ; Time: 1.487172 ; Training: accuracy=0.559650 ; Validation: accuracy=0.688837\nEpoch 4 ; Time: 1.980527 ; Training: accuracy=0.585757 ; Validation: accuracy=0.706885\nEpoch 5 ; Time: 2.401407 ; Training: accuracy=0.611368 ; Validation: accuracy=0.729111\nEpoch 6 ; Time: 2.828559 ; Training: accuracy=0.626818 ; Validation: accuracy=0.737968\nEpoch 7 ; Time: 3.244747 ; Training: accuracy=0.641606 ; Validation: accuracy=0.747995\nEpoch 8 ; Time: 3.668414 ; Training: accuracy=0.656808 ; Validation: accuracy=0.764539\nEpoch 9 ; Time: 4.111078 ; Training: accuracy=0.661599 ; Validation: accuracy=0.775067\nEpoch 10 ; Time: 4.531245 ; Training: accuracy=0.672670 ; Validation: accuracy=0.776571\nEpoch 1 ; Time: 0.771677 ; Training: accuracy=0.048418 ; Validation: accuracy=0.109903\nEpoch 1 ; Time: 0.296634 ; Training: accuracy=0.040661 ; Validation: accuracy=0.063802\nEpoch 1 ; Time: 0.351460 ; Training: accuracy=0.373680 ; Validation: accuracy=0.676288\nEpoch 2 ; Time: 0.662637 ; Training: accuracy=0.557013 ; Validation: accuracy=0.736365\nEpoch 3 ; Time: 0.954009 ; Training: accuracy=0.592987 ; Validation: accuracy=0.750629\nEpoch 4 ; Time: 1.287193 ; Training: accuracy=0.626320 ; Validation: accuracy=0.770767\nEpoch 5 ; Time: 1.585697 ; Training: accuracy=0.644307 ; Validation: accuracy=0.784024\nEpoch 6 ; Time: 1.869262 ; Training: accuracy=0.657921 ; Validation: accuracy=0.783521\nEpoch 7 ; Time: 2.149977 ; Training: accuracy=0.667739 ; Validation: accuracy=0.789730\nEpoch 8 ; Time: 2.429280 ; Training: accuracy=0.673680 ; Validation: accuracy=0.808021\nEpoch 9 ; Time: 2.714290 ; Training: accuracy=0.683993 ; Validation: accuracy=0.810371\nEpoch 10 ; Time: 2.998166 ; Training: accuracy=0.685066 ; Validation: accuracy=0.812049\nEpoch 1 ; Time: 0.515733 ; Training: accuracy=0.064484 ; Validation: accuracy=0.057229\nEpoch 1 ; Time: 2.022620 ; Training: accuracy=0.128690 ; Validation: accuracy=0.040404\nEpoch 1 ; Time: 0.673619 ; Training: accuracy=0.099174 ; Validation: accuracy=0.363529\nEpoch 2 ; Time: 1.343158 ; Training: accuracy=0.161405 ; Validation: accuracy=0.487395\nEpoch 3 ; Time: 1.932281 ; Training: accuracy=0.180165 ; Validation: accuracy=0.520000\nEpoch 1 ; Time: 0.447659 ; Training: accuracy=0.037450 ; Validation: accuracy=0.044098\nEpoch 1 ; Time: 0.304534 ; Training: accuracy=0.094919 ; Validation: accuracy=0.176763\nEpoch 1 ; Time: 0.313766 ; Training: accuracy=0.554605 ; Validation: accuracy=0.741190\nEpoch 2 ; Time: 0.623690 ; Training: accuracy=0.766859 ; Validation: accuracy=0.826297\nEpoch 3 ; Time: 0.978639 ; Training: accuracy=0.825905 ; Validation: accuracy=0.860539\nEpoch 4 ; Time: 1.332126 ; Training: accuracy=0.853207 ; Validation: accuracy=0.883810\nEpoch 5 ; Time: 1.613326 ; Training: accuracy=0.875329 ; Validation: accuracy=0.895944\nEpoch 6 ; Time: 1.872197 ; Training: accuracy=0.886924 ; Validation: accuracy=0.900765\nEpoch 7 ; Time: 2.174253 ; Training: accuracy=0.903536 ; Validation: accuracy=0.919382\nEpoch 8 ; Time: 2.461824 ; Training: accuracy=0.911102 ; Validation: accuracy=0.917387\nEpoch 9 ; Time: 2.771994 ; Training: accuracy=0.912089 ; Validation: accuracy=0.923703\nEpoch 10 ; Time: 3.055275 ; Training: accuracy=0.916941 ; Validation: accuracy=0.930020\nEpoch 1 ; Time: 0.795132 ; Training: accuracy=0.081215 ; Validation: accuracy=0.293349\nEpoch 1 ; Time: 0.310508 ; Training: accuracy=0.432822 ; Validation: accuracy=0.686882\nEpoch 2 ; Time: 0.589496 ; Training: accuracy=0.618815 ; Validation: accuracy=0.761434\nEpoch 3 ; Time: 0.856152 ; Training: accuracy=0.665396 ; Validation: accuracy=0.796113\nEpoch 4 ; Time: 1.121233 ; Training: accuracy=0.683050 ; Validation: accuracy=0.801809\nEpoch 5 ; Time: 1.384883 ; Training: accuracy=0.690924 ; Validation: accuracy=0.809181\nEpoch 6 ; Time: 1.648347 ; Training: accuracy=0.704766 ; Validation: accuracy=0.811359\nEpoch 7 ; Time: 1.928160 ; Training: accuracy=0.715209 ; Validation: accuracy=0.826102\nEpoch 8 ; Time: 2.225599 ; Training: accuracy=0.712474 ; Validation: accuracy=0.827609\nEpoch 9 ; Time: 2.523651 ; Training: accuracy=0.723912 ; Validation: accuracy=0.845033\nEpoch 1 ; Time: 0.274276 ; Training: accuracy=0.035629 ; Validation: accuracy=0.029167\nEpoch 1 ; Time: 0.765626 ; Training: accuracy=0.499008 ; Validation: accuracy=0.748485\nEpoch 2 ; Time: 1.523182 ; Training: accuracy=0.663388 ; Validation: accuracy=0.799327\nEpoch 3 ; Time: 2.195962 ; Training: accuracy=0.703802 ; Validation: accuracy=0.834007\nEpoch 4 ; Time: 2.887170 ; Training: accuracy=0.726033 ; Validation: accuracy=0.848485\nEpoch 5 ; Time: 3.544765 ; Training: accuracy=0.741240 ; Validation: accuracy=0.856061\nEpoch 6 ; Time: 4.183477 ; Training: accuracy=0.752479 ; Validation: accuracy=0.868182\nEpoch 7 ; Time: 5.014783 ; Training: accuracy=0.762727 ; Validation: accuracy=0.868855\nEpoch 8 ; Time: 5.735304 ; Training: accuracy=0.773802 ; Validation: accuracy=0.874411\nEpoch 9 ; Time: 6.396016 ; Training: accuracy=0.776198 ; Validation: accuracy=0.869529\nEpoch 10 ; Time: 7.036925 ; Training: accuracy=0.788843 ; Validation: accuracy=0.884512\nEpoch 1 ; Time: 0.576072 ; Training: accuracy=0.050289 ; Validation: accuracy=0.036622\nEpoch 1 ; Time: 1.536913 ; Training: accuracy=0.051410 ; Validation: accuracy=0.082155\nEpoch 1 ; Time: 0.621160 ; Training: accuracy=0.034012 ; Validation: accuracy=0.036424\nEpoch 1 ; Time: 0.395048 ; Training: accuracy=0.079219 ; Validation: accuracy=0.439780\nEpoch 2 ; Time: 0.786243 ; Training: accuracy=0.144546 ; Validation: accuracy=0.502082\nEpoch 3 ; Time: 1.172079 ; Training: accuracy=0.184570 ; Validation: accuracy=0.541896\nEpoch 1 ; Time: 0.374218 ; Training: accuracy=0.031209 ; Validation: accuracy=0.020234\nEpoch 1 ; Time: 0.299380 ; Training: accuracy=0.316190 ; Validation: accuracy=0.644314\nEpoch 2 ; Time: 0.589459 ; Training: accuracy=0.493913 ; Validation: accuracy=0.700167\nEpoch 3 ; Time: 0.849621 ; Training: accuracy=0.538799 ; Validation: accuracy=0.734114\nEpoch 1 ; Time: 2.387311 ; Training: accuracy=0.109251 ; Validation: accuracy=0.073557\nEpoch 1 ; Time: 0.382923 ; Training: accuracy=0.045323 ; Validation: accuracy=0.042876\nEpoch 1 ; Time: 0.838808 ; Training: accuracy=0.557025 ; Validation: accuracy=0.694949\nEpoch 2 ; Time: 1.538461 ; Training: accuracy=0.623719 ; Validation: accuracy=0.724242\nEpoch 3 ; Time: 2.292300 ; Training: accuracy=0.665041 ; Validation: accuracy=0.769865\nEpoch 1 ; Time: 0.496638 ; Training: accuracy=0.323991 ; Validation: accuracy=0.602823\nEpoch 2 ; Time: 1.079805 ; Training: accuracy=0.418651 ; Validation: accuracy=0.658938\nEpoch 3 ; Time: 1.529149 ; Training: accuracy=0.437665 ; Validation: accuracy=0.651714\nEpoch 1 ; Time: 0.407859 ; Training: accuracy=0.046248 ; Validation: accuracy=0.124750\nEpoch 1 ; Time: 0.323726 ; Training: accuracy=0.046756 ; Validation: accuracy=0.074313\nEpoch 1 ; Time: 0.569760 ; Training: accuracy=0.036045 ; Validation: accuracy=0.047643\nEpoch 1 ; Time: 0.377967 ; Training: accuracy=0.043134 ; Validation: accuracy=0.037413\nEpoch 1 ; Time: 0.360675 ; Training: accuracy=0.109400 ; Validation: accuracy=0.283041\nEpoch 1 ; Time: 0.551281 ; Training: accuracy=0.029859 ; Validation: accuracy=0.014785\nEpoch 1 ; Time: 0.308442 ; Training: accuracy=0.068294 ; Validation: accuracy=0.215184\nEpoch 1 ; Time: 0.373837 ; Training: accuracy=0.238792 ; Validation: accuracy=0.495632\nEpoch 2 ; Time: 0.738909 ; Training: accuracy=0.303143 ; Validation: accuracy=0.490423\nEpoch 3 ; Time: 1.097138 ; Training: accuracy=0.316625 ; Validation: accuracy=0.543179\nEpoch 1 ; Time: 0.581070 ; Training: accuracy=0.048657 ; Validation: accuracy=0.079344\nEpoch 1 ; Time: 0.338643 ; Training: accuracy=0.188911 ; Validation: accuracy=0.473081\nEpoch 2 ; Time: 0.667269 ; Training: accuracy=0.296836 ; Validation: accuracy=0.526255\nEpoch 3 ; Time: 0.989874 ; Training: accuracy=0.335970 ; Validation: accuracy=0.599369\nEpoch 1 ; Time: 0.546419 ; Training: accuracy=0.210631 ; Validation: accuracy=0.419371\nEpoch 2 ; Time: 1.097463 ; Training: accuracy=0.291356 ; Validation: accuracy=0.485948\nEpoch 3 ; Time: 1.634495 ; Training: accuracy=0.310482 ; Validation: accuracy=0.506858\nEpoch 1 ; Time: 0.404046 ; Training: accuracy=0.086400 ; Validation: accuracy=0.110944\nEpoch 1 ; Time: 0.414198 ; Training: accuracy=0.063847 ; Validation: accuracy=0.058081\nEpoch 1 ; Time: 0.806587 ; Training: accuracy=0.160314 ; Validation: accuracy=0.445248\nEpoch 2 ; Time: 1.544369 ; Training: accuracy=0.253080 ; Validation: accuracy=0.528680\nEpoch 3 ; Time: 2.369774 ; Training: accuracy=0.281604 ; Validation: accuracy=0.539781\nEpoch 1 ; Time: 0.712962 ; Training: accuracy=0.082891 ; Validation: accuracy=0.216221\nEpoch 1 ; Time: 0.730931 ; Training: accuracy=0.125413 ; Validation: accuracy=0.129200\nEpoch 1 ; Time: 1.711748 ; Training: accuracy=0.063184 ; Validation: accuracy=0.172896\nEpoch 1 ; Time: 1.068355 ; Training: accuracy=0.364887 ; Validation: accuracy=0.658537\nEpoch 2 ; Time: 2.149796 ; Training: accuracy=0.476625 ; Validation: accuracy=0.701430\nEpoch 3 ; Time: 3.158840 ; Training: accuracy=0.506963 ; Validation: accuracy=0.732044\nEpoch 1 ; Time: 2.254991 ; Training: accuracy=0.278929 ; Validation: accuracy=0.562836\nEpoch 2 ; Time: 4.238326 ; Training: accuracy=0.350962 ; Validation: accuracy=0.557796\nEpoch 3 ; Time: 6.035529 ; Training: accuracy=0.362069 ; Validation: accuracy=0.601815\nEpoch 1 ; Time: 0.359431 ; Training: accuracy=0.120242 ; Validation: accuracy=0.175770\nEpoch 1 ; Time: 0.330552 ; Training: accuracy=0.081732 ; Validation: accuracy=0.271000\nEpoch 1 ; Time: 0.302100 ; Training: accuracy=0.055248 ; Validation: accuracy=0.070761\nEpoch 1 ; Time: 0.540344 ; Training: accuracy=0.088731 ; Validation: accuracy=0.243817\nEpoch 1 ; Time: 0.309989 ; Training: accuracy=0.066244 ; Validation: accuracy=0.150706\nEpoch 1 ; Time: 0.386215 ; Training: accuracy=0.066672 ; Validation: accuracy=0.123931\nEpoch 1 ; Time: 0.453650 ; Training: accuracy=0.039669 ; Validation: accuracy=0.054167\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\n/home/ec2-user/anaconda3/envs/amazonei_mxnet_p36/lib/python3.6/site-packages/mxnet/gluon/block.py:693: UserWarning: Parameter squareddistance1_inverse_bandwidths_internal, matern521_covariance_scale_internal is not used by any computation. Is this intended?\n  out = self.forward(*args)\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.13817809 0.14072741 0.13892456 0.1380272  0.13749244 0.13925482\n 0.13816378 0.13839544 0.13987145 0.14103291 0.13998642 0.13836002\n 0.13812918 0.13948766 0.1389415  0.13987647 0.1389387  0.13950357\n 0.13799861 0.13827852]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.277591 ; Training: accuracy=0.547862 ; Validation: accuracy=0.765625\nEpoch 2 ; Time: 0.575418 ; Training: accuracy=0.763816 ; Validation: accuracy=0.815492\nEpoch 3 ; Time: 0.865744 ; Training: accuracy=0.804605 ; Validation: accuracy=0.865193\nEpoch 4 ; Time: 1.159667 ; Training: accuracy=0.837171 ; Validation: accuracy=0.882480\nEpoch 5 ; Time: 1.455425 ; Training: accuracy=0.849095 ; Validation: accuracy=0.894614\nEpoch 6 ; Time: 1.749769 ; Training: accuracy=0.861924 ; Validation: accuracy=0.906084\nEpoch 7 ; Time: 2.009121 ; Training: accuracy=0.873684 ; Validation: accuracy=0.903757\nEpoch 8 ; Time: 2.264214 ; Training: accuracy=0.883717 ; Validation: accuracy=0.911569\nEpoch 9 ; Time: 2.521037 ; Training: accuracy=0.875987 ; Validation: accuracy=0.916556\nEpoch 10 ; Time: 2.795248 ; Training: accuracy=0.890296 ; Validation: accuracy=0.924036\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.13473146 0.13446251 0.13588879 0.13518463 0.13375406 0.13546046\n 0.13446675 0.1345447  0.13555251 0.1360179  0.13462503 0.134403\n 0.13514663 0.13492877 0.13429784 0.13423682 0.13522282 0.13536276\n 0.13486206 0.13486266]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.836366 ; Training: accuracy=0.676427 ; Validation: accuracy=0.804592\nEpoch 2 ; Time: 1.668271 ; Training: accuracy=0.822581 ; Validation: accuracy=0.846992\nEpoch 3 ; Time: 2.546763 ; Training: accuracy=0.865426 ; Validation: accuracy=0.869113\nEpoch 4 ; Time: 3.373074 ; Training: accuracy=0.884615 ; Validation: accuracy=0.885537\nEpoch 5 ; Time: 4.077272 ; Training: accuracy=0.896030 ; Validation: accuracy=0.899615\nEpoch 6 ; Time: 4.781242 ; Training: accuracy=0.902068 ; Validation: accuracy=0.888721\nEpoch 7 ; Time: 5.492102 ; Training: accuracy=0.916460 ; Validation: accuracy=0.902631\nEpoch 8 ; Time: 6.214981 ; Training: accuracy=0.921340 ; Validation: accuracy=0.901961\nEpoch 9 ; Time: 7.132312 ; Training: accuracy=0.924979 ; Validation: accuracy=0.913357\nEpoch 10 ; Time: 7.858468 ; Training: accuracy=0.930356 ; Validation: accuracy=0.911178\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.12846103 0.1304712  0.12962275 0.1283974  0.12853788 0.12816476\n 0.12930831 0.12861891 0.12861271 0.12996122 0.12847547 0.12810839\n 0.12911253 0.12954914 0.12936217 0.12962424 0.13058383 0.12940144\n 0.12902549 0.12848941]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.855603 ; Training: accuracy=0.437490 ; Validation: accuracy=0.608360\nEpoch 2 ; Time: 1.603509 ; Training: accuracy=0.561101 ; Validation: accuracy=0.721001\nEpoch 3 ; Time: 2.421591 ; Training: accuracy=0.597330 ; Validation: accuracy=0.738123\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.13016414 0.13073327 0.12934023 0.12924267 0.13033697 0.13149308\n 0.12868967 0.13025803 0.1299375  0.12988957 0.12944828 0.13019231\n 0.13034644 0.12980851 0.13012756 0.12925639 0.12960548 0.12967547\n 0.12941801 0.13062147]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.312815 ; Training: accuracy=0.477303 ; Validation: accuracy=0.699302\nEpoch 2 ; Time: 0.580613 ; Training: accuracy=0.739145 ; Validation: accuracy=0.800366\nEpoch 3 ; Time: 0.898721 ; Training: accuracy=0.815132 ; Validation: accuracy=0.848238\nEpoch 4 ; Time: 1.217371 ; Training: accuracy=0.851562 ; Validation: accuracy=0.868351\nEpoch 5 ; Time: 1.534104 ; Training: accuracy=0.876069 ; Validation: accuracy=0.904422\nEpoch 6 ; Time: 1.825423 ; Training: accuracy=0.895806 ; Validation: accuracy=0.908245\nEpoch 7 ; Time: 2.124547 ; Training: accuracy=0.906908 ; Validation: accuracy=0.915891\nEpoch 8 ; Time: 2.442876 ; Training: accuracy=0.917845 ; Validation: accuracy=0.928856\nEpoch 9 ; Time: 2.760624 ; Training: accuracy=0.928125 ; Validation: accuracy=0.929854\nEpoch 10 ; Time: 3.128109 ; Training: accuracy=0.933141 ; Validation: accuracy=0.926695\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.07298112 0.07425089 0.07500078 0.07358345 0.07263881 0.07431134\n 0.07466144 0.07358698 0.07425579 0.07377217 0.07412609 0.07410824\n 0.07317174 0.07318022 0.07426919 0.07630542 0.07455934 0.07411502\n 0.07383478 0.07420838]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.352433 ; Training: accuracy=0.580069 ; Validation: accuracy=0.747073\nEpoch 2 ; Time: 0.696394 ; Training: accuracy=0.789664 ; Validation: accuracy=0.831047\nEpoch 3 ; Time: 1.214240 ; Training: accuracy=0.848831 ; Validation: accuracy=0.861994\nEpoch 4 ; Time: 1.566348 ; Training: accuracy=0.873025 ; Validation: accuracy=0.893944\nEpoch 5 ; Time: 1.917080 ; Training: accuracy=0.895902 ; Validation: accuracy=0.903145\nEpoch 6 ; Time: 2.276095 ; Training: accuracy=0.910221 ; Validation: accuracy=0.912847\nEpoch 7 ; Time: 2.637408 ; Training: accuracy=0.920260 ; Validation: accuracy=0.929241\nEpoch 8 ; Time: 3.006805 ; Training: accuracy=0.931205 ; Validation: accuracy=0.925226\nEpoch 9 ; Time: 3.375654 ; Training: accuracy=0.933427 ; Validation: accuracy=0.932586\nEpoch 10 ; Time: 3.746604 ; Training: accuracy=0.941820 ; Validation: accuracy=0.935764\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.06474885 0.06348388 0.06368626 0.0653221  0.0629716  0.06334304\n 0.06459517 0.06381017 0.06282113 0.06157791 0.06145324 0.06346661\n 0.06218249 0.06205753 0.06437655 0.06387353 0.06272944 0.06684741\n 0.06440607 0.06586005]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.284025 ; Training: accuracy=0.531908 ; Validation: accuracy=0.748836\nEpoch 2 ; Time: 0.566291 ; Training: accuracy=0.772862 ; Validation: accuracy=0.828624\nEpoch 3 ; Time: 0.858271 ; Training: accuracy=0.827714 ; Validation: accuracy=0.871676\nEpoch 4 ; Time: 1.126444 ; Training: accuracy=0.859786 ; Validation: accuracy=0.870678\nEpoch 5 ; Time: 1.388206 ; Training: accuracy=0.873766 ; Validation: accuracy=0.897440\nEpoch 6 ; Time: 1.670953 ; Training: accuracy=0.890049 ; Validation: accuracy=0.900598\nEpoch 7 ; Time: 1.925258 ; Training: accuracy=0.903207 ; Validation: accuracy=0.913398\nEpoch 8 ; Time: 2.237822 ; Training: accuracy=0.905839 ; Validation: accuracy=0.914561\nEpoch 9 ; Time: 2.572012 ; Training: accuracy=0.911349 ; Validation: accuracy=0.920711\nEpoch 10 ; Time: 2.902591 ; Training: accuracy=0.918421 ; Validation: accuracy=0.927527\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.06432889 0.06295608 0.06369022 0.06279422 0.06472707 0.06221455\n 0.06246181 0.06282201 0.06346137 0.06102108 0.06307278 0.06413775\n 0.06174162 0.0646587  0.06365846 0.0609213  0.06309732 0.06406106\n 0.06374773 0.06365681]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 1.091834 ; Training: accuracy=0.581399 ; Validation: accuracy=0.760807\nEpoch 2 ; Time: 2.056272 ; Training: accuracy=0.772049 ; Validation: accuracy=0.831119\nEpoch 3 ; Time: 3.080773 ; Training: accuracy=0.830405 ; Validation: accuracy=0.864424\nEpoch 4 ; Time: 4.120449 ; Training: accuracy=0.861572 ; Validation: accuracy=0.883768\nEpoch 5 ; Time: 5.541479 ; Training: accuracy=0.885444 ; Validation: accuracy=0.900252\nEpoch 6 ; Time: 7.009883 ; Training: accuracy=0.899287 ; Validation: accuracy=0.911522\nEpoch 7 ; Time: 8.476332 ; Training: accuracy=0.909068 ; Validation: accuracy=0.916232\nEpoch 8 ; Time: 9.942807 ; Training: accuracy=0.921336 ; Validation: accuracy=0.926829\nEpoch 9 ; Time: 11.412037 ; Training: accuracy=0.925895 ; Validation: accuracy=0.923129\nEpoch 10 ; Time: 12.706029 ; Training: accuracy=0.934267 ; Validation: accuracy=0.933894\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.06335078 0.06341331 0.06397044 0.06534865 0.06343931 0.0666396\n 0.06298489 0.06687102 0.06321891 0.06323473 0.0617037  0.0635215\n 0.06338422 0.06268658 0.06187025 0.06143267 0.06364839 0.0628295\n 0.06281746 0.06345171]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 1 ; Time: 0.374060 ; Training: accuracy=0.514489 ; Validation: accuracy=0.711279\nEpoch 2 ; Time: 0.748780 ; Training: accuracy=0.725782 ; Validation: accuracy=0.800673\nEpoch 3 ; Time: 1.079326 ; Training: accuracy=0.783325 ; Validation: accuracy=0.832492\nEpoch 4 ; Time: 1.414898 ; Training: accuracy=0.817437 ; Validation: accuracy=0.858923\nEpoch 5 ; Time: 1.777766 ; Training: accuracy=0.844179 ; Validation: accuracy=0.873064\nEpoch 6 ; Time: 2.113083 ; Training: accuracy=0.860242 ; Validation: accuracy=0.889899\nEpoch 7 ; Time: 2.429302 ; Training: accuracy=0.872992 ; Validation: accuracy=0.902020\nEpoch 8 ; Time: 2.739784 ; Training: accuracy=0.882348 ; Validation: accuracy=0.905556\nEpoch 9 ; Time: 3.095244 ; Training: accuracy=0.893277 ; Validation: accuracy=0.914141\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Fitting GP model\nRecomputing GP state\nBO Algorithm: Generating initial candidates.\nBO Algorithm: Scoring (and reordering) candidates.\nBO Algorithm: Selecting final set of candidates.\nCurrent best is [0.06393536 0.06531753 0.06248624 0.0647882  0.06508441 0.06445603\n 0.06215161 0.06405562 0.06282332 0.0647914  0.06376733 0.06331669\n 0.06356875 0.06481041 0.06306803 0.06339224 0.06444235 0.06488513\n 0.0656417  0.06130696]\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\nEpoch 1 ; Time: 0.306028 ; Training: accuracy=0.593421 ; Validation: accuracy=0.786735\nEpoch 2 ; Time: 0.659197 ; Training: accuracy=0.767352 ; Validation: accuracy=0.841090\nEpoch 3 ; Time: 1.007812 ; Training: accuracy=0.817845 ; Validation: accuracy=0.865691\nEpoch 4 ; Time: 1.355434 ; Training: accuracy=0.848602 ; Validation: accuracy=0.890126\nEpoch 5 ; Time: 1.698581 ; Training: accuracy=0.861020 ; Validation: accuracy=0.892952\nEpoch 6 ; Time: 1.948472 ; Training: accuracy=0.877220 ; Validation: accuracy=0.909741\nEpoch 7 ; Time: 2.235467 ; Training: accuracy=0.886842 ; Validation: accuracy=0.912566\nEpoch 8 ; Time: 2.489419 ; Training: accuracy=0.893010 ; Validation: accuracy=0.918218\nEpoch 9 ; Time: 2.782531 ; Training: accuracy=0.899260 ; Validation: accuracy=0.920878\n"
 }
]
```

### Analysing the results

The training history is stored in the `results_df`, the main fields are the
runtime and `'best'` (the objective).

**Note**: you will get slightly different
curves for different pairs of scheduler/searcher, the `time_out` here is a bit
too short to really see the difference in a significant way (it would be better
to set it to >1000s). Generally speaking though, hyperband stopping / promotion
+ model will tend to significantly outperform other combinations given enough
time.

```{.python .input  n=40}
results_df.head()
```

```{.json .output n=40}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>bracket</th>\n      <th>elapsed_time</th>\n      <th>epoch</th>\n      <th>error</th>\n      <th>eval_time</th>\n      <th>objective</th>\n      <th>runtime</th>\n      <th>searcher_data_size</th>\n      <th>searcher_params_kernel_alpha</th>\n      <th>searcher_params_kernel_delta</th>\n      <th>...</th>\n      <th>searcher_params_kernel_kernelx_inv_bw7</th>\n      <th>searcher_params_kernel_mean_lam</th>\n      <th>searcher_params_kernel_meanx_mean_value</th>\n      <th>searcher_params_noise_variance</th>\n      <th>target_epoch</th>\n      <th>task_id</th>\n      <th>time_since_start</th>\n      <th>time_step</th>\n      <th>time_this_iter</th>\n      <th>best</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>0.477525</td>\n      <td>1</td>\n      <td>0.478443</td>\n      <td>0.472982</td>\n      <td>0.521557</td>\n      <td>0.566952</td>\n      <td>NaN</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>...</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>0.0</td>\n      <td>0.001</td>\n      <td>10</td>\n      <td>13</td>\n      <td>0.569354</td>\n      <td>1.588863e+09</td>\n      <td>0.507336</td>\n      <td>0.478443</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>0</td>\n      <td>0.940767</td>\n      <td>2</td>\n      <td>0.354779</td>\n      <td>0.457808</td>\n      <td>0.645221</td>\n      <td>1.030194</td>\n      <td>1.0</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>...</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>0.0</td>\n      <td>0.001</td>\n      <td>10</td>\n      <td>13</td>\n      <td>1.032070</td>\n      <td>1.588863e+09</td>\n      <td>0.463201</td>\n      <td>0.354779</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>0</td>\n      <td>1.488021</td>\n      <td>3</td>\n      <td>0.311163</td>\n      <td>0.543604</td>\n      <td>0.688837</td>\n      <td>1.577448</td>\n      <td>1.0</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>...</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>0.0</td>\n      <td>0.001</td>\n      <td>10</td>\n      <td>13</td>\n      <td>1.578880</td>\n      <td>1.588863e+09</td>\n      <td>0.547263</td>\n      <td>0.311163</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>0</td>\n      <td>1.981300</td>\n      <td>4</td>\n      <td>0.293115</td>\n      <td>0.489309</td>\n      <td>0.706885</td>\n      <td>2.070727</td>\n      <td>2.0</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>...</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>0.0</td>\n      <td>0.001</td>\n      <td>10</td>\n      <td>13</td>\n      <td>2.072334</td>\n      <td>1.588863e+09</td>\n      <td>0.493268</td>\n      <td>0.293115</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>0</td>\n      <td>2.402163</td>\n      <td>5</td>\n      <td>0.270889</td>\n      <td>0.417534</td>\n      <td>0.729111</td>\n      <td>2.491590</td>\n      <td>2.0</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>...</td>\n      <td>1.0</td>\n      <td>0.5</td>\n      <td>0.0</td>\n      <td>0.001</td>\n      <td>10</td>\n      <td>13</td>\n      <td>2.492918</td>\n      <td>1.588863e+09</td>\n      <td>0.420863</td>\n      <td>0.270889</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 29 columns</p>\n</div>",
   "text/plain": "   bracket  elapsed_time  epoch     error  eval_time  objective   runtime  \\\n0        0      0.477525      1  0.478443   0.472982   0.521557  0.566952   \n1        0      0.940767      2  0.354779   0.457808   0.645221  1.030194   \n2        0      1.488021      3  0.311163   0.543604   0.688837  1.577448   \n3        0      1.981300      4  0.293115   0.489309   0.706885  2.070727   \n4        0      2.402163      5  0.270889   0.417534   0.729111  2.491590   \n\n   searcher_data_size  searcher_params_kernel_alpha  \\\n0                 NaN                           1.0   \n1                 1.0                           1.0   \n2                 1.0                           1.0   \n3                 2.0                           1.0   \n4                 2.0                           1.0   \n\n   searcher_params_kernel_delta  ...  searcher_params_kernel_kernelx_inv_bw7  \\\n0                           0.5  ...                                     1.0   \n1                           0.5  ...                                     1.0   \n2                           0.5  ...                                     1.0   \n3                           0.5  ...                                     1.0   \n4                           0.5  ...                                     1.0   \n\n   searcher_params_kernel_mean_lam  searcher_params_kernel_meanx_mean_value  \\\n0                              0.5                                      0.0   \n1                              0.5                                      0.0   \n2                              0.5                                      0.0   \n3                              0.5                                      0.0   \n4                              0.5                                      0.0   \n\n   searcher_params_noise_variance  target_epoch  task_id  time_since_start  \\\n0                           0.001            10       13          0.569354   \n1                           0.001            10       13          1.032070   \n2                           0.001            10       13          1.578880   \n3                           0.001            10       13          2.072334   \n4                           0.001            10       13          2.492918   \n\n      time_step  time_this_iter      best  \n0  1.588863e+09        0.507336  0.478443  \n1  1.588863e+09        0.463201  0.354779  \n2  1.588863e+09        0.547263  0.311163  \n3  1.588863e+09        0.493268  0.293115  \n4  1.588863e+09        0.420863  0.270889  \n\n[5 rows x 29 columns]"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.json .output n=42}
[
 {
  "data": {
   "text/plain": "Text(0, 0.5, 'Objective')"
  },
  "execution_count": 42,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 864x576 with 1 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```
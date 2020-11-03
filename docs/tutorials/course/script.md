# Tune Training Scripts
:label:`sec_customscript`

This tutorial demonstrates how to do hyperparameter optimization (HPO) to find optimal argument values of custom 
Python scripts. AutoGluon is a framework-agnostic HPO toolkit, which is compatible with any training code written in Python. See also :ref:`sec_customstorch`.

## Neural Network Fine-tuning Example

Import required packages, such as numpy, mxnet and gluoncv:

```{.python .input}
import os
import numpy as np

import mxnet as mx
from mxnet import gluon, init
from autogluon.mxnet.task.nets import get_built_in_network
```

Define a function for dataset meta data:

```{.python .input}
def get_dataset_meta(dataset, basedir='./datasets'):
    if dataset.lower() == 'apparel':
        num_classes = 18
        rec_train = os.path.join(basedir, 'Apparel_train.rec')
        rec_train_idx = os.path.join(basedir, 'Apparel_train.idx')
        rec_val = os.path.join(basedir, 'Apparel_test.rec')
        rec_val_idx = os.path.join(basedir, 'Apparel_test.idx')
    else:
        raise NotImplemented
    return num_classes, rec_train, rec_train_idx, rec_val, rec_val_idx
```

Define the test/evaluation function:


```{.python .input}
def test(net, val_data, ctx, batch_fn):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()
```

Define the training loop. This is a transfer learning script adapted from gluoncv tutorial:


```{.python .input}
def train_loop(args, reporter):
    lr_steps = [int(args.epochs*0.75), np.inf]
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]

    num_classes, rec_train, rec_train_idx, rec_val, rec_val_idx = get_dataset_meta(args.dataset)
    net = get_built_in_network(args.net, num_classes, ctx)

    train_data, val_data, batch_fn = get_data_rec(
            args.input_size, args.crop_ratio, rec_train, rec_train_idx,
            rec_val, rec_val_idx, args.batch_size, args.num_workers,
            args.jitter_param, args.max_rotate_angle)

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
                            'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    lr_counter = 0
    for epoch in range(args.epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*args.lr_factor)
            lr_counter += 1

        train_data.reset()
        metric.reset()
        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)
            with mx.autograd.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(args.batch_size)
            metric.update(label, outputs)

        _, train_acc = metric.get()
        _, val_acc = test(net, val_data, ctx, batch_fn)

        if reporter is not None:
            # reporter enables communications with autogluon
            reporter(epoch=epoch+1, accuracy=val_acc)
        else:
            print('[Epoch %d] Train-acc: %.3f | Val-acc: %.3f' %
                  (epoch, train_acc, val_acc))
```

### How to Do HPO Using AutoGluon on any Training Function

```{.python .input}
import autogluon.core as ag
from autogluon.mxnet.utils import get_data_rec

@ag.args(
    dataset='apparel',
    net='resnet18_v1b',
    epochs=ag.Choice(40, 80),
    lr=ag.Real(1e-4, 1e-2, log=True),
    lr_factor=ag.Real(0.1, 1, log=True),
    batch_size=256,
    momentum=0.9,
    wd=ag.Real(1e-5, 1e-3, log=True),
    num_gpus=8,
    num_workers=30,
    input_size=ag.Choice(224, 256),
    crop_ratio=0.875,
    jitter_param=ag.Real(0.1, 0.4),
    max_rotate_angle=ag.space.Int(0, 10),
)
def train_finetune(args, reporter):
    return train_loop(args, reporter)
```

### Create the Scheduler and Launch the Experiment

```{.python .input}
myscheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                         resource={'num_cpus': 16, 'num_gpus': 8},
                                         num_trials=5,
                                         time_attr='epoch',
                                         reward_attr="accuracy")
print(myscheduler)

```

```{.python .input}
# myscheduler.run()
# myscheduler.join_jobs()
```

Plot the results.

```{.python .input}
# myscheduler.get_training_curves(plot=True,use_legend=False)
# print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
#                                                                myscheduler.get_best_reward()))
```

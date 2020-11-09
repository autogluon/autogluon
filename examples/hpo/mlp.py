# Demonstrates hyperparameter optimization (HPO), with FIFO or
# multi-fidelity (successive halving) scheduler, and random or model based
# searcher. The problem is classification on a tabular OpenML dataset, based
# on a multi-layer perceptron with two hidden layers. HPO is running over
# 8 hyperparameters.
# The MLP model is written in MXNet.

import time
import multiprocessing # to count the number of CPUs available
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
import autogluon.core as ag
import argparse
import logging

from autogluon.mxnet.utils import load_and_split_openml_data

logger = logging.getLogger(__name__)


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Model-based Asynchronous HPO')

    parser.add_argument('--debug', action='store_true',
                        help='debug if needed')
    parser.add_argument('--epochs', type=int, default=9,
                        help='number of epochs')
    parser.add_argument('--scheduler', type=str, default='fifo',
                        choices=['fifo', 'hyperband_stopping', 'hyperband_promotion'],
                        help='Scheduler name (default: fifo)')
    parser.add_argument('--random_seed', type=int, default=31415927,
                        help='random seed')
    # Note: 'model' == 'bayesopt' (legacy)
    parser.add_argument('--searcher', type=str, default='random',
                        choices=['random', 'model', 'bayesopt'],
                        help='searcher name (default: random)')
    # Arguments for FIFOScheduler
    parser.add_argument('--num_trials', type=int,
                        help='number of trial tasks')
    parser.add_argument('--scheduler_timeout', type=float, default=120,
                        help='maximum time until trials are started')
    # Arguments for HyperbandScheduler
    parser.add_argument('--brackets', type=int,
                        help='number of brackets')
    parser.add_argument('--reduction_factor', type=int,
                        help='Reduction factor for successive halving')
    parser.add_argument('--grace_period', type=int,
                        help='minimum number of epochs to run with'
                             'hyperband_* scheduler')
    parser.add_argument('--use_single_rung_system', action='store_true',
                        help='Use single rung level system for all brackets')
    args = parser.parse_args()
    return args


def _enter_not_none(dct, key, val):
    if val is not None:
        dct[key] = val


def from_argparse(args) -> (dict, dict):
    """
    Given result from ArgumentParser.parse_args(), create both search_options
    and scheduler_options.

    :param args: See above
    :return: search_options, scheduler_options

    """
    # Options for searcher
    search_options = dict()
    _enter_not_none(search_options, 'debug_log', True)
    _enter_not_none(search_options, 'random_seed', args.random_seed)

    # Options for scheduler
    scheduler_options = dict()
    _enter_not_none(scheduler_options, 'num_trials', args.num_trials)
    _enter_not_none(scheduler_options, 'time_out', args.scheduler_timeout)
    if args.scheduler != 'fifo':
        if args.scheduler == 'hyperband_stopping':
            sch_type = 'stopping'
        else:
            sch_type = 'promotion'
        _enter_not_none(scheduler_options, 'type', sch_type)
        _enter_not_none(scheduler_options, 'reduction_factor',
                        args.reduction_factor)
        _enter_not_none(scheduler_options, 'max_t', args.epochs)
        _enter_not_none(scheduler_options, 'grace_period', args.grace_period)
        _enter_not_none(scheduler_options, 'brackets', args.brackets)
        _enter_not_none(scheduler_options, 'rung_system_per_bracket',
                        not args.use_single_rung_system)
        _enter_not_none(scheduler_options, 'random_seed', args.random_seed)

    return search_options, scheduler_options


OPENML_TASK_ID = 6                # describes the problem we will tackle
RATIO_TRAIN_VALID = 0.33          # split of the training data used for validation
RESOURCE_ATTR_NAME = 'epoch'      # how do we measure resources   (will become clearer further)
REWARD_ATTR_NAME = 'objective'    # how do we measure performance (will become clearer further)


def create_train_fn(X_train, X_valid, y_train, y_valid, n_classes, epochs):
    @ag.args(n_units_1=ag.space.Int(lower=16, upper=128),
             n_units_2=ag.space.Int(lower=16, upper=128),
             dropout_1=ag.space.Real(lower=0, upper=.75),
             dropout_2=ag.space.Real(lower=0, upper=.75),
             learning_rate=ag.space.Real(lower=1e-6, upper=1, log=True),
             batch_size=ag.space.Int(lower=8, upper=128),
             scale_1=ag.space.Real(lower=0.001, upper=10, log=True),
             scale_2=ag.space.Real(lower=0.001, upper=10, log=True),
             epochs=epochs)
    def run_mlp_openml(args, reporter):
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

    return run_mlp_openml


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Each job uses all available CPUs:
    num_cpus = multiprocessing.cpu_count()
    resources = dict(num_cpus=num_cpus, num_gpus=0)
    # Load data and create evaluation function
    X_train, X_valid, y_train, y_valid, n_classes = load_and_split_openml_data(
        OPENML_TASK_ID, RATIO_TRAIN_VALID, download_from_openml=False)

    # Searcher and scheduler options from args
    search_options, scheduler_options = from_argparse(args)
    scheduler_options['reward_attr'] = REWARD_ATTR_NAME
    scheduler_options['time_attr'] = RESOURCE_ATTR_NAME
    # Define search method
    if args.searcher == 'random':
        searcher_name = 'random'
    else:
        searcher_name = 'bayesopt'
    # Evaluation function:
    run_mlp_openml = create_train_fn(
        X_train, X_valid, y_train, y_valid, n_classes, epochs=args.epochs)

    # Build scheduler and searcher
    scheduler_cls = ag.scheduler.FIFOScheduler if args.scheduler == 'fifo' \
        else ag.scheduler.HyperbandScheduler
    myscheduler = scheduler_cls(
        run_mlp_openml,
        resource=resources,
        searcher=searcher_name,
        search_options=search_options,
        **scheduler_options)

    # Run experiment
    myscheduler.run()
    myscheduler.join_jobs()

    logger.info("Finished joining all tasks!")

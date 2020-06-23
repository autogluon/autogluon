import time
import multiprocessing # to count the number of CPUs available
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
import autogluon as ag
import pytest

from autogluon.utils import load_and_split_openml_data


OPENML_TASK_ID = 6                # describes the problem we will tackle
RATIO_TRAIN_VALID = 0.33          # split of the training data used for validation
RESOURCE_ATTR_NAME = 'epoch'      # how do we measure resources   (will become clearer further)
REWARD_ATTR_NAME = 'objective'    # how do we measure performance (will become clearer further)


def create_train_fn(X_train, X_valid, y_train, y_valid, n_classes, epochs=9):
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


def run_bayesopt_test(sch_type):
    # Each job uses all available CPUs:
    num_cpus = multiprocessing.cpu_count()
    resources = dict(num_cpus=num_cpus, num_gpus=0)
    # Load data and create evaluation function
    X_train, X_valid, y_train, y_valid, n_classes = load_and_split_openml_data(
        OPENML_TASK_ID, RATIO_TRAIN_VALID, download_from_openml=False)
    # Limit sizes for this test:
    n_train = min(X_train.shape[0], 100)
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    n_valid = min(X_valid.shape[0], 100)
    X_valid = X_valid[:n_valid]
    y_valid = y_valid[:n_valid]
    run_mlp_openml = create_train_fn(
        X_train, X_valid, y_train, y_valid, n_classes)
    # Create scheduler and searcher:
    # First two get_config are random, the next three should use BO
    search_options = {
        'num_init_random': 2,
        'debug_log': True}
    if sch_type == 'fifo':
        myscheduler = ag.scheduler.FIFOScheduler(
            run_mlp_openml,
            resource=resources,
            searcher='bayesopt',
            search_options=search_options,
            num_trials=5,
            time_attr=RESOURCE_ATTR_NAME,
            reward_attr=REWARD_ATTR_NAME)
    else:
        myscheduler = ag.scheduler.HyperbandScheduler(
            run_mlp_openml,
            resource=resources,
            searcher='bayesopt',
            search_options=search_options,
            num_trials=5,
            time_attr=RESOURCE_ATTR_NAME,
            reward_attr=REWARD_ATTR_NAME,
            type=sch_type,
            grace_period=1,
            reduction_factor=3,
            brackets=1)
    # Run HPO experiment
    myscheduler.run()
    myscheduler.join_jobs()
    # Not really needed...
    #results_df = process_training_history(
    #    myscheduler.training_history.copy(),
    #    start_timestamp=myscheduler._start_time)
    # Force synchronization
    mx.nd.waitall()

# Note: Currently, all tests are skipped. They crash the CI, for reasons most
# likely due to MXNet (the GP searchers are implemented using MXNet).
# This issue came up in #523 and could not be fixed.
#
# Very similar code to what is here, is contained in
#   docs/tutorials/course/mlp.md
# This code runs properly in the CI.

@pytest.mark.skip(reason="This test is currently crashing the CI (#523)")
def test_bayesopt_fifo():
    run_bayesopt_test('fifo')

@pytest.mark.skip(reason="This test is currently crashing the CI (#523)")
def test_bayesopt_hyperband_stopping():
    run_bayesopt_test('stopping')

@pytest.mark.skip(reason="This test is currently crashing the CI (#523)")
def test_bayesopt_hyperband_promotion():
    run_bayesopt_test('promotion')


if __name__ == "__main__":
    test_bayesopt_fifo()
    test_bayesopt_hyperband_stopping()
    test_bayesopt_hyperband_promotion()

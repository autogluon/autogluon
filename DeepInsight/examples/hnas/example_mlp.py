"""
In this is example we show how to optimize the hyperparameters of a simple 2 layer
feed forward neural networks on MNIST with multi-fidelity methods based on successive halving.
For more details about the method we refer to this paper:

Model-based Asynchronous Hyperparameter and Neural Architecture Search
Aaron Klein, Louis C. Tiao, Thibaut Lienart, Cedric Archambeau, Matthias Seeger
https://arxiv.org/abs/2003.10865

Furthermore, in this tutorial we also show how to save intermediate results on disk to allow for
on-the-fly visualization of the entire optimization process.

For a general introduction of HPO with AutoGluon on pytorch, have a look at this tutorial:
https://auto.gluon.ai/stable/tutorials/torch/hpo.html

We optimize the following hyperparameter:

- number of units first layer
- number of units second layer
- dropout first layer
- dropout second layer
- learning rate of ADAM
- batch size
- weight decay

We use pytorch to implement the neural network. To install it follow the steps described here:
https://pytorch.org/get-started/locally/
"""

import autogluon.core as ag
import time
import logging
import yaml
import argparse

import torch
import torch.nn as nn

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
from torchvision import transforms

logging.basicConfig(level=logging.INFO)


@ag.args(
    n_units_1=ag.space.Int(lower=4, upper=1024),
    n_units_2=ag.space.Int(lower=4, upper=1024),
    dropout_1=ag.space.Real(lower=0, upper=.99),
    dropout_2=ag.space.Real(lower=0, upper=.99),
    learning_rate=ag.space.Real(lower=1e-6, upper=1, log=True),
    batch_size=ag.space.Int(lower=8, upper=128),
    wd=ag.space.Real(lower=1e-8, upper=1, log=True),
    epochs=27
)
def objective_function(args, reporter):
    ts_start = time.time()

    # Hyperparameters to be optimized
    n_units_1 = args.n_units_1
    n_units_2 = args.n_units_2
    dropout_1 = args.dropout_1
    dropout_2 = args.dropout_2
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    wd = args.wd

    # Make sure you downloaded the dataset before running AutoGluon. For example in python:
    # >>> from torchvision import datasets
    # >>> datasets.MNIST(root='data', train=True, download=True)
    #
    # Downloading the data inside the objective function might lead to crashes
    # of the python process (at least on MAC OS).
    data_train = datasets.MNIST(root='data', train=True,
                                download=False, transform=transforms.ToTensor())

    # We use 50000 samples for training and 10000 samples for validation
    indices = list(range(data_train.data.shape[0]))
    train_idx, valid_idx = indices[:50000], indices[50000:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               sampler=train_sampler, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               sampler=valid_sampler, drop_last=True)

    # Define the network architecture
    model = nn.Sequential(
        nn.Linear(28 * 28, n_units_1),
        nn.Dropout(p=dropout_1),
        nn.ReLU(),
        nn.Linear(n_units_1, n_units_2),
        nn.Dropout(p=dropout_2),
        nn.ReLU(),
        nn.Linear(n_units_2, 10)
    )

    # Define the SGD optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    curr_best = None
    for epoch in range(args.epochs):

        # training
        model.train()

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.view(batch_size, -1))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # validating
        model.eval()
        correct = 0
        total = 0
        for data, target in valid_loader:
            output = model(data.view(batch_size, -1))
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        acc = correct / total

        # Every time we improve in terms of validation accuracy, we save the network parameters.
        if curr_best is None or acc > curr_best:
            curr_best = acc
            print("checkpoint model")
            torch.save(model.state_dict(), 'model_checkpoint')

        # We also report the hyperparameter configuration back to AutoGluon,
        # so we can process it later in the callback function, for example for visualization.
        config = dict()
        config["n_units_1"] = n_units_1
        config["n_units_2"] = n_units_2
        config["dropout_1"] = dropout_1
        config["dropout_2"] = dropout_2
        config["batch_size"] = batch_size
        config["learning_rate"] = learning_rate
        config["wd"] = wd

        ts_now = time.time()
        eval_time = ts_start - ts_now
        reporter(
            epoch=epoch + 1,
            performance=float(curr_best),
            eval_time=eval_time,
            time_step=ts_now, **config)


def callback(training_history, start_timestamp):
    # This callback function will be executed every time AutoGluon collected some new data.
    # In this example we will parse the training history into a .csv file and save it to disk, such that we can
    # for example plot AutoGluon's performance during the optimization process.
    # If you don't care about analyzing performance of AutoGluon online, you can also ignore this callback function and
    # just save the training history after AutoGluon has finished.
    import pandas as pd
    task_dfs = []

    # this function need to be changed if you return something else than accuracy
    def compute_error(df):
        return 1.0 - df["performance"]

    def compute_runtime(df, start_timestamp):
        return df["time_step"] - start_timestamp

    for task_id in training_history:
        task_df = pd.DataFrame(training_history[task_id])
        task_df = task_df.assign(task_id=task_id,
                                 runtime=compute_runtime(task_df, start_timestamp),
                                 error=compute_error(task_df),
                                 target_epoch=task_df["epoch"].iloc[-1])
        task_dfs.append(task_df)

    result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)

    # re-order by runtime
    result = result.sort_values(by="runtime")

    # calculate incumbent best -- the cumulative minimum of the error.
    result = result.assign(best=result["error"].cummin())

    result.to_csv("history.csv")


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Runs autogluon to optimize '
                                                 'the hyperparameters of a simple MLP of MNIST ')
    parser.add_argument('--num_trials', default=10, type=int,
                        help='number of trial tasks. It is enough to either set num_trials or timout.')
    parser.add_argument('--timeout', default=1000, type=int,
                        help='runtime of autogluon in seconds. It is enough to either set num_trials or timout.')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='number of GPUs available to a given trial.')
    parser.add_argument('--num_cpus', type=int, default=2,
                        help='number of CPUs available to a given trial.')
    parser.add_argument('--hostfile', type=argparse.FileType('r'))
    parser.add_argument('--store_results_period', type=int, default=100,
                        help='If specified, results are stored in intervals of '
                             'this many seconds (they are always stored at '
                             'the end)')
    parser.add_argument('--scheduler', type=str, default='hyperband_promotion',
                        choices=['hyperband_stopping', 'hyperband_promotion'],
                        help='Asynchronous scheduler type. In case of doubt leave it to the default')
    parser.add_argument('--reduction_factor', type=int, default=3,
                        help='Reduction factor for successive halving')
    parser.add_argument('--brackets', type=int, default=1,
                        help='Number of brackets. Setting the number of brackets to 1 means '
                             'that we run effectively successive halving')
    parser.add_argument('--min_resource_level', type=int, default=1,
                        help='Minimum resource level (i.e epochs) on which a configuration is evaluated on.')
    parser.add_argument('--searcher', type=str, default='bayesopt',
                        choices=['random', 'bayesopt'],
                        help='searcher to sample new configurations')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # In case you want to run AutoGluon across multiple instances, you need to provide it with a list of
    # the IP addresses of the instances. Here, we assume that the IP addresses are stored in a .yaml file.
    # If you want to run AutoGluon on a single instance just pass None.
    # However, keep in mind that it will still parallelize the optimization process across multiple threads then.
    # If you really want to run it purely sequentially, set the num_cpus equal to the number of VCPUs of the machine.
    dist_ip_addrs = yaml.load(args.hostfile) if args.hostfile is not None else []

    del args.hostfile

    print("Got worker host IP addresses [{}]".format(dist_ip_addrs))

    if args.scheduler == "hyperband_stopping":
        hyperband_type = "stopping"

    elif args.scheduler == "hyperband_promotion":
        hyperband_type = "promotion"

    scheduler = ag.scheduler.HyperbandScheduler(objective_function,
                                                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                                                # Autogluon runs until it either reaches num_trials or time_out
                                                num_trials=args.num_trials,
                                                time_out=args.timeout,
                                                # This argument defines the metric that will be maximized.
                                                # Make sure that you report this back in the objective function.
                                                reward_attr='performance',
                                                # The metric along we make scheduling decision. Needs to be also
                                                # reported back to AutoGluon in the objective function.
                                                time_attr='epoch',
                                                brackets=args.brackets,
                                                checkpoint=None,
                                                searcher=args.searcher,  # Defines searcher for new configurations
                                                dist_ip_addrs=dist_ip_addrs,
                                                training_history_callback=callback,
                                                training_history_callback_delta_secs=args.store_results_period,
                                                reduction_factor=args.reduction_factor,
                                                type=hyperband_type,
                                                # defines the minimum resource level for Hyperband,
                                                # i.e the minimum number of epochs
                                                grace_period=args.min_resource_level
                                                )
    scheduler.run()
    scheduler.join_jobs()

    # final training history
    training_history = scheduler.training_history

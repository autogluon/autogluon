"""
This example show how to run AG on HPOBench (https://github.com/automl/HPOBench) benchmarks.
For simplicity we use the NASBench201 benchmark.
To run this benchmark you need to follow the installation guide of HPOBench.

We run AG here on a single instance. If you want to run it across instances, have a look at the mlp_examply.py script
in the example folder (that also shows you how to store intermediate results in a callback function).

"""

import autogluon.core as ag
import time
import logging
import argparse

from hpolib.benchmarks.nas.nasbench_201 import NasBench201BaseBenchmark


def make_benchmark(dataset_name='cifar10-valid'):
    b = NasBench201BaseBenchmark(dataset_name)
    cs = b.get_configuration_space()

    d = dict()
    for h in cs.get_hyperparameters():
        d[h.name] = ag.space.Categorical(*h.choices)

    @ag.args(**d, epochs=199)
    def objective_function(args, reporter, **kwargs):

        ts_start = time.time()

        config = dict()
        for h in cs.get_hyperparameters():
            config[h.name] = args[h.name]

        for epoch in range(args.epochs):

            res = b.objective_function(config, fidelity={'epoch': epoch})

            acc = 1 - res['function_value'] / 100

            ts_now = time.time()
            eval_time = ts_start - ts_now
            reporter(
                epoch=epoch + 1,
                performance=acc,
                eval_time=eval_time,
                time_step=ts_now, **config)

    return objective_function


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='runs autogluon on autoff benchmarks')

    parser.add_argument('--num_trials', default=10, type=int,
                        help='number of trial tasks')
    parser.add_argument('--timeout', default=1000, type=int,
                        help='runtime of autogluon in seconds')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='number of GPUs available to a given trial.')
    parser.add_argument('--num_cpus', type=int, default=2,
                        help='number of CPUs available to a given trial.')

    parser.add_argument('--store_results_period', type=int, default=100,
                        help='If specified, results are stored in intervals of '
                             'this many seconds (they are always stored at '
                             'the end)')
    parser.add_argument('--scheduler', type=str, default='hyperband_promotion',
                        choices=['hyperband_stopping', 'hyperband_promotion'],
                        help='Asynchronous scheduler type. In case of doubt leave it to the default')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    logging.root.setLevel(logging.INFO)

    if args.scheduler == "hyperband_stopping":
        hyperband_type = "stopping"

    elif args.scheduler == "hyperband_promotion":
        hyperband_type = "promotion"

    brackets = 1  # setting the number of brackets to 1 means that we run effectively successive halving

    f = make_benchmark('cifar10-valid')
    scheduler = ag.scheduler.HyperbandScheduler(f,
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
                                                brackets=brackets,
                                                checkpoint=None,
                                                searcher="bayesopt",  # Defines searcher for new configurations
                                                training_history_callback_delta_secs=args.store_results_period,
                                                reduction_factor=3,
                                                type=hyperband_type,
                                                # defines the minimum resource level for Hyperband,
                                                # i.e the minimum number of epochs
                                                grace_period=1)
    scheduler.run()
    scheduler.join_jobs()

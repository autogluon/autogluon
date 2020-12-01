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

logging.basicConfig(level=logging.INFO)


def make_benchmark(dataset_name='cifar10-valid', do_sleep=True):
    b = NasBench201BaseBenchmark(dataset_name)
    cs = b.get_configuration_space()

    d = dict()
    for h in cs.get_hyperparameters():
        d[h.name] = ag.space.Categorical(*h.choices)

    @ag.args(**d, epochs=199, do_sleep=do_sleep)
    def objective_function(args, reporter):

        ts_start = time.time()

        config = dict()
        for h in cs.get_hyperparameters():
            config[h.name] = args[h.name]

        for epoch in range(args.epochs):

            res = b.objective_function(config, fidelity={'epoch': epoch})

            acc = 1 - res['function_value'] / 100

            if do_sleep:
                # To simulate the dynamics of asynchronous HNAS, we put the worker asleep for the same time
                # it would have taken to evaluate the real benchmark. Note that, while this does not give us a
                # speed up in wall-clock time, it still allows us to use CPUs instead of more expensive GPU instances.
                time.sleep(res['eval_cost'])
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
    parser = argparse.ArgumentParser(description='runs autogluon on HPOLib2 benchmarks')

    parser.add_argument('--num_trials', default=10, type=int,
                        help='number of trial tasks')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed of the random number generator')
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
                        choices=['hyperband_stopping', 'hyperband_promotion', 'fifo'],
                        help='Asynchronous scheduler type. In case of doubt leave it to the default')
    parser.add_argument('--searcher', type=str, default='bayesopt',
                        choices=['bayesopt', 'random'],
                        help='Defines if we sample configuration randomly or from a model.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    if args.scheduler == "hyperband_stopping":
        hyperband_type = "stopping"

    elif args.scheduler == "hyperband_promotion":
        hyperband_type = "promotion"

    brackets = 1  # setting the number of brackets to 1 means that we run effectively successive halving

    f = make_benchmark('cifar10-valid')

    if args.scheduler == 'fifo':
        scheduler = ag.scheduler.FIFOScheduler(f,
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
                                               search_options={'random_seed': args.seed},
                                               searcher=args.searcher,  # Defines searcher for new configurations
                                               training_history_callback_delta_secs=args.store_results_period)
    else:
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
                                                    searcher=args.searcher,  # Defines searcher for new configurations
                                                    training_history_callback_delta_secs=args.store_results_period,
                                                    reduction_factor=3,
                                                    type=hyperband_type,
                                                    search_options={'random_seed': args.seed},
                                                    # defines the minimum resource level for Hyperband,
                                                    # i.e the minimum number of epochs
                                                    grace_period=1)
    scheduler.run()
    scheduler.join_jobs()

import argparse
from datetime import datetime
import autogluon.core as ag
from imageClassification import train_image_classification
import wandb

parser = argparse.ArgumentParser()

parser.add_argument('--num_cpus', default=2, type=int, help='number of CPUs to use')
parser.add_argument('--num_gpus', default=0, type=int, help='number of GPUs to use')
parser.add_argument('--num_trials', default=6, type=int, help='number of trials to run')
parser.add_argument('--ip', default=None, help='additional ips to be added')

args = parser.parse_args()

# define all the tasks
tasks = [
    train_image_classification, # image classification task
]

# define run time table
run_times = []
ext_ips = [args.ip]

# Run every task with all available schedulers
for task in tasks:

    # define all schedulers
    schedulers = [
        ag.scheduler.FIFOScheduler(
            task,
            resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
            num_trials=args.num_trials,
            time_attr='epoch',
            reward_attr='accuracy',
            dist_ip_addrs=ext_ips
        ),    # add the FIFO scheduler

        ag.scheduler.HyperbandScheduler(
            task,
            resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
            num_trials=args.num_trials,
            time_attr='epoch',
            reward_attr='accuracy',
            dist_ip_addrs=ext_ips
        ),  # add the Hyperband scheduler

        ag.scheduler.RLScheduler(
            task,
            resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
            num_trials=args.num_trials,
            time_attr='epoch',
            reward_attr='accuracy',
            dist_ip_addrs = ext_ips
        )  # add the FIFO scheduler
    ]

    # define the scheduler run time list
    scheduler_runtimes = []

    # run the task with each scheduler
    for scheduler in schedulers:

        # initialize the logging
        run = wandb.init(project='autogluon')

        # display the scheduler and available resources
        print('')
        print(scheduler)
        print('')

        # start the clock
        start_time = datetime.now()

        # run the job with the scheduler
        scheduler.run()
        scheduler.join_jobs()

        # stop the clock
        stop_time = datetime.now()
        scheduler_runtimes.append((stop_time-start_time).total_seconds())

        # end the run
        run.finish()

    run_times.append(scheduler_runtimes)


print(run_times)
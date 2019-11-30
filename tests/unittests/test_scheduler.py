from unittest import TestCase
import numpy as np
import autogluon as ag

@ag.args(
    lr=ag.space.Real(1e-3, 1e-2, log=True),
    wd=ag.space.Real(1e-3, 1e-2))
def train_fn(args, reporter):
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

@ag.args(
    lr=ag.space.Categorical(1e-3, 1e-2),
    wd=ag.space.Categorical(1e-3, 1e-2))
def rl_train_fn(args, reporter):
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

def test_schedulers():
    def test_fifo_scheduler():
        scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                               resource={'num_cpus': 2, 'num_gpus': 0},
                                               num_trials=10,
                                               reward_attr='accuracy',
                                               time_attr='epoch')
        scheduler.run()
        scheduler.join_jobs()

    def test_hyperband_scheduler():
        scheduler = ag.scheduler.HyperbandScheduler(train_fn,
                                                    resource={'num_cpus': 2, 'num_gpus': 0},
                                                    num_trials=10,
                                                    reward_attr='accuracy',
                                                    time_attr='epoch',
                                                    grace_period=1)
        scheduler.run()
        scheduler.join_jobs()

    def test_rl_scheduler():
        scheduler = ag.scheduler.RLScheduler(rl_train_fn,
                                             resource={'num_cpus': 2, 'num_gpus': 0},
                                             num_trials=10,
                                             reward_attr='accuracy',
                                             time_attr='epoch')
        scheduler.run()
        scheduler.join_jobs()

    test_fifo_scheduler()
    test_hyperband_scheduler()
    test_rl_scheduler()

if __name__ == '__main__':
    test_schedulers()

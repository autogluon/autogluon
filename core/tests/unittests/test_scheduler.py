import numpy as np
import pickle
import time
import autogluon.core as ag


def test_fifo_scheduler():
    @ag.args(
        lr=ag.space.Real(1e-3, 1e-2, log=True),
        wd=ag.space.Real(1e-3, 1e-2),
        epochs=10)
    def train_fn(args, reporter):
        for e in range(args.epochs):
            dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2 * e))
            reporter(epoch=e + 1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

    scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                           resource={'num_cpus': 4, 'num_gpus': 0},
                                           num_trials=10,
                                           reward_attr='accuracy',
                                           time_attr='epoch',
                                           checkpoint=None)
    scheduler.run()
    scheduler.join_jobs()
    best_config = scheduler.get_best_config()
    best_task_id = scheduler.get_best_task_id()
    assert pickle.dumps(scheduler.config_history[best_task_id]) == pickle.dumps(best_config)


def test_hyperband_scheduler():
    @ag.args(
        lr=ag.space.Real(1e-3, 1e-2, log=True),
        wd=ag.space.Real(1e-3, 1e-2),
        epochs=10)
    def train_fn(args, reporter):
        for e in range(args.epochs):
            dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2 * e))
            reporter(epoch=e + 1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

    scheduler = ag.scheduler.HyperbandScheduler(train_fn,
                                                resource={'num_cpus': 4, 'num_gpus': 0},
                                                num_trials=10,
                                                reward_attr='accuracy',
                                                time_attr='epoch',
                                                grace_period=1,
                                                checkpoint=None)
    scheduler.run()
    scheduler.join_jobs()
    best_config = scheduler.get_best_config()
    best_task_id = scheduler.get_best_task_id()
    assert pickle.dumps(scheduler.config_history[best_task_id]) == pickle.dumps(best_config)


def test_timeout_scheduler():
    @ag.args(lr=ag.space.Real(1E-5, 1E-3))
    def foo(args, reporter):
        start_tick = time.time()
        for i in range(10):
            # Sleep for 1 second
            time.sleep(1)
            reporter(reward=time.time() - start_tick,
                     time_attr=i)
    scheduler = ag.scheduler.FIFOScheduler(foo,
                                           resource={'num_cpus': 4, 'num_gpus': 0},
                                           num_trials=3,
                                           reward_attr='reward',
                                           time_attr='time_attr',
                                           time_out=3,
                                           checkpoint=None)
    scheduler.run()
    scheduler.join_jobs(timeout=3)

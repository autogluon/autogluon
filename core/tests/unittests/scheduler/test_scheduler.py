import pickle
import time

import numpy as np

from autogluon.common import space
from autogluon.core.scheduler import LocalSequentialScheduler


def test_local_sequential_scheduler():
    search_space = dict(
        lr=space.Real(1e-3, 1e-2, log=True),
        wd=space.Real(1e-3, 1e-2),
        epochs=10,
    )

    def train_fn(args, reporter):
        for e in range(args["epochs"]):
            dummy_reward = 1 - np.power(1.8, -np.random.uniform(e, 2 * e))
            reporter(epoch=e + 1, reward=dummy_reward, lr=args.lr, wd=args.wd)

    scheduler = LocalSequentialScheduler(train_fn, search_space=search_space, num_trials=10)
    scheduler.run()
    scheduler.join_jobs()
    best_config = scheduler.get_best_config()
    best_task_id = scheduler.get_best_task_id()
    assert pickle.dumps(scheduler.config_history[best_task_id]) == pickle.dumps(best_config)


def test_timeout_scheduler():
    search_space = dict(
        lr=space.Real(1e-5, 1e-3),
    )

    def train_fn(args, reporter):
        start_tick = time.time()
        time.sleep(0.01)
        reporter(reward=time.time() - start_tick, time_attr=0)

    scheduler = LocalSequentialScheduler(
        train_fn, search_space=search_space, num_trials=7, time_attr="time_attr", time_out=0.025
    )
    scheduler.run()
    scheduler.join_jobs()
    assert len(scheduler.config_history) <= 2

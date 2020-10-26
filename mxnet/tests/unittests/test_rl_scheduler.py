import numpy as np
import pickle
import autogluon.core as ag


def test_rl_scheduler():
    @ag.args(
        lr=ag.space.Categorical(1e-3, 1e-2),
        wd=ag.space.Categorical(1e-3, 1e-2),
        epochs=10)
    def rl_train_fn(args, reporter):
        for e in range(args.epochs):
            dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2 * e))
            reporter(epoch=e + 1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)
    scheduler = ag.scheduler.RLScheduler(rl_train_fn,
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

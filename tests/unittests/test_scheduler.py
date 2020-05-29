import numpy as np
import autogluon as ag

@ag.args(
    lr=ag.space.Real(1e-3, 1e-2, log=True),
    wd=ag.space.Real(1e-3, 1e-2))
def train_fn(args, reporter):
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

@ag.args(
    lr=ag.space.Categorical(1e-3, 1e-2),
    wd=ag.space.Categorical(1e-3, 1e-2))
def rl_train_fn(args, reporter):
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)


def test_fifo_scheduler():
    scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                           resource={'num_cpus': 4, 'num_gpus': 0},
                                           num_trials=10,
                                           reward_attr='accuracy',
                                           time_attr='epoch',
                                           checkpoint=None)
    scheduler.run()
    scheduler.join_jobs()

def test_hyperband_scheduler():
    scheduler = ag.scheduler.HyperbandScheduler(train_fn,
                                                resource={'num_cpus': 4, 'num_gpus': 0},
                                                num_trials=10,
                                                reward_attr='accuracy',
                                                time_attr='epoch',
                                                grace_period=1,
                                                checkpoint=None)
    scheduler.run()
    scheduler.join_jobs()

def test_rl_scheduler():
    scheduler = ag.scheduler.RLScheduler(rl_train_fn,
                                         resource={'num_cpus': 4, 'num_gpus': 0},
                                         num_trials=10,
                                         reward_attr='accuracy',
                                         time_attr='epoch',
                                         checkpoint=None)
    scheduler.run()
    scheduler.join_jobs()

def test_nvismi_util():
    from autogluon.scheduler.resource.nvutil import NviSMI
    from autogluon.scheduler.resource import get_gpu_count
    num_gpu = get_gpu_count()
    if not num_gpu > 0:
        return
    import mxnet as mx
    gpu_arrays = [mx.nd.zeros((1, 2, 3), ctx=mx.gpu(i)) for i in range(num_gpu)]
    monitor = [NviSMI(i) for i in range(num_gpu)]
    for i in range(num_gpu):
        assert monitor[i].get_memory_info().used > 0
        u = monitor[i].get_utilization_rates().gpu
        assert u >= 0 and u <= 100

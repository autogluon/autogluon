import time
import argparse
import socket
import logging
import numpy as np
import autogluon as ag

logger = logging.getLogger(__name__)

@ag.autogluon_method
def my_task(args):
    print('task_id: {}, lr is {}'.format(args.task_id, args.lr))
    time.sleep(3.0)

def test_scheduler():
    scheduler = ag.scheduler.TaskScheduler()
    print('scheduler', scheduler)
    args = argparse.ArgumentParser()
    config = {'lr': ag.searcher.sample_from(
            lambda: np.power(10.0, np.random.uniform(-4, -1)))}
    for i in range(10):
        resource = ag.resource.Resources(num_cpus=2, num_gpus=1)
        task = ag.scheduler.Task(my_task, {'args': args, 'config': config}, resource)
        scheduler.add_task(task)
    scheduler.join_tasks()

def test_dist_scheduler():
    scheduler = ag.distributed.DistributedTaskScheduler(['172.31.3.95'])
    print('scheduler', scheduler)
    args = argparse.ArgumentParser()
    config = {'lr': ag.searcher.sample_from(
            lambda: np.power(10.0, np.random.uniform(-4, -1)))}
    for i in range(10):
        resource = ag.resource.DistributedResource(num_cpus=2, num_gpus=1)
        task = ag.scheduler.Task(my_task, {'args': args, 'config': config}, resource)
        scheduler.add_task(task)
    scheduler.join_tasks()
    scheduler.shutdown()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import nose
    nose.runmodule()

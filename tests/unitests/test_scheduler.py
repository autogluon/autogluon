import time
import socket
import logging
import autogluon as ag

logger = logging.getLogger(__name__)

def my_task():
    import mxnet as mx
    print('Running my_task on machine {}'.format(socket.gethostbyname(socket.gethostname())))
    time.sleep(3.0)
    print('GPU devices: {}'.format(mx.test_utils.list_gpus()))

def test_scheduler():
    scheduler = ag.scheduler.TaskScheduler()
    print('scheduler', scheduler)
    for i in range(10):
        resource = ag.resource.Resources(num_cpus=2, num_gpus=1)
        task = ag.scheduler.Task(my_task, {}, resource)
        scheduler.add_task(task)
    scheduler.join_tasks()

def test_dist_scheduler():
    scheduler = ag.dist.DistributedTaskScheduler(['172.31.3.95'])
    print('scheduler', scheduler)
    for i in range(10):
        resource = ag.resource.DistributedResource(num_cpus=2, num_gpus=1)
        task = ag.scheduler.Task(my_task, {}, resource)
        scheduler.add_task(task)
    scheduler.join_tasks()
    scheduler.exit()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import nose
    nose.runmodule()

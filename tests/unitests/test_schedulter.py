import time
import logging
import autogluon as ag

logger = logging.getLogger(__name__)

def my_task():
    logger.debug('running, my_task')
    time.sleep(3.0)

def test_scheduler():
    scheduler = ag.scheduler.TaskScheduler()
    for i in range(10):
        resource = ag.scheduler.Resources(num_cpus=2, num_gpus=1)
        task = ag.scheduler.Task(my_task, {}, resource)
        scheduler.add_task(task)

    for i in range(4):
        scheduler.logging_running_tasks()
        time.sleep(5.0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import nose
    nose.runmodule()

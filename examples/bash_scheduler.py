import subprocess
import numpy as np
import autogluon as ag

def my_task(lr):
    subprocess.run(['python', 'examples/mnist_native.py', '--lr', str(lr)])

if __name__ == '__main__':
    scheduler = ag.scheduler.TaskScheduler()
    for i in range(10):
        lr = np.power(10.0, np.random.uniform(-4, -1))
        resource = ag.scheduler.Resources(num_cpus=2, num_gpus=1)
        task = ag.scheduler.Task(my_task, {'lr': lr}, resource)
        scheduler.add_task(task)

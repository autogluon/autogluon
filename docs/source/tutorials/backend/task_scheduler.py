"""1. Understand FIT Backend
=========================

This is a quick tutorial for understanding fit backend.
"""
from autogluon import Task
from autogluon.resource import Resources
from autogluon.scheduler import TaskScheduler
import time

################################################################
# Understanding Task, Resource and Scheduler
# __________________________________________
# 
# .. image:: ../../../_static/img/scheduler.png
#
# Resources contains number of cpus and gpus. Task includes execute function, args
# and its resource. TaskScheduler automatically request resource for tasks and execut
# it as soon as its resource is ready.
#
# **Define Custimized Task Function**

def my_task():
    print('running, my_task')
    time.sleep(3.0)

################################################################

# **Create Scheduler**
scheduler = TaskScheduler()

for i in range(5):
    resource = Resources(num_cpus=2, num_gpus=0)
    task = Task(my_task, {}, resource)
    scheduler.add_task(task)

################################################################
# Launch Task with Different Configurations
# _________________________________________

import autogluon as ag
from autogluon import autogluon_method

################################################################
# **Configuration handled by `autogluon_method`**

@autogluon_method
def train_fn(args):
    print('lr is {}'.format(args.lr))
    
import argparse
import numpy as np
args = argparse.ArgumentParser()
config = {'lr': ag.searcher.sample_from(
          lambda: np.power(10.0, np.random.uniform(-4, -1)))}

################################################################
# **Schedule Tasks**

myscheduler = TaskScheduler()
for i in range(5):
    resource = Resources(num_cpus=2, num_gpus=0)
    task = Task(train_fn, {'args': args, 'config': config}, resource)
    myscheduler.add_task(task)

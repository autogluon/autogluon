"""2. Setup Distributed Machines
=============================

This is a quick tutorial for setting up AutoGluon with Distributed Training.
AutoGluon automatically schedule tasks onto remote machines, just like local one.
AutoGluon handles the communications and provides the same experience of
a big machine with many GPUs.

System Implementation Logics
----------------------------

The main training script (main python file) is serialized and scheduled remotely.
AutoGluon distributed schedulers monitors the training process and gather the results.

Any files (such as python scripts and dataset) beyond the main training script need to
be made accessible. We recommand the following practice:

- Use ``scheduler.upload_files(files_list)`` to upload individual python scripts or small datasets to the excecution folder, so that the main script can import or load.

- Make a python library for many files in the same folder and install it manually on different machine.

- Upload large files (such as dataset) mannually to different machines and share the same absolute filepath, because the tasks can be scheduled to different machines.


Distributed Training Setup on AWS EC2
-------------------------------------

Here are the steps for preparing the environment to run imagenet example ``examples/imagenet_autogluon.py``.
We will first set up two machines, one master machine and one worker machine.
Then we may use EC2 AMI to clone as many worker machines as you want.

- Create two EC2 instances in the same zone for speed purpose, using AWS Deep Learnng AMI (optional). Make sure the SSH port and All TCP ports are open (0 - 65535) in your security group. You may refer to this `tutorial <http://cs231n.github.io/aws-tutorial/>`_ , if you don't have the experience using AWS EC2.

- Install AutoGluon and MXNet on each machine. If you have other dependencies in your customized training scripts, please also install them.

- Make the worker machine accessible by the master machine through ssh. The following steps may be needed:
 
    - Generate ssh key by executing `ssh-keygen` on master machine.

    - Copy the public key from master machine `cat ~/.ssh/id_rsa.pub` and paste the terminal output to the worker machine `~/.ssh/authorized_keys`.

    - `ssh worker_ip_address` to the worker machine through master. (Note that the worker ip address can be found at the terminal, for example `ubuntu@ip-172-31-23-33` means the ip address is `172.31.23.33`)

- Upload the datasets or large to each machines if needed.

- Create EC2 image of the worker machine, and use that to create more worker machines if needed.

- You are all set for running experiments. Just provide the list of remote ip addresses to the scheduler.


Resource Management
-------------------

.. image:: ../../../_static/img/distributed_resource_manager.png


A Toy Example
-------------

"""

import time
import numpy as np
import autogluon as ag
from autogluon.basic import autogluon_register_args
from autogluon.resource import DistributedResource

################################################################
# Construct a fake training function for demo
#

@autogluon_register_args(
    batch_size=64,
    lr=ag.LogLinearSpace(1e-4, 1e-1),
    momentum=0.9,
    wd=ag.LinearSpace(1e-4, 5e-4),
    )
def train_fn(args, reporter):
    print('task_id: {}, lr: {}'.format(args.task_id, args.lr))
    for e in range(10):
        top1_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=top1_accuracy)
    # wait for 1 sec
    time.sleep(1.0)

################################################################
# Create a Random Searcher
#

searcher = ag.searcher.RandomSampling(train_fn.cs)
print( searcher.get_config())

################################################################
# Provide a list of ip addresses for remote machines
#

extra_node_ips = ['172.31.3.95']

################################################################
# Create a distributed scheduler. If no ipaddresses are provided, 
# scheduler will only use the local resources
#

scheduler = ag.distributed.DistributedFIFOScheduler(
    train_fn, train_fn.args,
    resource={'num_cpus': 2, 'num_gpus': 1},
    searcher=searcher,
    dist_ip_addrs=extra_node_ips)
print(scheduler)

################################################################
# Launch 16 Tasks
#

scheduler.run(16)
scheduler.join_tasks()

################################################################
# Plot the results and shut down (required for distributed version)
#

scheduler.get_training_curves()
scheduler.shutdown()

"""2. AutoGluon Image Classification - Advanced
============================================

In the following\ *, we use Image Classification as a running example*
to illustrate the usage of AutoGluon’s main APIs.
"""

import warnings
warnings.filterwarnings("ignore")

from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)

################################################################
# We first introduce the basic configuration ``autogluon.space``, which is
# used to represent the search space of each task components, we will then
# go throught each components, including
#
# -  ``autogluon.Dataset``
# -  ``autogluon.Nets``
# -  ``autogluon.Optimizers``
# -  ``autogluon.Losses``
# -  ``autogluon.Metrics``
#
# and finally put all together to ``fit`` to generate best results.
#
# Import AutoGluon
# ~~~~~~~~~~~~~~~~


import autogluon as ag

# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use CIFAR10 for image classfication for demo purpose.


dataset = task.Dataset(name='CIFAR10') # case insentive

################################################################
# We then will use ``autogluon.Nets`` and ``autogluon.Optimizers`` as
# examples to show the usage of auto objects. The remainining auto objects
# are using default value.
#
# Create AutoGluon Nets
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon.Nets`` is a list of auto networks, and allows search for the
# best net
#
# -  from a list of provided (or default) networks
# -  by choosing the best architecture regarding to each auto net.
#


# type of net_list is ag.space.List

# method 1 (complex but flexiable): specify the net_list using get_model
# net_list = [task.model_zoo.get_model('resnet18_v1'), # TODO: pretrained and pretrained_dataset would be supported
#             task.model_zoo.get_model('resnet34_v1')]

# method 2 (easy and less flexiable): specify the net_list using model name
net_list = ['resnet18_v1',
            'resnet34_v1']

# default net list for image classification would be overwritten
# if net_list is provided
nets = ag.Nets(net_list)

print(nets)

################################################################
# Create AutoGluon Optimizers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon.Optimizers`` defines a list of optimization algorithms that
# allows search for the best optimization algorithm
#
# -  from a list of provided (or default) optimizers
# -  by choosing the best hyper-parameters regarding to each auto
#    optimizer
#

# method 1 (complex but flexiable): specify the optim_list using get_optim
# optimizers = ag.Optimizers([ag.optim.get_optim('sgd'),
#                             ag.optim.get_optim('adam')])

# method 2 (easy and less flexiable): specify the optim_list using get_model
optimizers = ag.Optimizers(['sgd', 'adam'])

print(optimizers)


################################################################
# Create AutoGluon Fit - Put all together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

time_limits = 1*60*60
max_metric = 1.0
max_trial_count = 4
max_num_gpus = 1
max_num_cpus = 4
max_training_epochs = 2
demo = True

stop_criterion = {
    'time_limits': time_limits,
    'max_metric': max_metric,
    'max_trial_count': max_trial_count
}

resources_per_trial = {
    'max_num_gpus': max_num_gpus, # set this to more than 1 if you have GPU machine to run more efficiently.
    'max_num_cpus': max_num_cpus,
    'max_training_epochs': max_training_epochs
}

results = task.fit(dataset,
                   nets,
                   optimizers,
                   stop_criterion=stop_criterion,
                   resources_per_trial=resources_per_trial,
                   demo=demo) # demo=True is recommened when running on no GPU machine

################################################################
# The best accuracy is:


print('%.2f acc' % (results.metric * 100))

################################################################
# The best associated configuration is:

print(results.config)

################################################################
# Total time cost is:


print('%.2f s' % results.time)

################################################################
# Use Search Algorithm
# ~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon.searcher`` will support both basic and SOTA searchers for
# both hyper-parameter optimization and architecture search. We now
# support random search. The default is using searcher is random searcher.


# cs is CS.ConfigurationSpace() where import ConfigSpace as CS, this is just example code;
# in practice, this is in fit function, and cs should not be None
cs = None
searcher = ag.searcher.RandomSampling(cs)

print(searcher)

################################################################
# Or simply use string name:
#

searcher = 'random'

print(searcher)

################################################################
# Use Trial Scheduler
# ~~~~~~~~~~~~~~~~~~~
#
# ``ag.scheduler`` supports scheduling trials in serial order and with
# early stopping.
#
# We support basic FIFO scheduler.
#


# this is just example code; in practice, this is in fit function
savedir = 'checkpoint/demo.ag'

# trial_scheduler = ag.scheduler.FIFO_Scheduler(
#                 task.pipeline.train_image_classification,
#                 None,
#                 {
#                     'num_cpus': 4,
#                     'num_gpus': 0,
#                 },
#                 searcher,
#                 checkpoint=savedir)
#
# print(trial_scheduler)

################################################################
# We also support Hyperband which is an early stopping mechanism.


# this is just example code; in practice, this is in fit function
# trial_scheduler = ag.scheduler.Hyperband_Scheduler(
#                 task.pipeline.train_image_classification,
#                 None,
#                 {
#                     'num_cpus': 4,
#                     'num_gpus': 1,
#                 },
#                 searcher,
#                 time_attr='epoch',
#                 reward_attr='accuracy',
#                 max_t=10,
#                 grace_period=1,
#                 checkpoint=savedir)
#
# print(trial_scheduler)

################################################################
# Resume Fit and Checkpointer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We use the resume and checkpoint dir in the scheduler.


savedir = 'checkpoint/demo.ag'
resume = False


# trial_scheduler = ag.scheduler.Hyperband_Scheduler(
#                 task.pipeline.train_image_classification,
#                 None,
#                 {
#                     'num_cpus': 4,
#                     'num_gpus': 1,
#                 },
#                 searcher,
#                 checkpoint=savedir,
#                 resume=resume,
#                 time_attr='epoch',
#                 reward_attr='accuracy',
#                 max_t=10,
#                 grace_period=1)
#
# print(trial_scheduler)

################################################################
# Or simply specify the trial scheduler with the string name:

trial_scheduler = 'hyperband'

################################################################
# Visualize Using Tensor/MXBoard
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We could visualize the traing curve using Tensorboad or MXboard. To
# start the Tensorboard or MXboard, please use:
#
# ``tensorboard --logdir=./checkpoint/demo/logs --host=127.0.0.1 --port=8889``
#
# An example is shown below.
#
# Create Stop Criterion
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon`` supports overall automatic constraints in
# ``stop_criterion``.
#
max_metric = 0.80

stop_criterion = {
    'time_limits': time_limits,
    'max_metric': max_metric, #if you know, otherwise use the default 1.0
    'max_trial_count': max_trial_count
}

################################################################
# Create Resources Per Trial
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon`` supports constraints for each trial
# ``in resource_per_trial``.
#

max_training_epochs = 1

resources_per_trial = {
    'max_num_gpus': max_num_gpus,
    'max_num_cpus': max_num_cpus,
    'max_training_epochs': max_training_epochs
}

################################################################
# Create AutoGluon Fit with Full Capacity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

results = task.fit(dataset,
                  nets,
                  optimizers,
                  searcher=searcher,
                  trial_scheduler='fifo',
                  resume=resume,
                  savedir=savedir,
                  stop_criterion=stop_criterion,
                  resources_per_trial=resources_per_trial,
                  demo=True) # only set demo=True when running on no GPU machine

################################################################
# The best accuracy is

print('%.2f acc' % (results.metric * 100))

################################################################
# The associated best configuration is:


print(results.config)

################################################################
# The total time cost is:

print('%.2f s' % results.time)

################################################################
# Resume AutoGluon Fit
# ~~~~~~~~~~~~~~~~~~~~
#
# We could resume the previous training for more epochs to achieve better
# results. Similarly, we could also increase ``max_trial_count`` for
# better results.
#
# Here we increase the ``max_training_epochs`` from 1 to 3,
# ``max_trial_count`` from 2 to 3, and set ``resume = True`` which will
# load the checking point in the savedir.

max_trial_count = 3

stop_criterion = {
    'time_limits': time_limits,
    'max_metric': max_metric,
    'max_trial_count': max_trial_count
}

max_training_epochs = 3

resources_per_trial = {
    'max_num_gpus': max_num_gpus,
    'max_num_cpus': max_num_cpus,
    'max_training_epochs': max_training_epochs
}

resume = True

results = task.fit(dataset,
                  nets,
                  optimizers,
                  searcher=searcher,
                  trial_scheduler='fifo',
                  resume=resume,
                  savedir=savedir,
                  stop_criterion=stop_criterion,
                  resources_per_trial=resources_per_trial,
                  demo=True)

################################################################
# The best accuracy is


print('%.2f acc' % (results.metric * 100))

################################################################
# The associated best configuration is:


print(results.config)

################################################################
# The total time cost is:


print('%.2f s' % results.time)

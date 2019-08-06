'''
2. AutoGluon Text Classification - Advanced
=============================================
'''

import warnings
warnings.filterwarnings("ignore")

from autogluon import text_classification as task

import logging
logging.basicConfig(level=logging.INFO)

##########################################################################
# We first introduce the basic configuration ``autogluon.space``, which is
# used to represent the search space of each task components, we will then
# go through each components, including
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
# We use Stanford Sentiment Treebank - 2 (SST2) dataset for this tutorial.


dataset = task.Dataset(name='sst_2',
                       train_path='/home/ubuntu/sst_2/train-61f1f238.json',
                       val_path='/home/ubuntu/sst_2/dev-65511587.json',
                       data_format='json')

################################################################
# We then will use ``autogluon.Nets`` and ``autogluon.Optimizers`` as
# examples to show the usage of auto objects. The remaining auto objects
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
# net_list = [task.model_zoo.get_model('standard_lstm_lm_200'), # TODO: pretrained and pretrained_dataset would be supported
#             task.model_zoo.get_model('awd_lstm_lm_1150')]

# method 2 (easy and less flexiable): specify the net_list using model name
net_list = ['standard_lstm_lm_200',
            'awd_lstm_lm_1150']

# default net list for text classification would be overwritten
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
#                             ag.optim.get_optim('ftml')])

# method 2 (easy and less flexiable): specify the optim_list using get_model
optimizers = ag.Optimizers(['sgd', 'ftml'])

print(optimizers)


################################################################
# Create AutoGluon Fit - Put all together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


stop_criterion = {
    'time_limits': 1*60*60,
    'max_metric': 1.0,
    'max_trial_count': 2
}

resources_per_trial = {
    'max_num_gpus': 1, # set this to 0 if running on CPU
    'max_num_cpus': 4,
    'max_training_epochs': 3
}

results = task.fit(dataset,
                   nets,
                   optimizers,
                   stop_criterion=stop_criterion,
                   resources_per_trial=resources_per_trial)

################################################################
# The best accuracy is:


print('%.2f acc' % (results.metric * 100))

################################################################
# The best associated configuration is:

print(results.config)

################################################################
# Total time cost is:


print('%.2f s' % results.time)


# ################################################################
# # Use Search Algorithm
# # ~~~~~~~~~~~~~~~~~~~~
# #
# # ``autogluon.searcher`` will support both basic and SOTA searchers for
# # both hyper-parameter optimization and architecture search. We now
# # support random search. The default is using searcher is random searcher.
#
#
# # cs is CS.ConfigurationSpace() where import ConfigSpace as CS, this is just example code;
# # in practice, this is in fit function, and cs should not be None
# cs = None
# searcher = ag.searcher.RandomSampling(cs)
#
# print(searcher)
#
# ################################################################
# # Or simply use string name:
# #
#
# searcher = 'random'
#
# print(searcher)
#
# ################################################################
# # Use Trial Scheduler
# # ~~~~~~~~~~~~~~~~~~~
# #
# # ``ag.scheduler`` supports scheduling trials in serial order and with
# # early stopping.
# #
# # We support basic FIFO scheduler.
# #
#
#
# # this is just example code; in practice, this is in fit function
# savedir = 'checkpoint/demo.ag'
#
# trial_scheduler = ag.scheduler.FIFO_Scheduler(
#                 task.pipeline.train_text_classification,
#                 None,
#                 {
#                     'num_cpus': 4,
#                     'num_gpus': 4,
#                 },
#                 searcher,
#                 checkpoint=savedir)
#
# print(trial_scheduler)
#
# ################################################################
# # We also support Hyperband which is an early stopping mechanism.
#
#
# # this is just example code; in practice, this is in fit function
# trial_scheduler = ag.scheduler.Hyperband_Scheduler(
#                 task.pipeline.train_text_classification,
#                 None,
#                 {
#                     'num_cpus': 4,
#                     'num_gpus': 4,
#                 },
#                 searcher,
#                 time_attr='epoch',
#                 reward_attr='accuracy',
#                 max_t=10,
#                 grace_period=1,
#                 checkpoint=savedir)
#
# print(trial_scheduler)
#
# ################################################################
# # Resume Fit and Checkpointer
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # We use the resume and checkpoint dir in the scheduler.
#
#
# savedir = 'checkpoint/demo.ag'
# resume = False
#
#
# trial_scheduler = ag.scheduler.Hyperband_Scheduler(
#                 task.pipeline.train_text_classification,
#                 None,
#                 {
#                     'num_cpus': 4,
#                     'num_gpus': 4,
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
#
# ################################################################
# # Or simply specify the trial scheduler with the string name:
#
# trial_scheduler = 'hyperband'
#
# ################################################################
# # Visualize Using Tensor/MXBoard
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
# # We could visualize the traing curve using Tensorboad or MXboard. To
# # start the Tensorboard or MXboard, please use:
# #
# # ``tensorboard --logdir=./checkpoint/demo/logs --host=127.0.0.1 --port=8889``
# #
# # An example is shown below.
# #
# # Create Stop Criterion
# # ~~~~~~~~~~~~~~~~~~~~~
# #
# # ``autogluon`` supports overall automatic constraints in
# # ``stop_criterion``.
# #
#
# stop_criterion = {
#     'time_limits': 1*60*60,
#     'max_metric': 0.80, #if you know, otherwise use the default 1.0
#     'max_trial_count': 2
# }
#
# ################################################################
# # Create Resources Per Trial
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
# # ``autogluon`` supports constraints for each trial
# # ``in resource_per_trial``.
# #
#
#
# resources_per_trial = {
#     'max_num_gpus': 4,
#     'max_num_cpus': 4,
#     'max_training_epochs': 1
# }
#
# ################################################################
# # Create AutoGluon Fit with Full Capacity
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# results = task.fit(dataset,
#                   nets,
#                   optimizers,
#                   searcher=searcher,
#                   trial_scheduler='fifo',
#                   resume=resume,
#                   savedir=savedir,
#                   stop_criterion=stop_criterion,
#                   resources_per_trial=resources_per_trial,
#                   demo=True) # only set demo=True when running on no GPU machine
#
# ################################################################
# # The best accuracy is
#
# print('%.2f acc' % (results.metric * 100))
#
# ################################################################
# # The associated best configuration is:
#
#
# print(results.config)
#
# ################################################################
# # The total time cost is:
#
# print('%.2f s' % results.time)
#
# ################################################################
# # Resume AutoGluon Fit
# # ~~~~~~~~~~~~~~~~~~~~
# #
# # We could resume the previous training for more epochs to achieve better
# # results. Similarly, we could also increase ``max_trial_count`` for
# # better results.
# #
# # Here we increase the ``max_training_epochs`` from 1 to 3,
# # ``max_trial_count`` from 2 to 3, and set ``resume = True`` which will
# # load the checking point in the savedir.
#
#
# stop_criterion = {
#     'time_limits': 1*60*60,
#     'max_metric': 0.80,
#     'max_trial_count': 3
# }
#
# resources_per_trial = {
#     'max_num_gpus': 4,
#     'max_num_cpus': 4,
#     'max_training_epochs': 3
# }
#
# resume = True
#
# results = task.fit(dataset,
#                   nets,
#                   optimizers,
#                   searcher=searcher,
#                   trial_scheduler='fifo',
#                   resume=resume,
#                   savedir=savedir,
#                   stop_criterion=stop_criterion,
#                   resources_per_trial=resources_per_trial,
#                   demo=True)
#
# ################################################################
# # The best accuracy is
#
#
# print('%.2f acc' % (results.metric * 100))
#
# ################################################################
# # The associated best configuration is:
#
#
# print(results.config)
#
# ################################################################
# # The total time cost is:
#
#
# print('%.2f s' % results.time)
#
# ################################################################
# # Refereces
# # ---------
# #
# # code: https://code.amazon.com/packages/AutoGluon/trees/heads/mainline

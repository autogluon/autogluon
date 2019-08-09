# TODO: add cross validation, final fit and predict, example running results


"""2. Image Classification - Advanced
============================================

In the following\ *, we use Image Classification as a running example*
to illustrate the usage of AutoGluon’s main APIs.
Different from the last demo, we focus how to customize autogluon ``Dataset``,
``Nets``, ``Optimizers``, ``Searcher`` and ``Scheduler``.
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

################################################################
# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use a small subset of `Shopee-IET` dataset prepared in the data preparation section.


dataset = task.Dataset(name='shopeeiet', train_path='data/train', val_path='data/val')

################################################################
# We then will use ``autogluon.Nets`` and ``autogluon.Optimizers`` as
# examples to show the usage of auto objects. The remainining auto objects
# are using default value.
################################################################
# Before that, let's first understand the `Space` object in AutoGluon.
#
# Create AutoGluon Space
# ~~~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon.space`` is a search space containing a set of configuration
# candidates. We provide three basic space types.
#
# -  Categorical Space


list_space = ag.space.List('listspace', ['0', '1', '2'])
print(list_space)

################################################################
# -  Linear Space


linear_space = ag.space.Linear('linspace', 0, 10)
print(linear_space)

################################################################
# -  Log Space


log_space = ag.space.Log('logspace', 10**-10, 10**-1)
print(log_space)

################################################################
# -  An Example of Random Sample from the Combined Space


print(ag.space.sample_configuration([list_space, linear_space, log_space]))

################################################################
# Create AutoGluon Nets
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon.Nets`` is a list of auto networks, and allows search for the
# best net
#
# -  from a list of provided (or default) networks
# -  by choosing the best architecture regarding to each auto net.
#

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

# method 1: using the task-specific default optimizer configuration.
optimizers_default = ag.Optimizers(['sgd', 'adam'])

# method 2: customize the hyperparamters of optimizer in the search space.
adam_opt = ag.optims.Adam(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
                          wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
sgd_opt = ag.optims.SGD(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
                        momentum=ag.space.Linear('momentum', 0.85, 0.95),
                        wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
optimizers = ag.Optimizers([adam_opt, sgd_opt])

print(optimizers)

################################################################
# Use Search Algorithm
# ~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon.searcher`` will support both basic and SOTA searchers for
# both hyper-parameter optimization and architecture search. We now
# support random search. The default searcher is random searcher.
# We can simply use string name to specify the searcher:

searcher = 'random'


################################################################
# Use Trial Scheduler
# ~~~~~~~~~~~~~~~~~~~
#
# ``ag.scheduler`` supports scheduling trials in serial order and with
# early stopping.
#
# We support basic FIFO scheduler and early stopping scheduler: Hyperband.
# We can simply use string name to specify the scheduler:

trial_scheduler = 'fifo'

################################################################
# We use the resume and checkpoint dir in the scheduler.

savedir = 'checkpoint/demo.ag'
resume = False

################################################################
# Create AutoGluon Fit - Put all together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's first set the customized stop criterion.

time_limits = 1*60*60
max_metric = 1.0
max_trial_count = 4

stop_criterion = {
    'time_limits': time_limits,
    'max_metric': max_metric,
    'max_trial_count': max_trial_count
}

################################################################
# Let's then set the customized resources per trial.
# We use `demo = True` for showing results faster.

max_num_gpus = 1
max_num_cpus = 4
max_training_epochs = 2
demo = True

resources_per_trial = {
    'max_num_gpus': max_num_gpus,
    'max_num_cpus': max_num_cpus,
    'max_training_epochs': max_training_epochs
}

results = task.fit(dataset,
                   nets,
                   optimizers,
                   searcher=searcher,
                   trial_scheduler=trial_scheduler,
                   resume=resume,
                   savedir=savedir,
                   stop_criterion=stop_criterion,
                   resources_per_trial=resources_per_trial,
                   demo=demo)

################################################################
# The search space is:


print(results.metadata)

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
# Resume AutoGluon Fit
# ~~~~~~~~~~~~~~~~~~~~
#
# We could resume the previous training for more epochs to achieve better
# results. Similarly, we could also increase ``max_trial_count`` for
# better results.
#
# Here we increase the ``max_training_epochs`` from 2 to 3,
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
                   trial_scheduler=trial_scheduler,
                   resume=resume,
                   savedir=savedir,
                   stop_criterion=stop_criterion,
                   resources_per_trial=resources_per_trial,
                   demo=demo)

################################################################
# The search space is:


print(results.metadata)

################################################################
# The best accuracy is


print('%.2f acc' % (results.metric * 100))

################################################################
# The associated best configuration is:


print(results.config)

################################################################
# The total time cost is:


print('%.2f s' % results.time)

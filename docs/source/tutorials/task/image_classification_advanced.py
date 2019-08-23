"""3. Image Classification - Advanced
============================================

In the following\ *, we use Image Classification as a running example*
to illustrate the usage of AutoGluon’s main APIs.
We focus on how to customize autogluon ``Dataset``,
``Nets``, ``Optimizers``, ``Searcher`` and ``Scheduler``.
"""
from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)

################################################################
# We first introduce the basic configuration ``autogluon.space``, which is
# used to represent the search space of each task component, we will then
# go through each component, including
#
# -  ``autogluon.Dataset``
# -  ``autogluon.Nets``
# -  ``autogluon.Optimizers``
# -  ``autogluon.Losses``
# -  ``autogluon.Metrics``
#
# and finally put all together to ``fit`` to generate the best result.

################################################################
# Create AutoGluon Space
# ~~~~~~~~~~~~~~~~~~~~~~~
# Let's first understand the `Space` object in AutoGluon.
#
# ``autogluon.space`` is a search space containing a set of configuration
# candidates. We provide three basic space types.
#
# -  Categorical Space

import autogluon as ag
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
# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use a small subset of `Shopee-IET` dataset prepared in the data preparation section.


dataset = task.Dataset(name='shopeeiet', train_path='data/train', test_path='data/test')

################################################################
# We then will use ``autogluon.Nets`` and ``autogluon.Optimizers`` as
# examples to show the usage of auto objects. The remaining auto objects
# are using default value.

################################################################
# Create AutoGluon Nets
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``autogluon.Nets`` is a list of networks, and allows search for the
# best network from a list of provided (or default) networks by choosing
# the best architecture regarding to each network.

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
# allows search for the best optimization algorithm from a list of provided
# (or default) optimizers by choosing the best hyper-parameters regarding to each
# optimizer.

# method 1: using the task-specific default optimizer configuration.
optimizers_default = ag.Optimizers(['sgd', 'adam'])

# method 2: customize the hyper-parameters of optimizer in the search space.
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
# ``autogluon.scheduler`` supports scheduling trials in serial order and with
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

time_limits = 5*60
max_metric = 1.0
num_trials = 8
stop_criterion = {
    'time_limits': time_limits,
    'max_metric': max_metric,
    'num_trials': num_trials
}

################################################################
# Let's then set the customized resources per trial.

num_gpus = 1
num_training_epochs = 10
resources_per_trial = {
    'num_gpus': num_gpus,
    'num_training_epochs': num_training_epochs
}

results = task.fit(dataset,
                   nets,
                   optimizers,
                   searcher=searcher,
                   trial_scheduler=trial_scheduler,
                   resume=resume,
                   savedir=savedir,
                   stop_criterion=stop_criterion,
                   resources_per_trial=resources_per_trial)

################################################################
# The best Top-1 accuracy on the validation set is:

print('Top-1 acc: %.3f' % results.metric)

###############################################################################
# We could test the model results on the test data.

test_acc = task.evaluate()
print('Top-1 test acc: %.3f' % test_acc)

###############################################################################
# We could select an example image to predict the label and probability.

image = './data/test/BabyBibs/BabyBibs_1084.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))


################################################################
# Resume AutoGluon Fit
# ~~~~~~~~~~~~~~~~~~~~
#
# We could resume the previous training for more epochs to achieve better
# results. Similarly, we could also increase ``num_trials`` for
# better results.
#
# Here we increase the ``num_training_epochs`` from 10 to 12,
# ``num_trials`` from 8 to 10, and set ``resume = True`` which will
# load the checking point in the savedir.

num_trials = 10
stop_criterion = {
    'time_limits': time_limits,
    'max_metric': max_metric,
    'num_trials': num_trials
}

num_training_epochs = 12
resources_per_trial = {
    'num_gpus': num_gpus,
    'num_training_epochs': num_training_epochs
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
                   resources_per_trial=resources_per_trial)

################################################################
# The best Top-1 accuracy on the validation set is:

print('Top-1 acc: %.3f' % results.metric)

###############################################################################
# We could test the model results on the test data.

test_acc = task.evaluate()
print('Top-1 test acc: %.3f' % test_acc)

###############################################################################
# We could select an example image to predict the label and probability.

image = './data/test/BabyBibs/BabyBibs_1084.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))

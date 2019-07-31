'''
AutoGluon Text Classification - Quick Start
============================================

AutoGluon aims to provide automatic machine learning (Auto ML) support
for MXNet and Gluon. AutoGluon focuses on automatic deep learning (Auto
DL). AutoGluon targets:

-  *Beginners* are 70~80% of the customers who would be interested
   in AutoGluon. The basic Auto ML scenario: customers have a
   traditional machine learning task by hand, provide own raw data,
   watch the search process, and finally obtain a good quality model.
   The beginners include but not limited to engineers and students, who
   are generally new to machine learning.
-  *Advanced users* aim to own full control and access to the Auto ML
   overall process as well as each important component, such as
   constructing own networks, metrics, losses, optimizers, searcher and
   trial scheduler. The advanced users could potentially have more
   specified constraints regarding to the automatic searching procedure.
   The advanced users include but not limited to experienced machine
   learning researchers and engineers.
-  *Contributors:* Contributors are Advanced users who will create
   strategies that are useful for beginners either extending to new
   datasets, new domains, new algorithms or bringing state of art
   results to save time and effort.

The AutoGluon's design principles are:

-  *Easy to use:* Deep learning framework users could use AutoGluon
   almost right away. The only usage difference between AutoGluon and
   Gluon is that: rather than providing a fixed value to different deep
   learning components, we enable a searchable range to let Auto ML
   decides which are the best, whereas all the major APIsâ€™ usage stays
   the same.
-  *Easy to extend:* From user perspective, we organize the AutoGluon
   by tasks, users could easily use all the task specific components,
   such as data preprocessing, model zoo, metrics and losses, so that
   adding a new task could very straightforward. In this way, advanced
   ML tasks, such as GAN ,could be easily incorporated by providing a
   new task module. From system perspective, multiple back-ends could be
   used since the front-end are designed to be separate from the
   backends, this could be beneficial to extend to production-level Auto
   ML.

The AutoGluon's overall system design is as below:

.. image:: ../../../_static/img/autogluon_overview.png

In the following, we use Text Classification as a running example
to illustrate the usage of AutoGluon's main APIs.

Preparation
-----------

Install AutoGluon
~~~~~~~~~~~~~~~~~

TODO: prepare wheel

.. code:: bash

    git clone ssh://git.amazon.com/pkg/AutoGluon
    cd AutoGluon
    python setup.py install

EC2
~~~

TODO: prepare EC2 machine

Import Task
~~~~~~~~~~~

We are using text classification as an example in this notebook.


'''

import warnings
warnings.filterwarnings("ignore")

from autogluon import text_classification as task

import logging
logging.basicConfig(level=logging.INFO)


################################################################
# A Quick Text Classification Example
# ------------------------------------
#
# We first show the most basic usage by first creating a dataset and then
# fiting the dataset to generate the results with the text classification example.
# We will use Stanford Sentiment Treebank Dataset for this tutorial.
#
# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use Stanford Sentiment Treebank - 2 (SST2) dataset.


dataset = task.Dataset(name='sst_2',
                       train_path='/home/ubuntu/sst_2/train-61f1f238.json',
                       val_path='/home/ubuntu/sst_2/dev-65511587.json')

################################################################
# The constructed dataset contains the ``gluon.data.DataLoader`` for training and validation datasets.

################################################################
# Then we will use the default configuration of the text classification
# to generate: \* Best result of the search in terms of accuracy \*
# Get the best configuration corresponding to the best result obtained.
#
# To acheive this, we are using ``fit`` function to generate the above
# results based on the datasets.
#
# The default configruation is based on ``max_trial_count=2`` and
# ``max_training_epochs=3``. If running on no GPU environment, please set
# ``demo=True`` in the ``fit``. The process would approximately cost one
# and half minutes. If want to watch the ``fit``, we default provide
# Tensorboad to visualize the process. Please type
# ``tensorboard --logdir=./checkpoint/exp1/logs --host=127.0.0.1 --port=8888``
# in the command.

# TODO: add demo arg
results = task.fit(dataset)

################################################################
# The best accuracy is:


print('%.2f acc' % (results.metric * 100))

################################################################
# The associated best configuration is:


print(results.config)

################################################################
# Total time cost is:


print('%.2f s' % results.time)

################################################################
# A Step-by-step Text Classification Example
# -------------------------------------------
# TODO: split out an advanced user
#
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

################################################################
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
                   resources_per_trial=resources_per_trial,
                   demo=False) # demo=True is recommened when running on no GPU machine

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
# Refereces
# ---------
#
# code: https://code.amazon.com/packages/AutoGluon/trees/heads/mainline

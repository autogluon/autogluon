'''
1. AutoGluon Text Classification - Quick Start
============================================

In the following, we use Text Classification as a running example
to illustrate the usage of AutoGluon's main APIs.

'''

import warnings
warnings.filterwarnings("ignore")

from autogluon import text_classification as task

import logging
logging.basicConfig(level=logging.INFO)


################################################################
# We first show the most basic usage by first creating a dataset and then
# fitting the dataset to generate the results with the text classification example.
#
# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use Stanford Sentiment Treebank - 2 (SST2) dataset for this tutorial.


dataset = task.Dataset(name='sst_2',
                       train_path='/home/ubuntu/sst_2/train-61f1f238.json',
                       val_path='/home/ubuntu/sst_2/dev-65511587.json',
                       data_format='json')

################################################################
# Then we will use the default configuration for text classification
# task to generate: \* Best result of the search in terms of accuracy \*
# Get the best configuration corresponding to the best result obtained.
#
# To achieve this, we are using ``fit`` method to generate the above
# results based on the datasets.
#
# The default configruation is based on ``max_trial_count=2`` and
# ``max_training_epochs=3``. The process would approximately cost one
# and half minutes. If want to watch the ``fit``, we default provide
# Tensorboad to visualize the process. Please type
# ``tensorboard --logdir=./checkpoint/exp1/logs --host=127.0.0.1 --port=8888``
# in the command.

max_num_gpus = 1
max_num_cpus = 4
max_training_epochs = 2

resources_per_trial = {
    'max_num_gpus': max_num_gpus, # set this to more than 1 if you have a multi GPU machine for speedup.
    'max_num_cpus': max_num_cpus,
    'max_training_epochs': max_training_epochs
}

results = task.fit(dataset, resources_per_trial=resources_per_trial)

################################################################
# The best accuracy is:


print('%.2f acc' % (results.metric * 100))

################################################################
# The associated best configuration is:


print(results.config)

################################################################
# Total time cost is:


print('%.2f s' % results.time)

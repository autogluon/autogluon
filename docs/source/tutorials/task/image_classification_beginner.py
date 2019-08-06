"""1. AutoGluon Image Classification - Quick Start
====================================================

In the following\ *, we use Image Classification as a running example*
to illustrate the usage of AutoGluon’s main APIs.
"""

import warnings
warnings.filterwarnings("ignore")

from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)


################################################################
# We first show the most basic usage by first creating a dataset and then
# fiting the dataset to generate the results with the image classification
# example.
#
# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use CIFAR10 for image classfication for demo purpose.

dataset = task.Dataset(name='CIFAR10') # case insentive

print(dataset) # show a quick summary of the dataset, e.g. #example for train, #classes

################################################################
# The constructed dataset contains the CIFAR10 training and validation
# datasets.


# dataset.train[0] # access the first example
# dataset.val[-2:] # access the last 2 validation examples

################################################################
# Then we will use the default configuration of the image classification
# to generate: \* Best result of the search in terms of accuracy \*
# According best configuration regarding to the best result
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

max_num_gpus = 1
max_num_cpus = 4
max_training_epochs = 2
demo = True

resources_per_trial = {
    'max_num_gpus': max_num_gpus, # set this to more than 1 if you have GPU machine to run more efficiently.
    'max_num_cpus': max_num_cpus,
    'max_training_epochs': max_training_epochs
}

results = task.fit(dataset,
                   resources_per_trial=resources_per_trial, demo=demo)

################################################################
# The best accuracy is:


print('%.2f acc' % (results.metric * 100))

################################################################
# The associated best configuration is:


print(results.config)

################################################################
# Total time cost is:


print('%.2f s' % results.time)


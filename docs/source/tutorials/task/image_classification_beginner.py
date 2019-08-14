"""1. Image Classification - Quick Start
====================================================

In the following\ *, we use Image Classification as a running example*
to illustrate the usage of AutoGluon’s main APIs.
"""

# TODO: remove the warnings
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
# We use a small subset of `Shopee-IET dataset<https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data>`__ from Kaggle for image classfication.
# Please click `link<http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip>`__ to download the subset.

# TODO: finalize data link
# import subprocess
# get_dataset = 'wget http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip && unzip data.zip'
# subprocess.run(get_dataset.split())
#
# ################################################################
# # We then construct the dataset using the downloaded dataset.
#
dataset = task.Dataset(name='shopeeiet', train_path='data/train', val_path='data/val') # case insentive

# dataset = task.Dataset(name='cifar10')

print(dataset) # show a quick summary of the dataset, e.g. #example for train, #classes


################################################################
# Create AutoGluon Fit
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Then we will use the default configuration of the image classification
# to generate: \* Best result of the search in terms of accuracy \*
# According best configuration regarding to the best result
#
# To acheive this, we are using ``fit`` function to generate the above
# results based on the datasets.
#
# Let's first set the resources would be used per trial based on the situation we have one GPU.
# We are using ``demo = True`` to quickly achieve the results.

# #TODO: use num_gpus, remove max_num_cpus, use num_training_epochs, remove demo
# max_num_gpus = 1
# max_num_cpus = 4
# max_training_epochs = 2
# demo = True
#
# #TODO: only task.fit now resources_per_trial in the fit
# resources_per_trial = {
#     'max_num_gpus': max_num_gpus, # set this to more than 1 if you have GPU machine to run more efficiently.
#     'max_num_cpus': max_num_cpus,
#     'max_training_epochs': max_training_epochs
# }

results = task.fit(dataset)

#TODO: only show results.metric (but specify the top-1 accuracy)

################################################################
# The best Top-1 accuracy is:

print('%.2f acc' % (results.metric * 100))

test_acc, test_loss = task.evaluate()
print('test data %.2f acc, %.2f loss' % (test_acc * 100, test_loss))

image = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/plane-draw.jpeg'
ind, prob = task.predict(image)
print('The input picture is classified as %d, with probability %.3f.' %
      (ind.asscalar(), prob.asscalar()))


################################################################
# An example visulization using MXBoard:
# .. image:: ../../../_static/img/shopee_accuracy_curves_1.svg
#TODO: add evaluation task.predict() (have a default image, or upload an image)

#TODO: put the results.metadata, config, time in the decription at the end.


################################################################
# The associated best configuration is:
# print(results.config)

# ###############################################################
# Total time cost is:

print('%.2f s' % results.time)

################################################################
# The search space is:
#print(results.metadata)
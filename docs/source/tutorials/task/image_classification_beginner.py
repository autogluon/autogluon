"""1. Image Classification - Quick Start
====================================================

In the following\ *, we use Image Classification as a running example*
to illustrate the usage of AutoGluon’s main APIs.
"""

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
# We use a small subset of `Shopee-IET dataset <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data>`__ from Kaggle for image classfication.
# Please download the data subset via `data link <http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com>`__ .
# Then unzip it via:
# ::
#   unzip data.zip

dataset = task.Dataset(name='shopeeiet', train_path='data/train', val_path='data/val') # case insentive


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

results = task.fit(dataset)

################################################################
# The best Top-1 accuracy on the validation set is:

print('Top-1 acc: %.3f' % results.metric)

###############################################################################
# We could test the model results on the test data.

test_acc = task.evaluate()
print('Top-1 test acc: %.3f' % test_acc)

###############################################################################
# We could select an example image to predict the label and probability.

image = './data/val/BabyBibs/BabyBibs_1084.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))

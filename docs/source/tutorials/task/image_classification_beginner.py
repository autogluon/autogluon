"""1. Image Classification - Quick Start
====================================================

In the following, we use Image Classification as a running example
to illustrate the usage of AutoGluon’s main APIs.
"""

from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)

################################################################
# We first show the most basic usage by first creating a dataset and then
# fitting the dataset to generate the results with the image classification
# example.
#
# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use a small subset of `Shopee-IET dataset <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data>`__ from Kaggle for image classfication.
# Please download the data subset via `data link <http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip>`__ .
# Then unzip it via:
# ::
#   !wget http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip
#   !unzip data.zip
# After downloading the dataset, we will automatically construct the train/validation data split
# based on the provided train data, where the train data has 90% and validation data has 10% of the
# whole training data respectively.

dataset = task.Dataset(name='shopeeiet', train_path='data/train')


################################################################
# Create AutoGluon Fit
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Then we will use the default configuration of the image classification
# to generate best result of the search in terms of accuracy
# according to best configuration.
#
# To achieve this, we are using TODO `fit` function to generate the above
# results based on the datasets.
# Due to the time limitation for demo purpose,
# we specify the `time_limits`, `num_training_epochs` to small values.

#TODO
time_limits = 30
num_training_epochs = 10
results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)

################################################################
# The best config is selected based on the validation split
# from train data, the best model is finally fitted on full train data using the best config.
# The best Top-1 accuracy on the validation set is:

print('Top-1 val acc: %.3f' % results.metric)

###############################################################################
# We could construct the test dataset similarly with the train dataset.
# Test results are reported based on the final model.

test_dataset = task.Dataset(name='shopeeiet', test_path='data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)

###############################################################################
# We could select an example image to predict the label and probability.

image = './data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))

###############################################################################
# We show the best config regarding to the above results as below.

print('The best configuration is:')
print(results.config)

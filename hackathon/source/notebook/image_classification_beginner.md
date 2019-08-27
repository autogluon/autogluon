# Image Classification - Quick Start

Here we adopt the task of Image Classification as a running example to illustrate basic usage of AutoGluon’s main APIs.  
This task involves a few steps which we demonstrate how to easily execute in AutoGluon, sticking mainly with default options for simplicity. 

In this tutorial, we will: load image+label data into AutoGluon and use this data to train a convolutional neural network that can classify new images.  
AutoGluon will automatically tune various hyperparameters of our neural network, and rather than attempting to train a complex model from scratch on our data, AutoGluon will instead fine-tune a residual network that has already been pretrained on the large-scale ImageNet dataset.  
Although the ImageNet dataset involves entirely different images and classes, the idea here is that lower-level features captured in the representations of the pretrained ImageNet network (such as edge/texture detectors) are likely to remain useful for our own image dataset.  

We begin in Python by specifying `image_classification` as our task of interest:

```python
from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)
```


## Create AutoGluon Dataset

Our image classfication task will be based on a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle.  
Each image in this data depicts a clothing item and the corresponding label specifies its clothing sub-category (Our subset of the data contains the following possible labels: TODO: list the possible labels!).  
Note that we only use a small subset of the data to ensure quick runtimes in this tutorial; to obtain models that perform competitively will require using the full original dataset.  

We download the data subset from this [link](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip)
and unzip it via the following commands:

```python
import os
os.system('wget http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip')
os.system('unzip data.zip')
```

If the above command fails, just manually download the data from the link and unzip it yourself.  
Once the dataset resides on our machine, we load it into an AutoGluon `Dataset` object: 

```python
dataset = task.Dataset(name='shopeeiet', train_path='data/train')
```

In the above call, a train/validation data split is automatically constructed based on the provided data, where 90% of the images are used for training and 10% held-out for validation.  AutoGluon will automatically tune various hyperparameters of our neural network model in order to maximize classification performance on the validation data.  



## Use AutoGluon to fit models

Now, we train a neural network classifier using AutoGluon.  
While we stick with mostly default settings in this Beginner tutorial, the Advanced tutorial will cover various options that you can specify for greater control over the training process.  
With just a single call to AutoGluon's `fit` function, AutoGluon will train many models with different hyperparameter settings and return the best model.

However, neural network training can be quite time-costly.  
To ensure quick runtimes, we tell AutoGluon to obey strict limits: `num_training_epochs` specifies how much computational effort can be devoted to training any single network (ie. under a given setting of the hyperparameters), while `time_limits` specifies how much time `fit` has to return a model (ie. the total amount of time that may be devoted to exploring how well networks perform when trained under various hyperparameter settings).  
For demo purposes, we specify unreasonably small values for `time_limits`, `num_training_epochs` here:

```python
time_limits = 5*60
num_training_epochs = 10
results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

Within `fit`, the model with the best hyperparameter configuration is selected based on its validation accuracy after being trained on the data in the training split.  The best Top-1 accuracy achieved on the validation set is:

```python
print('Top-1 val acc: %.3f' % results.metric)
```

Within `fit`, this model is also finally fitted on our entire dataset (ie. merging training+validation) using the same optimal hyperparameter configuration. The resulting model can then be applied to classify new images.

We now construct a test dataset similarly as we did with the train dataset, and then evaluate the final model produced by `fit` on the test data:

```python
test_dataset = task.Dataset(name='shopeeiet', test_path='data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Given an example image, we can easily use our trained model to predict the label (and the conditional class-probability):

```python
image = './data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))
```

The `results` object returned by `fit` contains summaries describing various aspects of the training process.
For example, we can inspect the best hyperparameter configuration corresponding to the model which achieved the above results:

```python
print('The best configuration is:')
print(results.config)
```

TODO: We can also inspect how many models (with different hyperparameter settings) AutoGluon managed to try out within our specified `time_limits`:

```python
TODO: insert code snippet here to print() number of trials completed by scheduler
```
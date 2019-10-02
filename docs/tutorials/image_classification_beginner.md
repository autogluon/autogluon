# Image Classification - Quick Start
:label:`sec_imgquick`

We adopt the task of Image Classification as a running example to illustrate basic usage of AutoGluon’s main APIs. This task involves a few steps which we demonstrate how to get started with AutoGluon. 

In this tutorial, we will load images and the corresponding labels into AutoGluon and use this data to obtain a neural network that can classify new images. Different from traditional machine learning where we need to manually define the neural network, and specify the hyperparameters in the training process, with just a single call to `AutoGluon`'s `fit` function, AutoGluon will automatically train many models and thousands of different hyperparameter configurations regarding to the training process and return the best model.
Besides, you could easily specify for greater control over the training process such as providing the time limits for the training procedures and how many computation resource you want each training runs leverage, we will cover such advanced usage in :ref:`sec_imgadvanced`. 

We begin by specifying `image_classification` as our task of interest:

```{.python .input}
from autogluon import ImageClassification as task

import logging
logging.basicConfig(level=logging.INFO)
```


## Create AutoGluon Dataset

Our image classification task is based on a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle. Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`. Note that we only use a small subset of the data to ensure quick runtimes in this tutorial; to obtain models that perform competitively will require using the full original dataset, we will cover this in the next tutorial.  

We download the data subset from this [link](../data.zip)
and unzip it via the following commands:

```{.python .input}
import os
os.system('wget http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip')
os.system('unzip -o data.zip -d ~/')
```

Once the dataset resides on our machine, we load it into an AutoGluon `Dataset` object: 

```{.python .input}
dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')
```

In the above call, a train/validation data split is automatically constructed based on the provided data, where 90% of the images are used for training and 10% held-out for validation. AutoGluon will automatically tune various hyperparameters of our neural network models in order to maximize classification performance on the validation data.  



## Use AutoGluon to fit models

Now, we want to obtain a neural network classifier using AutoGluon. In the default configuration, rather than attempting to train complex models from scratch on our data, AutoGluon will instead fine-tune neural networks that have already been pretrained on ImageNet dataset. Although the ImageNet dataset involves entirely different images and classes, the idea here is that lower-level features captured in the representations of the pretrained ImageNet network (such as edge/texture detectors) are likely to remain useful for our own image dataset.  

While we stick with mostly default configurations in this Beginner tutorial, the Advanced tutorial will cover various options that you can specify for greater control over the training process. With just a single call to AutoGluon's `fit` function, AutoGluon will train many models with different hyperparameter configurations and return the best model.

However, neural network training can be quite time-costly. To ensure quick runtimes, we tell AutoGluon to obey strict limits: `num_training_epochs` specifies how much computational effort can be devoted to training any single network, while `time_limits` in seconds specifies how much time `fit` has to return a model. For demo purposes, we specify only small values for `time_limits`, `num_training_epochs`:

```{.python .input}
time_limits = 3*60 # 3mins
<<<<<<< HEAD
epochs = 10
results = task.fit(dataset,
                   time_limits=time_limits,
                   epochs=epochs)
=======
num_training_epochs = 10
results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
>>>>>>> c8b325866201574caeb688c623d02b23799a65fc
```

Within `fit`, the model with the best hyperparameter configuration is selected based on its validation accuracy after being trained on the data in the training split.  

The best Top-1 accuracy achieved on the validation set is:

```{.python .input}
print('Top-1 val acc: %.3f' % results.reward)
```

Within `fit`, this model is also finally fitted on our entire dataset (ie. merging training+validation) using the same optimal hyperparameter configuration. The resulting model is considered as final model to be applied to classify new images.

We now construct a test dataset similarly as we did with the train dataset, and then `evaluate` the final model produced by `fit` on the test data:

```{.python .input}
test_dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Given an example image, we can easily use the final model to `predict` the label (and the conditional class-probability):

```{.python .input}
image = '/home/ubuntu/data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.synsets[ind.asscalar()], prob.asscalar()))
```

The `results` object returned by `fit` contains summaries describing various aspects of the training process.
For example, we can inspect the best hyperparameter configuration corresponding to the final model which achieved the above results:

```{.python .input}
print('The best configuration is:')
print(results.config)
```

This configuration is used to generate the above results.

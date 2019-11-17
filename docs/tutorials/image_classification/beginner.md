# Image Classification - Quick Start
:label:`sec_imgquick`

We adopt the task of Image Classification as a running example to illustrate basic usage of AutoGluon’s main APIs. This task involves a few steps which we demonstrate how to get started with AutoGluon. 

In this tutorial, we will load images and the corresponding labels into AutoGluon and use this data to obtain a neural network that can classify new images. Different from traditional machine learning where we need to manually define the neural network, and specify the hyperparameters in the training process, with just a single call to `AutoGluon`'s `fit` function, AutoGluon will automatically train many models and thousands of different hyperparameter configurations regarding to the training process and return the best model.

We begin by specifying `image_classification` as our task of interest:

```{.python .input}
import autogluon as ag
from autogluon import ImageClassification as task
```

## Create AutoGluon Dataset

Our image classification task is based on a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle. Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`. Note that we only use a small subset of the data to ensure quick runtimes in this tutorial; to obtain models that perform competitively will require using the full original dataset, we will cover this in the next tutorial.  

We download the data subset and unzip it via the following commands:

```{.python .input}
filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
ag.unzip(filename)
```

Once the dataset resides on our machine, we load it into an AutoGluon `Dataset` object: 

```{.python .input}
dataset = task.Dataset(train_path='data/train')
```

## Use AutoGluon to fit models

Now, we want to obtain a neural network classifier using AutoGluon:

```{.python .input}
classifier = task.fit(dataset,
                      epochs=10,
                      ngpus_per_trial=1)
```

Within `fit`, the model with the best hyperparameter configuration is selected based on its validation accuracy after being trained on the data in the training split.  

The best Top-1 accuracy achieved on the validation set is:

```{.python .input}
print('Top-1 val acc: %.3f' % classifier.results[classifier.results['reward_attr']])
```

Within `fit`, this model is also finally fitted on our entire dataset (ie. merging training+validation) using the same optimal hyperparameter configuration. The resulting model is considered as final model to be applied to classify new images.

We now evaluate the classifier on a test dataset:

```{.python .input}
test_dataset = task.Dataset(test_path='data/test')
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Given an example image, we can easily use the final model to `predict` the label (and the conditional class-probability):

```{.python .input}
image = 'data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = classifier.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.init().classes[ind.asscalar()], prob.asscalar()))
```

The `classifier.results` contains summaries describing various aspects of the training process.
For example, we can inspect the best hyperparameter configuration corresponding to the final model which achieved the above results:

```{.python .input}
print('The best configuration is:')
print(classifier.results['best_config'])
```

Finish and exit:
```{.python .input}
ag.done()
```

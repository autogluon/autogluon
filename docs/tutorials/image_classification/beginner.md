# Image Classification - Quick Start
:label:`sec_imgquick`

In this quick start, we'll use the task of image classification to illustrate how to use AutoGluon’s APIs. 

In this tutorial, we load images and the corresponding labels into AutoGluon and use this data to obtain a neural network that can classify new images. This is different from traditional machine learning where we need to manually define the neural network and then specify the hyperparameters in the training process. Instead, with just a single call to AutoGluon's [fit](/api/autogluon.task.html#autogluon.task.ImageClassification.fit) function, AutoGluon automatically trains many models with different hyperparameter configurations and returns the model that achieved the highest level of accuracy.

We begin by specifying [ImageClassification](/api/autogluon.task.html#autogluon.task.ImageClassification) as our task of interest as follows:

```{.python .input}
import autogluon as ag
from autogluon import ImageClassification as task
```

## Create AutoGluon Dataset

For demonstration purposes, we use a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle.
Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

We download the data subset and unzip it using the following commands:

```{.python .input}
filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
ag.unzip(filename)
```

After the dataset is downloaded, we load it into a [`Dataset`](/api/autogluon.task.html#autogluon.task.ImageClassification.Dataset) object: 

```{.python .input}
dataset = task.Dataset('data/train')
```

Load the test dataset as follows:

```{.python .input}
test_dataset = task.Dataset('data/test', train=False)
```

If you don't have a GPU, change the dataset to 'FashionMNIST' to ensure that it doesn't take too long to run:

```{.python .input}
if ag.get_gpu_count() == 0:
    dataset = task.Dataset(name='FashionMNIST')
    test_dataset = task.Dataset(name='FashionMNIST', train=False)
```

## Use AutoGluon to Fit Models

Now, we fit a classifier using AutoGluon as follows:

```{.python .input}
classifier = task.fit(dataset,
                      epochs=5,
                      ngpus_per_trial=1,
                      verbose=False)
```

Within `fit`, the dataset is automatically split into training and validation sets.
The model with the best hyperparameter configuration is selected based on its performance on the validation set.
The best model is finally retrained on our entire dataset (i.e., merging training+validation) using the best configuration.

The best Top-1 accuracy achieved on the validation set is as follows:

```{.python .input}
print('Top-1 val acc: %.3f' % classifier.results['best_reward'])
```

## Predict on a New Image

Given an example image, we can easily use the final model to `predict` the label (and the conditional class-probability):

```{.python .input}
# skip this if training FashionMNIST on CPU.
if ag.get_gpu_count() > 0:
    image = 'data/test/BabyShirt/BabyShirt_323.jpg'
    ind, prob, _ = classifier.predict(image, plot=True)

    print('The input picture is classified as [%s], with probability %.2f.' %
          (dataset.init().classes[ind.asscalar()], prob.asscalar()))

    image = 'data/test/womenchiffontop/womenchiffontop_184.jpg'
    ind, prob, _ = classifier.predict(image, plot=True)

    print('The input picture is classified as [%s], with probability %.2f.' %
          (dataset.init().classes[ind.asscalar()], prob.asscalar()))
```

## Evaluate on Test Dataset

We now evaluate the classifier on a test dataset.

The validation and test top-1 accuracy are:

```{.python .input}
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

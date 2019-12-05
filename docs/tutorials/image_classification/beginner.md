# Image Classification - Quick Start
:label:`sec_imgquick`

We adopt the task of Image Classification as a running example to illustrate basic usage of AutoGluon’s main APIs. This task involves a few steps which we demonstrate how to get started with AutoGluon. 

In this tutorial, we will load images and the corresponding labels into AutoGluon and use this data to obtain a neural network that can classify new images. Different from traditional machine learning where we need to manually define the neural network, and specify the hyperparameters in the training process, with just a single call to `AutoGluon`'s `fit` function, AutoGluon will automatically train many models with different hyperparameter configurations and return the best model.

We begin by specifying `image_classification` as our task of interest:

```{.python .input}
import autogluon as ag
from autogluon import ImageClassification as task
```

## Create AutoGluon Dataset

<<<<<<< HEAD
Our image classification task is based on a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle. Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
=======
For demo purpose, we will use a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle.
Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
>>>>>>> origin/master
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

We download the data subset and unzip it via the following commands:

```{.python .input}
filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
ag.unzip(filename)
```

Once the dataset resides on our machine, we load it into a `Dataset` object: 

```{.python .input}
dataset = task.Dataset('data/train')
```

<<<<<<< HEAD
=======
Load the test dataset:

```{.python .input}
test_dataset = task.Dataset('data/test', train=False)
```

If you do not do not have a GPU, we will change the dataset to 'FashionMNIST' for demo purpose.
Otherwise, it will take forever to run:

```{.python .input}
if ag.get_gpu_count() == 0:
    dataset = task.Dataset(name='FashionMNIST')
    test_dataset = task.Dataset(name='FashionMNIST', train=False)
```

>>>>>>> origin/master
## Use AutoGluon to Fit Models

Now, we want to fit a classifier using AutoGluon:

```{.python .input}
classifier = task.fit(dataset,
                      epochs=10,
                      ngpus_per_trial=1,
                      verbose=True)
```

Within `fit`, the dataset is automatically splited into training and validation sets.
The model with the best hyperparameter configuration is selected based on its performance on validation set.
The best model is finally retrained on our entire dataset (ie. merging training+validation) using the best configuration.

The best Top-1 accuracy achieved on the validation set is:

```{.python .input}
print('Top-1 val acc: %.3f' % classifier.results['best_reward'])
```

## Predict on A New Image

Given an example image, we can easily use the final model to `predict` the label (and the conditional class-probability):

```{.python .input}
<<<<<<< HEAD
image = 'data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = classifier.predict(image)

print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.init().classes[ind.asscalar()], prob.asscalar()))
```

## Evaluate on Test Dataset

We now evaluate the classifier on a test dataset:

Load the test dataset:

```{.python .input}
test_dataset = task.Dataset('data/test', train=False)
```

The validation and test top-1 accuracy are:

```{.python .input}
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

=======
# skip this if training FashionMNIST on CPU.
if ag.get_gpu_count() > 0:
    image = 'data/test/BabyShirt/BabyShirt_323.jpg'
    ind, prob = classifier.predict(image)

    print('The input picture is classified as [%s], with probability %.2f.' %
          (dataset.init().classes[ind.asscalar()], prob.asscalar()))
```

## Evaluate on Test Dataset

We now evaluate the classifier on a test dataset:


The validation and test top-1 accuracy are:
>>>>>>> origin/master

```{.python .input}
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

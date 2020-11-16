# Image Classification - Quick Start
:label:`sec_imgquick`

In this quick start, we'll use the task of image classification to illustrate how to use AutoGluon’s APIs.

In this tutorial, we load images and the corresponding labels into AutoGluon and use this data to obtain a neural network that can classify new images. This is different from traditional machine learning where we need to manually define the neural network and then specify the hyperparameters in the training process. Instead, with just a single call to AutoGluon's [fit](/api/autogluon.task.html#autogluon.vision.ImageClassification.fit) function, AutoGluon automatically trains many models with different hyperparameter configurations and returns the model that achieved the highest level of accuracy.

We begin by specifying [ImageClassification](/api/autogluon.task.html#autogluon.vision.ImageClassification) as our task of interest as follows:

```{.python .input}
import autogluon.core as ag
from autogluon.vision import ImageClassification as Task
```

## Create AutoGluon Dataset

For demonstration purposes, we use a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle.
Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

We can load a dataset by downloading a url data automatically:

```{.python .input}
train_dataset, _, test_dataset = Task.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
print(train_dataset)
```

## Use AutoGluon to Fit Models

Now, we fit a classifier using AutoGluon as follows:

```{.python .input}
task = Task()
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
classifier = task.fit(train_dataset, epochs=2)  # you can trust the default config, we reduce the # epoch to save some build time
```

Within `fit`, the dataset is automatically split into training and validation sets.
The model with the best hyperparameter configuration is selected based on its performance on the validation set.
The best model is finally retrained on our entire dataset (i.e., merging training+validation) using the best configuration.

The best Top-1 accuracy achieved on the validation set is as follows:

```{.python .input}
fit_result = task.fit_summary()
print('Top-1 train acc: %.3f, val acc: %.3f' %(fit_result['train_acc'], fit_result['valid_acc']))
```

## Predict on a New Image

Given an example image, we can easily use the final model to `predict` the label (and the conditional class-probability denoted as `score`):

```{.python .input}
image_path = test_dataset.iloc[0]['image']
result = classifier.predict(image_path)

print(result)
```

You can also feed in multiple images all together, let's use images in test dataset as an example:
```{.python .input}
bulk_result = classifier.predict(test_dataset)

print(bulk_result)
```


## Evaluate on Test Dataset

You can evaluate the classifier on a test dataset rather than retrieving the predictions.

The validation and test top-1 accuracy are:

```{.python .input}
test_acc, _ = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Save and load classifiers

You can directly save the instances of classifiers:

```{.python .input}
filename = 'classifier.ag'
classifier.save(filename)
classifier_loaded = Task.load(filename)
# use classifier_loaded as usual
result = classifier_loaded.predict(image_path)
print(result)
```

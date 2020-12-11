# Image Prediction - Quick Start
:label:`sec_imgquick`

In this quick start, we'll use the task of image classification to illustrate how to use AutoGluon’s APIs.

In this tutorial, we load images and the corresponding labels into AutoGluon and use this data to obtain a neural network that can classify new images. This is different from traditional machine learning where we need to manually define the neural network and then specify the hyperparameters in the training process. Instead, with just a single call to AutoGluon's [fit](/api/autogluon.task.html#autogluon.vision.ImagePredictor.fit) function, AutoGluon automatically trains many models with different hyperparameter configurations and returns the model that achieved the highest level of accuracy.

We begin by specifying [ImagePredictor](/api/autogluon.task.html#autogluon.vision.ImagePredictor) as our task of interest as follows:

```{.python .input}
import autogluon.core as ag
from autogluon.vision import ImagePredictor
```

## Create AutoGluon Dataset

For demonstration purposes, we use a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle.
Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

We can load a dataset by downloading a url data automatically:

```{.python .input}
train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
print(train_dataset)
```

## Use AutoGluon to Fit Models

Now, we fit a classifier using AutoGluon as follows:

```{.python .input}
predictor = ImagePredictor()
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train_dataset, hyperparameters={'epochs': 2})  # you can trust the default config, we reduce the # epoch to save some build time
```

Within `fit`, the dataset is automatically split into training and validation sets.
The model with the best hyperparameter configuration is selected based on its performance on the validation set.
The best model is finally retrained on our entire dataset (i.e., merging training+validation) using the best configuration.

The best Top-1 accuracy achieved on the validation set is as follows:

```{.python .input}
fit_result = predictor.fit_summary()
print('Top-1 train acc: %.3f, val acc: %.3f' %(fit_result['train_acc'], fit_result['valid_acc']))
```

## Predict on a New Image

Given an example image, we can easily use the final model to `predict` the label (and the conditional class-probability denoted as `score`):

```{.python .input}
image_path = test_dataset.iloc[0]['image']
result = predictor.predict(image_path)
print(result)
```

If probabilities of all categories are needed, you can call `predict_proba`:

```{.python .input}
proba = predictor.predict_proba(image_path)
print(proba)
```

You can also feed in multiple images all together, let's use images in test dataset as an example:
```{.python .input}
bulk_result = predictor.predict(test_dataset)
print(bulk_result)
```

An extra column will be included in bulk prediction, indicate the corresponding image for the row. There will be (# image) rows in the result, each row includes `class`, `score`, `id` and `image` for prediction class, prediction confidence, class id, and image path respectively.


## Generate image features with a classifier

Extracting representation from the whole image learned by a model is also very useful. We provide `predict_feature` function to allow predictor to return the N-dimensional image feature where `N` depends on the model(usually a 512 to 2048 length vector)

```{.python .input}
image_path = test_dataset.iloc[0]['image']
feature = predictor.predict_feature(image_path)
print(feature)
```



## Evaluate on Test Dataset

You can evaluate the classifier on a test dataset rather than retrieving the predictions.

The validation and test top-1 accuracy are:

```{.python .input}
test_acc, _ = predictor.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Save and load classifiers

You can directly save the instances of classifiers:

```{.python .input}
filename = 'predictor.ag'
predictor.save(filename)
predictor_loaded = ImagePredictor.load(filename)
# use predictor_loaded as usual
result = predictor_loaded.predict(image_path)
print(result)
```

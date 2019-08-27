# Image Classification - How to use your own datasets

This tutorial demonstrates how to use AutoGluon with your own custom datasets.
As an example, we use a dataset from Kaggle to show what steps are needed to format image data properly for AutoGluon.

## Step 1: Organizing the dataset into proper directories

After completing this step, you will have the following directory structure on your machine:

```
    data/
    ├── class1/
    ├── class2/
    ├── class3/
    ├── ...
```

Here `data` is a folder containing the raw images categorized into classes (for example, subfolder `class1` contains all images that belong to the first class, `class2` contains all images belonging to the second class, etc). 
We generally recommend at least 100 training images per class for reasonable classification performance, but this will depend on the type of images in your specific use-case.

Under each class, the following image formats are supported when training your model:

- JPG
- JPEG
- PNG

TODO: must all images be of the same format, or can some be JPG and some PNG within the same dataset?

You will need to organize your dataset into the above directory structure before using AutoGluon.
Below, we demonstrate how to do this organization for a Kaggle dataset.

### Example: Kaggle dataset

Kaggle is a popular machine learning competition platform and contains lots of
datasets for different machine learning tasks including image classification.
If you don't have Kaggle account, please register one at [Kaggle](https://www.kaggle.com/). 
Then, please follow the [Kaggle installation](https://github.com/Kaggle/kaggle-api/) to obtain access to Kaggle's data downloading API.

To find image classification datasets in Kaggle, let's go to [Kaggle](https://www.kaggle.com/) 
and search using keyword `image classification` either under `Datasets` or `Competitions`.

For example, we find the `Shopee-IET Machine Learning Competition` under the `InClass` tab in `Competitions`.

We then navigate to [Data](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) to download the dataset using the Kaggle API.

An example shell script to download the dataset to `~/data/shopeeiet/` can be found here: [download_shopeeiet.sh](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/_static/script/download_shopeeiet.sh).

After downloading this script to your machine, run it with:

```sh
sh download_shopeeiet.sh
```

Now we have the desired directory structure under `~/data/shopeeiet/`, which in this case looks as follows:

```
    shopeeiet
    ├── BabyBibs
    ├── BabyHat
    ├── BabyPants
    ├── ...
```

Here is an example image from this data:

![](../img/shopeeiet_example.png)



## Step 2: Split data into training/validation/test sets

A fundamental step in machine learning is to split the data into disjoint sets used for different purposes.

Training Set: The majority of your data should be in the training set.
This is the data your model sees during training:
it is used to learn the parameters of the model,
namely the weights of the neural network classifier.

Validation Set: A separate validation set (sometimes also called the dev set)
is also used during AutoGluon's training process. While neural network weights are updated based on the training data, 
each neural network requires the user to specify many hyperparameters (eg. learning rates, regularization strength, etc.) that will greatly affect the training process.  AutoGluon automatically tries many different values of these hyperparameters and evaluates each hyperparameter setting by measuring the performance of the resulting network on the validation set.

Test Set: A separate set of images, possibly without available labels. These data are never used during any part of the model-construction process. If unlabeled, these may correspond to images whose labels we would like to predict. If labeled, these images may correspond to images we reserve for estimating the performance of our final model.


### Dataset format after splitting

The following directory format is used by AutoGluon's `image_classification` task:

```
    data/
    ├── train/
        ├── class1/
        ├── class2/
        ├── class3/
        ├── ...
    └── test/
```

Here, the `train` directory has the same format as the `data` directory described in Step 1.

When there are no labels available for test images, the `test` directory simply contains a collection of image files.
Otherwise, `test` should have the same format as the `data` directory described in Step 1 (ie. same format as `train`) if we wish to evaluate the accuracy of our model on the test data using AutoGluon.

We show an example below on how to convert data source obtained in Step 1
to Training/Test split with the required format.  
In this example, we provide a script to split the Kaggle data into the required format;
please click the download link of [prepare_shopeeiet.py](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/_static/script/prepare_shopeeiet.py).

After running the command:

```sh
python prepare_shopeeiet.py --data ~/data/shopeeiet/
```

the resulting data should be converted into the following directory structure:

```
    shopeeiet
    ├── train
        ├── BabyBibs
        ├── BabyHat
        ├── BabyPants
        ├── ...
    └── test
```

Now you have a dataset ready used in AutoGluon.

### Create Dataset object in AutoGluon

To tell AutoGluon where the training data is located, use:  

```python
from autogluon import image_classification as task
dataset = task.Dataset(name='shopeeiet', train_path='data/shopeeiet/train')
```

AutoGluon will automatically infer how many classes there are based on the directory structure. 
By default, AutoGluon automatically constructs the training/validation set split:

- Training Set: 90% of images.
- Validation Set: 10% of images.

where the images that fall into the validation set are randomly chosen from the training data.

TODO: is the random train/val split process stratified based on the class, or just purely random?


### Manually-specified training/validation split


Instead, you can also split your labeled data manually into training and validation sets.
Manually splitting your data is a good choice when you want to exercise more control over the process
or if there are specific images that you're sure you want included in a certain part of your model training lifecycle.
For example, if images are collected over time in your application, you may wish to hold-out the most recent images as the validation data to more faithfully reflect the fact that your classifier's performance during deployment will be solely determined based on future data.

TODO: Missing example of how to manually specify the validation dataset and what directory structure is needed in this case. E.g. add example of the how the directory structure should look and code snippet with: task.Dataset(train_path='data/shopeeiet/train', val_path='data/shopeeiet/val')


## Step 3: Use AutoGluon fit() to generate a classification model (Optional)

Now that we have a `Dataset` object, we can use AutoGluon's default configuration to train an image classification model.
All you have to do is simply call the ``fit`` function. 

Due to the large size of the Kaggle dataset and time constraints of this demo,
we don't recommend directly running `fit` here since it will take a while to execute.

On your own, please feel free to try running the following commands with small time limits (just uncomment the code):

```python
# time_limits = 10 * 60 # 10mins
# results = task.fit(dataset, time_limits=time_limits)
```

The top-1 accuracy of the best model on the validation set is:

```python
# print('Top-1 acc: %.3f' % results.metric)
```

###  Using AutoGluon to generate predictions on test images 

We can ask our trained model to generate predictions on the provided test images.
We first load the test data as a `Dataset` object and then call [predict](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.task.image_classification.ImageClassification.predict):

```python
# test_dataset = task.Dataset(test_path='data/shopeeiet/test')
# inds, probs = task.predict(test_dataset)
```

`inds` above contains the indices of the predicted class for each test image, while `probs` contains the confidence in these predictions.


Here are the results of AutoGluon's default `fit` and `predict` under different `time_limits` when executed on a p3.16xlarge EC2 instance:

- TODO: The validation top-1 accuracy within 5h is 0.xxx (on test data, ranks ??th place in Kaggle competition)
- TODO: The validation top-1 accuracy within 24h is 0.yyy (on test data, ranks ??th place in Kaggle competition)
- TODO: The validation top-1 accuracy within 72h is 0.zzz (on test data, ranks ??th place in Kaggle competition)



## Step 4: Submit test predictions to Kaggle (Optional)

If you wish to upload the model's predictions to Kaggle, here is how to convert them into a format suitable for entry into the Kaggle competition:

```python
# utils.generate_csv(inds, 'data/shopeeiet/submission.csv')
```

will produce a submission file located at: `data/shopeeiet/submission.csv`.

To see an example submission, check out the file `sample submission.csv` at this link: [Data](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data).

To make your own submission, click [Submission](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/submit)
and then follow the steps in the submission page (upload submission file, describe the submission,
and click the `Make Submission` button). Let's see how your model fares in this competition!


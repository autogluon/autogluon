# Image Classification - How to Use Your Own Datasets

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

Here `data` is a folder containing the raw images categorized into classes. For example, subfolder `class1` contains all images that belong to the first class, `class2` contains all images belonging to the second class, etc. 
We generally recommend at least 100 training images per class for reasonable classification performance, but this might depend on the type of images in your specific use-case.

Under each class, the following image formats are supported when training your model:

- JPG
- JPEG
- PNG

In the same dataset, all the images should be in the same format.

You will need to organize your dataset into the above directory structure before using AutoGluon.
Below, we demonstrate how to construct this organization for a Kaggle dataset.

### Example: Kaggle dataset

Kaggle is a popular machine learning competition platform and contains lots of
datasets for different machine learning tasks including image classification.
If you don't have Kaggle account, please register one at [Kaggle](https://www.kaggle.com/). 
Then, please follow the [Kaggle installation](https://github.com/Kaggle/kaggle-api/) to obtain access to Kaggle's data downloading API.

To find image classification datasets in Kaggle, let's go to [Kaggle](https://www.kaggle.com/) 
and search using keyword `image classification` either under `Datasets` or `Competitions`.

For example, we find the `Shopee-IET Machine Learning Competition` under the `InClass` tab in `Competitions`.

We then navigate to [Data](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) to download the dataset using the Kaggle API.

An example shell script to download the dataset to `~/data/shopeeiet/` can be found here: [download_shopeeiet.sh](../static/download_shopeeiet.sh).

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

Here are some example images from this data:

![](../img/shopeeiet_example.png)



## Step 2: Split data into training/validation/test sets

A fundamental step in machine learning is to split the data into disjoint sets used for different purposes.

Training Set: The majority of your data should be in the training set.
This is the data your model sees during training:
it is used to learn the parameters of the model,
namely the weights of the neural network classifier.

Validation Set: A separate validation set (sometimes also called the dev set)
is also used during AutoGluon's training process. While neural network weights are updated based on the training data, 
each neural network requires the user to specify many hyperparameters (eg. learning rates, etc.) that will greatly affect the training process.  AutoGluon automatically tries many different values of these hyperparameters and evaluates each hyperparameter setting by measuring the performance of the resulting network on the validation set.

Test Set: A separate set of images, possibly without available labels. These data are never used during any part of the model construction or learning process. If unlabeled, these may correspond to images whose labels we would like to predict. If labeled, these images may correspond to images we reserve for estimating the performance of our final model.


### Dataset format after splitting

The following directory format is used by AutoGluon's `image_classification` task:

```
    data/
    ├── train/
        ├── class1/
        ├── class2/
        ├── class3/
        ├── ...
    ├── val(optional)/
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
to Training/Validation/Test split with the required format.  
In this example, we provide a script to split the Kaggle data into the required format;
please click the download link of [prepare_shopeeiet.py](../static/prepare_shopeeiet.py).

### Automatic training/validation split

Since AutoGluon provides the automatic Training/Validation split, we can skip the Validation split by running the command:

```sh
python prepare_shopeeiet.py --data ~/data/shopeeiet/ --split 0
```

where `--split 0` would skip the validation split, therefore all the data in `data` directory would be used as `train` data, later on the AutoGluon `Dataset` would automatically split into Training (90% of the data) and Validation (10% of the data).

The resulting data should be converted into the following directory structure:

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

To tell AutoGluon where the training data is located, which means let AutoGluon conduct the Training/Validation split, use:  

```{.python .input}
# from autogluon import image_classification as task
# dataset = task.Dataset(name='shopeeiet', train_path='~/data/shopeeiet/train')
```

AutoGluon will automatically infer how many classes there are based on the directory structure. 
By default, AutoGluon automatically constructs the training/validation set split:

- Training Set: 90% of images.
- Validation Set: 10% of images.

where the images that fall into the validation set are randomly chosen from the training data based on the class.

### Manually-specified training/validation split

Instead, you can also split your labeled data manually into training and validation sets.
Manually splitting your data is a good choice when you want to exercise more control over the process
or if there are specific images that you're sure you want included in a certain part of your model training lifecycle.

If we want to manually specify the Training/Validation split, we could construct by running the command:

```sh
# python prepare_shopeeiet.py --data ~/data/shopeeiet/ --split 1
```

where `--split 1` would sample 10% data from the `data` directory as Validation set, and the rest 90% data would be Training set.

The resulting data should be looking as the following structure:

```
    shopeeiet
    ├── train
        ├── BabyBibs
        ├── BabyHat
        ├── BabyPants
        ├── ...
    ├── val
        ├── BabyBibs
        ├── BabyHat
        ├── BabyPants
        ├── ...
    └── test
```

Then tell AutoGluon where the training and validation data is, which means we disable AutoGluon's automatic Training/Validation split functionality, instead, we manually provide the Training/Validation split via:

```{.python .input}
# from autogluon import image_classification as task
# dataset = task.Dataset(name='shopeeiet', train_path='data/shopeeiet/train', val_path='data/shopeeiet/val')
```

## Step 3: Use AutoGluon fit to generate a classification model (Optional)

Now that we have a `Dataset` object, we can use AutoGluon's default configuration to obtain an image classification model.
All you have to do is simply call the `fit` function. 

Due to the large size of the Kaggle dataset and time constraints of this demo,
we don't recommend directly running `fit` here since it will take a while to execute.

On your own, please feel free to try running the following commands with small time limits (just uncomment the code):

```{.python .input}
# time_limits = 10 * 60 # 10mins
# results = task.fit(dataset, time_limits=time_limits)
```

The top-1 accuracy of the best model on the validation set is:

```{.python .input}
# print('Top-1 acc: %.3f' % results.metric)
```

###  Using AutoGluon to generate predictions on test images 

We can ask our final model to generate predictions on the provided test images.
We first load the test data as a `Dataset` object and then call [predict](../api/autogluon.task.base.html#autogluon.task.base.BaseTask.predict):

```{.python .input}
# test_dataset = task.Dataset(test_path='data/shopeeiet/test')
# inds, probs = task.predict(test_dataset)
```

`inds` above contains the indices of the predicted class for each test image, while `probs` contains the confidence in these predictions.


Here are the results of AutoGluon's default `fit` and `predict` under different `time_limits` when executed on a p3.16xlarge EC2 instance:

- The validation top-1 accuracy within 5h is 0.842, and ranks 14th place in Kaggle competition.
- The validation top-1 accuracy within 24h is 0.846, and ranks 12th place in Kaggle competition.
- The validation top-1 accuracy within 72h is 0.852, and ranks 9th place in Kaggle competition.


## Step 4: Submit test predictions to Kaggle (Optional)

If you wish to upload the model's predictions to Kaggle, here is how to convert them into a format suitable for a submission into the Kaggle competition:

```{.python .input}
# utils.generate_csv(inds, 'data/shopeeiet/submission.csv')
```

will produce a submission file located at: `data/shopeeiet/submission.csv`.

To see an example submission, check out the file `sample submission.csv` at this link: [Data](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data).

To make your own submission, click [Submission](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/submit)
and then follow the steps in the submission page (upload submission file, describe the submission,
and click the `Make Submission` button). Let's see how your model fares in this competition!
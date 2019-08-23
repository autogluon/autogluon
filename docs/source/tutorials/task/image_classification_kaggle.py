#TODO: change it to an end to end example including
# 1, download kaggle dataset (remove two kaggle images)
# 2, split the data into training, valididation, test, using validation to select the model, and report the accuracy on test data
# 3, using default fit function to generate the results
# 4, submit model to the kaggle and get the results
# make sure the links are working (download link)

"""2. Image Classification - A Kaggle Dataset Example
=======================================================

We show how to use custom dataset in AutoGluon by using dataset from Kaggle as an example.

Step 1: Prepare Kaggle Dataset
----------------------------------------
The output of this step would be

::

    data
    ├── images
        ├── class 1
        ├── class 2
        ├── class 3
        ├── ...

where `data` is the data folder, `images` is the subfolder of `data` contains the raw images
categorized into classes.

We recommend about 1000 training images per class.
The minimum per class is 10, or 50 for advanced models.

Under each class, the following image formats are supported when training your model:

- JPG
- JPEG
- PNG


We show how to prepare the Kaggle dataset as below.

Kaggle is a popular machine learning competition platform and contains lots of
datasets for different machine learning tasks including image classification.

Before downloading dataset, if you don't have Kaggle account,
please direct to `Kaggle <https://www.kaggle.com/>`__
to register one. Then please follow the Kaggle
`installation <https://github.com/Kaggle/kaggle-api/>`__ to install Kaggle API
for downloading the data.

To download an image classfication dataset from Kaggle,
We first go to `Kaggle <https://www.kaggle.com/>`__,
second, let's search using keyword 'image classification' either under `Datasets` or `Competitions`.

For example, we find `Shopee-IET Machine Learning Competition` under `Competitions` `InClass`.

We then go to `Data <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data>`__ to download the dataset using the Kaggle API.

An example shell script to download the dataset to `~/data/shopeeiet/images` can be found:

:download:`Download download_shopeeiet.sh <../../_static/script/download_shopeeiet.sh>`

Run it with

::

    sh download_shopeeiet.sh

Now we have the following structure under `~/data/shopeeiet/images`:

::

    shopeeiet
    ├── images
        ├── BabyBibs
        ├── BabyHat
        ├── BabyPants
        ├── ...

Example images of some classes are:

.. image:: ../../../_static/img/shopeeiet_example.png


Step 2: Split Dataset to Training/Validation
---------------------------------------------

Training Set: The vast majority of your data should be in the training set.
This is the data your model "sees" during training:
it's used to learn the parameters of the model,
namely the weights of the connections between nodes of the neural network.

Validation Set: The validation set, sometimes also called the "dev" set,
is also used during the training process.
After the model learning framework incorporates training data
during each iteration of the training process,
it uses the model's performance on the validation set to tune the model's hyperparameters,
which are variables that specify the model's structure.

Manual Splitting: You can also split your dataset yourself.
Manually splitting your data is a good choice
when you want to exercise more control over the process
or if there are specific examples that you're sure you want
included in a certain part of your model training lifecycle.

AutoGluon would automatically conduct the training/validation set split:
- Training Set: 90% of images.
- Validation Set: 10% of images.

The Required Format
~~~~~~~~~~~~~~~~~~~~

The following format is used in AutoGluon image classification task.

::

    data
    ├── images
    ├── train
        ├── class 1
        ├── class 2
        ├── class 3
        ├── ...
    └── test

We show an example below on how to convert data source obtained in Step 1
to Training/Validation split with the required format.

We provide a script to convert Shopee-IET data to the required format:

:download:`Download prepare_shopeeiet.py <../../_static/script/prepare_shopeeiet.py>`

After running with:

::

    python prepare_shopeeiet.py --data ~/data/shopeeiet/

Thus, the resulting data is in the following format:

::

    shopeeiet
    ├── images
    ├── train
        ├── BabyBibs
        ├── BabyHat
        ├── BabyPants
        ├── ...
    └── test

Now you have a dataset ready used in AutoGluon.

Create Train/Validation Split using AutoGluon Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from autogluon import image_classification as task
    dataset = task.Dataset(name='shopeeiet', train_path='data/shopeeiet/train')

Step 3: Use AutoGluon Fit to Generate Model
--------------------------------------------
Then we will use the default configuration of the image classification
to generate: \* Best result of the search in terms of accuracy \*
According best configuration regarding to the best result

To acheive this, we are using ``fit`` function to generate the above
results based on the datasets.

.. code-block:: python

    results = task.fit(dataset)

The best top-1 accuracy on the validation set is:

.. code-block:: python

    print('Top-1 acc: %.3f' % results.metric)

Step 4: Submit Test Results to Kaggle
---------------------------------------

Generate Test Result using AutoGluon Predict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    file = task.predict(test_path='data/shopeeiet/test')

A submission example named `sample submission.csv` could be found at `Data <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data>`__

Finally it is time to make your submission.
You would click `Submission <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/submit>`__
and then follow the two steps in the submission page by uploading your submission file and describe the submission,
and click the `Make Submission` button.
"""

"""1. Image Classification - Data Preparation
==============================================

We show how to get custom data and prepare it in the format required by AutoGluon.

Step 1: Prepare Your Images (Optional)
----------------------------------------
If you already have a dataset for image classification, please skip this step.

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

- JPEG
- PNG


We show data source examples.

Example 1: Kaggle
~~~~~~~~~~~~~~~~~~~

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

.. image:: ../../../_static/img/image_classification_data_search.png
.. image:: ../../../_static/img/shopeeiet_search.png

We then go to `Data` to download the dataset using the Kaggle API.

.. image:: ../../../_static/img/shopeeiet_data.png

An example shell script to download the dataset to `~/data/shopeeiet/images` can be found:

:download:`Download download_shopeeiet.sh<../../_static/script/download_shopeeiet.sh>`

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

Example image of some classes are:

.. image:: ../../../_static/img/shopeeiet_example.png


Example 2: Amazon Internal Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also access `Amazon Image assist <https://image-assist.amazon.com>`__
and search the images using ASINs.



Step 2: Convert Data Source into Required Format
--------------------------------------------------

Training/Validation Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

AutoGluon recommends:

- Training Set: 90% of images .
- Validation Set: 10% of images.

The Required Format
~~~~~~~~~~~~~~~~~~~~

The following format is used in AutoGluon image classification task.

::

    data
    ├── images
    ├── class 1
        ├── class 2
        ├── class 3
        ├── ...
    ├── train
    └── val
    └── test

We show an example below on how to convert data source obtained in Step 1
to Training/Validation split with the required format.

Example: Convert Kaggle Data (Shopee-IET) to Required Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a script to convert Shopee-IET data to the required format:

:download:`Download prepare_shopeeiet.py<../../_static/script/prepare_shopeeiet.py>`

After running with:

::

    python prepare_shopeeiet.py --data ~/data/shopeeiet/ --split 9

We can then generate the training/val split where 90% of the images go to training set and
10% of the images are in validation set
(`split` in the above command means # of training samples/# of validation samples).

Thus, the resulting data is in the following format:

::

    shopeeiet
    ├── images
        ├── BabyBibs
        ├── BabyHat
        ├── BabyPants
        ├── ...
    ├── train
    └── val
    └── test

Now you have a dataset ready used in AutoGluon.
"""

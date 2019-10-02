"""2. Image Classification - A Kaggle Dataset Example
=======================================================

We show how to use custom dataset in AutoGluon by using dataset from Kaggle as an example.

Step 1: Prepare Kaggle Dataset
----------------------------------------
The output of this step would be

::

    data
    ├── class 1
    ├── class 2
    ├── class 3
    ├── ...

where `data` is the data folder containing the raw images categorized into classes.

We recommend about 100 training images per class.

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

To download an image classification dataset from Kaggle,
we first go to `Kaggle <https://www.kaggle.com/>`__,
second, let's search using keyword `image classification` either under `Datasets` or `Competitions`.

For example, we find `Shopee-IET Machine Learning Competition` under `Competitions` `InClass`.

We then go to `Data <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data>`__ to download the dataset using the Kaggle API.

An example shell script to download the dataset to `~/data/shopeeiet/` can be found via
the download link of `download_shopeeiet.sh <http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/_static/script/download_shopeeiet.sh>`__ .

Run it with

::

    sh download_shopeeiet.sh

Now we have the following structure under `~/data/shopeeiet/`:

::

    shopeeiet
    ├── BabyBibs
    ├── BabyHat
    ├── BabyPants
    ├── ...

Example images of some classes are:

.. image:: ../../../_static/img/shopeeiet_example.png


Step 2: Split Dataset to Training/Validation
---------------------------------------------

Training Set: The vast majority of your data should be in the training set.
This is the data your model sees during training:
it's used to learn the parameters of the model,
namely the weights of the connections between nodes of the neural network.

Validation Set: The validation set, sometimes also called the dev set,
is also used during the training process.
After the model learning framework incorporates training data
during each iteration of the training process,
it uses the model's performance on the validation set to tune the model's hyper-parameters,
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
    ├── train
        ├── class 1
        ├── class 2
        ├── class 3
        ├── ...
    └── test

We show an example below on how to convert data source obtained in Step 1
to Training/Validation split with the required format.

We provide a script to convert Shopee-IET data to the required format,
please click the download link of `prepare_shopeeiet.py <http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/_static/script/prepare_shopeeiet.py>`__ .

After running with:

::

    python prepare_shopeeiet.py --data ~/data/shopeeiet/

Thus, the resulting data is in the following format:

::

    shopeeiet
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

Step 3: Use AutoGluon Fit to Generate Model (Optional)
-------------------------------------------------------
Then we will use the default configuration of the image classification
to generate best result of the search in terms of accuracy
according to the best configuration.

To achieve this, we are using ``fit`` function to generate the above
results based on the datasets.
Due to the large scale of the Kaggle dataset and time limitation of the demo,
we don't recommend to directly run the following command.
We provide the results regarding to different time_limits on p3.16xlarge EC2 as below.
The validation top-1 accuracy within 5h is 0.xxx.
The validation top-1 accuracy within 24h is 0.yyy.
The validation top-1 accuracy within 72h is 0.zzz.

However, please also feel free to try to run the following commands with small time limits to run
it through for the Kaggle completition dataset. For example,

.. code-block:: python

    time_limits = 10 * 60 # 10mins
    results = task.fit(dataset, time_limits=time_limits)

The best top-1 accuracy on the validation set is:

.. code-block:: python

    print('Top-1 acc: %.3f' % results.metric)

Step 4: Submit Test Results to Kaggle
---------------------------------------

Generate Test Result using AutoGluon Predict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can then generate the test results based on provided test data.
Let's first construct the test dataset and then use [predict](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.task.image_classification.ImageClassification.predict)
to generate the results.

.. code-block:: python

    test_dataset = task.Dataset(test_path='data/shopeeiet/test')
    inds, probs = task.predict(test_dataset)

Produce the Required Format for Submission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    utils.generate_csv(inds, filepath)

The submission file would be `filepath` (default 'data/shopeeiet/submission.csv).

A submission example named `sample submission.csv` could be found at `Data <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data>`__.

Finally it is time to make your submission.
You would click `Submission <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/submit>`__
and then follow the two steps in the submission page by uploading your submission file and describing the submission,
and click the `Make Submission` button.
"""

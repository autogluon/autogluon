AutoGluon: AutoML for Text, Image, and Tabular Data
====================================================

.. |ReleaseVersion| image:: https://img.shields.io/badge/doc%20release-v0.6.0-blue
    :target: https://auto.gluon.ai/dev/versions.html
    :scale: 100%
.. |StableVersion| image:: https://img.shields.io/github/v/release/awslabs/autogluon?color=blue&label=stable%20release&sort=semver
    :target: https://auto.gluon.ai/stable/index.html
    :scale: 100%
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue
    :target: https://pypi.org/project/autogluon/
.. |GitHub| image:: https://img.shields.io/github/stars/awslabs/autogluon?style=social
    :target: https://github.com/awslabs/autogluon/stargazers
.. |Twitter| image:: https://img.shields.io/twitter/follow/autogluon?style=social
    :target: https://twitter.com/autogluon
.. |Downloads| image:: https://pepy.tech/badge/autogluon/month
    :target: https://pepy.tech/project/autogluon
.. |License| image:: https://img.shields.io/github/license/awslabs/autogluon?color=blue
    :target: https://github.com/awslabs/autogluon/blob/master/LICENSE

|ReleaseVersion| |StableVersion| |PythonVersion| |License| |Downloads| |GitHub| |Twitter| 

`AutoGluon` enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning image, text, and tabular data. Intended for both ML beginners and experts, `AutoGluon` enables you to:

- Quickly prototype deep learning and classical ML solutions for your raw data with a few lines of code.
- Automatically utilize state-of-the-art techniques (where appropriate) without expert knowledge.
- Leverage automatic hyperparameter tuning, model selection/ensembling, architecture search, and data processing.
- Easily improve/tune your bespoke models and data pipelines, or customize `AutoGluon` for your use-case.

.. note::

   Example using AutoGluon to train and deploy a high-performance model on a tabular dataset:

   >>> from autogluon.tabular import TabularDataset, TabularPredictor
   >>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
   >>> test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
   >>> predictor = TabularPredictor(label='class').fit(train_data=train_data)
   >>> predictions = predictor.predict(test_data)

   AutoGluon can be applied for prediction tasks that involve image and text data. For adopting state-of-the-art deep learning models for multimodal prediction problems, you may try `autogluon.multimodal`:

   >>> from autogluon.multimodal import MultiModalPredictor
   >>> from datasets import load_dataset
   >>> train_data = load_dataset("glue", 'mrpc')['train'].to_pandas().drop('idx', axis=1)
   >>> test_data = load_dataset("glue", 'mrpc')['validation'].to_pandas().drop('idx', axis=1)
   >>> predictor = MultiModalPredictor(label='label').fit(train_data)
   >>> predictions = predictor.predict(test_data)
   >>> score = predictor.evaluate(test_data)


Installation
------------

.. include:: install-include.rst


Quick Start
-----------

.. raw:: html
   :file: static/application.html


Tutorials
---------

.. container:: cards

   .. card::
      :title: Tabular Prediction
      :link: tutorials/tabular_prediction/index.html

      How to predict variables in tabular datasets.

   .. card::
      :title: Multimodal Prediction
      :link: tutorials/multimodal/index.html

      How to solve problems that contain Image, Text, and Tabular features at the same time.

   .. card::
      :title: Time Series Forecasting
      :link: tutorials/timeseries/index.html

      How to train time series models for forecasting.

   .. card::
      :title: Image Prediction (Legacy Version)
      :link: tutorials/image_prediction/index.html

      How to classify images into various categories.

   .. card::
      :title: Object Detection (Legacy Version)
      :link: tutorials/object_detection/index.html

      How to detect objects and their location in images.

   .. card::
      :title: Text Prediction (Legacy Version)
      :link: tutorials/text_prediction/index.html

      How to solve NLP problems via supervised learning from raw text.

Launch Tutorial Notebooks
-------------------------

.. raw:: html
   :file: static/platforms.html

.. toctree::
   :maxdepth: 3
   :hidden:

   tutorials/tabular_prediction/index
   tutorials/multimodal/index
   tutorials/timeseries/index
   tutorials/cloud_fit_deploy/index
   cheatsheet.rst
   tutorials/image_prediction/index
   tutorials/object_detection/index
   tutorials/text_prediction/index
   api/autogluon.predictor
   api/autogluon.features
   api/autogluon.tabular.models

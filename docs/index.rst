AutoGluon: AutoML for Text, Image, and Tabular Data
====================================================

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

   AutoGluon can be applied just as easily for prediction tasks with image or text data.


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
      :title: Image Prediction
      :link: tutorials/image_prediction/index.html

      How to classify images into various categories.

   .. card::
      :title: Object Detection
      :link: tutorials/object_detection/index.html

      How to detect objects and their location in images.

   .. card::
      :title: Text Prediction
      :link: tutorials/text_prediction/index.html

      How to solve NLP problems via supervised learning from raw text.

   .. card::
      :title: Multimodal Prediction
      :link: tutorials/tabular_prediction/tabular-multimodal.html

      How to solve problems that contain Image, Text, and Tabular features at the same time.


.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/tabular_prediction/index
   tutorials/image_prediction/index
   tutorials/object_detection/index
   tutorials/text_prediction/index
   tutorials/tabular_prediction/tabular-multimodal
   tutorials/cloud_fit_deploy/index
   api/autogluon.predictor
   api/autogluon.features
   api/autogluon.tabular.models

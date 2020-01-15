AutoGluon: AutoML Toolkit for Deep Learning
===========================================

`AutoGluon` enables easy-to-use and easy-to-extend AutoML with a focus on deep learning and real-world applications spanning image, text, or tabular data. Intended for both ML beginners and experts, `AutoGluon` enables you to:

- Quickly prototype deep learning solutions for your data with few lines of code.
- Leverage automatic hyperparameter tuning, model selection / architecture search, and data processing.
- Automatically utilize state-of-the-art deep learning techniques without expert knowledge.
- Easily improve existing bespoke models and data pipelines, or customize `AutoGluon` for your use-case.

.. note::

   Example using AutoGluon to train and deploy high-performance model on a tabular dataset:
   
   >>> from autogluon import TabularPrediction as task
   >>> predictor = task.fit(train_data=task.Dataset(file_path=TRAIN_DATA.csv), label=COLUMN_NAME)
   >>> predictions = predictor.predict(task.Dataset(file_path=TEST_DATA.csv))
   
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
      :title: Image Classification
      :link: tutorials/image_classification/index.html

      How to classify images into various categories.
    
   .. card::
      :title: Object Detection
      :link: tutorials/object_detection/index.html

      How to detect objects and their location in images.

   .. card::
      :title: Text Classification
      :link: tutorials/text_classification/index.html

      How to make predictions based on text data.


Advanced Topics
~~~~~~~~~~~~~~~

.. container:: cards

   .. card::
      :title: Customize AutoGluon
      :link: tutorials/course/index.html

      Advanced usage of AutoGluon APIs for customized applications.

   .. card::
      :title: Neural Architecture Search
      :link: tutorials/nas/index.html

      How to perform neural architecture search.

   .. card::
      :title: For PyTorch Users
      :link: tutorials/torch/index.html

      How to do hyperparameter tuning or architecture search for any PyTorch model.

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/tabular_prediction/index
   tutorials/image_classification/index
   tutorials/object_detection/index
   tutorials/text_classification/index
   tutorials/course/index
   tutorials/nas/index
   tutorials/torch/index
   api/autogluon.space
   api/autogluon.core
   api/autogluon.task
   api/autogluon.scheduler
   api/autogluon.searcher
   api/autogluon.utils
   model_zoo/index

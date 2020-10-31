AutoGluon: AutoML for Text, Image, and Tabular Data
====================================================

`AutoGluon` enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning tabular, text, and image data. Intended for both ML beginners and experts, `AutoGluon` enables you to:

- Quickly prototype deep learning and classical ML solutions for your raw data with a few lines of code.
- Automatically utilize state-of-the-art techniques (where appropriate) without expert knowledge.
- Leverage automatic hyperparameter tuning, model selection/ensembling, architecture search, and data processing.
- Easily improve/tune your bespoke models and data pipelines, or customize `AutoGluon` for your use-case.

.. note::

   Example using AutoGluon to train and deploy high-performance model on a tabular dataset:
   
   >>> from autogluon.tabular import TabularPrediction as task
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
      :title: Text Prediction
      :link: tutorials/text_prediction/index.html

      How to solve NLP problems via supervised learning from raw text.


Advanced Topics
~~~~~~~~~~~~~~~

.. container:: cards

   .. card::
      :title: Tune Custom Models
      :link: tutorials/course/index.html

      How to hyperparameter-tune your own custom models or Python code.

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
   tutorials/text_prediction/index
   tutorials/course/index
   tutorials/nas/index
   tutorials/torch/index
   api/autogluon.core
   api/autogluon.task
   api/autogluon.core.space
   api/autogluon.core.scheduler
   api/autogluon.core.searcher
   api/autogluon.core.utils
   model_zoo/index

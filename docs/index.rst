AutoGluon: AutoML Toolkit with MXNet Gluon
==========================================

`AutoGluon` enables easy-to-use and easy-to-extend AutoML with a focus on deep learning and real-world applications spanning image, text, or tabular data. Intended for both ML beginners and experts, `AutoGluon` enables you to:

- Quickly prototype deep learning solutions for your data with few lines of code.
- Leverage automatic hyperparameter tuning, model selection / architecture search, and data processing.
- Automatically utilize state-of-the-art deep learning techniques without expert knowledge.
- Easily improve existing bespoke models and data pipelines, or customize `AutoGluon` for your use-case.

.. note::
Here's a basic example using AutoGluon to train and deploy a high-performance model on a tabular dataset:
```
from autogluon import TabularPrediction as task
predictor = task.fit(train_data=task.Dataset(file_path=<TRAINING_DATA_TABLE_CSV>), label_column=<VARIABLE_TO_PREDICT>)
predictions = predictor.predict(task.Dataset(file_path=<TEST_DATA_TABLE_CSV>))
```
AutoGluon can be applied just as easily for prediction tasks involving images or text data.


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

      Tutorials for predicting variables in tabular datasets.

   .. card::
      :title: Image Classification
      :link: tutorials/image_classification/index.html

      Tutorials for image classification tasks.
    
   .. card::
      :title: Object Detection
      :link: tutorials/object_detection/index.html

      Tutorials for object detection tasks.

   .. card::
      :title: Text Classification
      :link: tutorials/text_classification/index.html

      Tutorials for text classification tasks.


Advanced Topics
~~~~~~~~~~~~~~~

.. container:: cards

   .. card::
      :title: Customize AutoGluon
      :link: tutorials/course/index.html

      Advanced usage of AutoGluon APIs.

   .. card::
      :title: Neural Architecture Search
      :link: tutorials/nas/index.html

      Tutorials on neural architecture search.

   .. card::
      :title: For PyTorch Users
      :link: tutorials/torch/index.html

      Hyperparameter-tuning/architecture-search with PyTorch models.

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

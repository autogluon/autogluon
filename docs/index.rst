AutoGluon: AutoML for Tabular, Text, and Image Data
====================================================

AutoGluon automatically trains models on raw data with a few lines of code.
Here is an example to predict the ``class`` column on a tabular dataset. 

>>> from autogluon.tabular import TabularPredictor
>>> predictor = TabularPredictor(label='class').fit('train.csv')
>>> preds = predictor.predict('test.csv')

The ``fit`` function tries multiple state-of-the-art machine learning and deep learning models within a specified time limit. 
It tunes hyperparameters, selects, ensembles and stacks models for the best performance. 

AutoGluon provides multiple modules to handle a diverse set of data types. The following example uses the ``vision`` module 
to train an object classifier on images stored in the folder ``train``. 

>>> from autogluon.vision import ImagePredictor, ImageDataset
>>> predictor = ImagePredictor().fit(ImageDataset.from_folder('train'))

Before diving into AutoGluon's usages, let's 

Installation
------------

.. include:: install-include.rst



Applications
------------

AutoGluon supports the following applications. You can start with the 


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
   tutorials/image_prediction/index
   tutorials/object_detection/index
   tutorials/text_prediction/index
   tutorials/tabular_prediction/tabular-multimodal
   tutorials/cloud_fit_deploy/index
   tutorials/course/index
   tutorials/nas/index
   tutorials/torch/index
   api/autogluon.predictor
   api/autogluon.core
   api/autogluon.core.space
   api/autogluon.core.scheduler
   api/autogluon.core.searcher
   api/autogluon.core.utils
   api/autogluon.features
   api/autogluon.tabular.models
   model_zoo/index

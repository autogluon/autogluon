Tutorials
=========


Image Prediction
----------------
.. container:: cards

   .. card::
      :title: Dataset Preparation for Image Predictor
      :link: image_prediction/dataset.html

      Quick dataset preparation guide for image prediction.

   .. card::
      :title: Quick Start Using FIT
      :link: image_prediction/beginner.html

      5 min tutorial on classifying images.

   .. card::
      :title: Customized Search and HPO
      :link: image_prediction/hpo.html

      Hyperparameter tuning to improve image classification accuracy.

   .. card::
      :title: Use Your Own Dataset
      :link: image_prediction/kaggle.html

      Example use of AutoGluon for image classification competition on Kaggle.


Object Detection
----------------
.. container:: cards

   .. card::
      :title: Dataset Preparation for Object Detector
      :link: object_detection/dataset.html

      Quick dataset preparation guide for object detection.

   .. card::
      :title: Quick Start Using FIT
      :link: object_detection/beginner.html

      5 min tutorial on detecting objects in images.


Tabular Prediction
------------------
.. container:: cards

   .. card::
      :title: Quick Start Tutorial
      :link: tabular_prediction/tabular-quickstart.html

      5 min tutorial on fitting models with tabular datasets.

   .. card::
      :title: In-depth FIT Tutorial
      :link: tabular_prediction/tabular-indepth.html

      In-depth tutorial on controlling various aspects of model fitting.

   .. card::
      :title: Kaggle Tutorial
      :link: tabular_prediction/tabular-kaggle.html

      Using AutoGluon for Kaggle competitions with tabular data.

   .. card::
      :title: Data Tables Containing Text
      :link: tabular_prediction/tabular-multimodal-text-others.html

      Modeling data tables with text and numeric/categorical features.

   .. card::
      :title: Multi-Label Prediction
      :link: tabular_prediction/tabular-multilabel.html

      How to predict multiple columns in a data table.

   .. card::
      :title: Adding a Custom Model
      :link: tabular-custom-model.html

      How to add a custom model to AutoGluon.

   .. card::
      :title: FAQ
      :link: tabular_prediction/tabular-faq.html

      Frequently asked questions about AutoGluon-Tabular.

Text Prediction
---------------
.. container:: cards

   .. card::
      :title: Quick Start Tutorial
      :link: text_prediction/beginner.html

      5 min tutorial on fitting prediction models with text in the dataset.

   .. card::
      :title: Text Prediction for Multimodal Tables with Text
      :link: text_prediction/multimodal_text.html

      Learning how AutoGluon Text can be used to train model for multimodal data table with text.


   .. card::
      :title: Customize Search Space and HPO
      :link: text_prediction/customization.html

      In-depth tutorial on learning how to customize the search space and try different
      HPO algorithms.


Tune Custom Models
--------------------
.. container:: cards

   .. card::
      :title: Search Space and Decorator
      :link: course/core.html

      AutoGluon's Core API for search spaces and searchable
      objects used to tune any training function's argument-values.

   .. card::
      :title: Search Algorithms
      :link: course/algorithm.html

      Use AutoGluon's search algorithms to tune
      arbitrary models/training-scripts.

   .. card::
      :title: Distributed Search Tutorial
      :link: course/distributed.html

      Easily distribute AutoGluon tuning
      across multiple machines.

   .. card::
      :title: Tune a Multi-Layer Perceptron
      :link: course/mlp.html

      Complete example of how to use AutoGluon
      and its state-of-the-art schedulers and
      model-based searchers to tune a MLP.

   .. card::
      :title: Fair Bayesian Optimization
      :link: course/fairbo.html

      Example of how to use constrained 
      Bayesian Optimization in AutoGluon to tune 
      ML models under fairness constraints.


Neural Architecture Search
--------------------------
.. container:: cards

   .. card::
      :title: Reinforcement Learning
      :link: nas/rl_searcher.html

      Comparing search via Reinforcement Learning against Random search.

   .. card::
      :title: Efficient NAS on Target Hardware
      :link: nas/enas_proxylessnas.html

      Efficient Neural Architecture Search for low latency model
      on target hardware.


.. toctree::
   :maxdepth: 2
   :hidden:

   course/index
   image_prediction/index
   object_detection/index
   text_prediction/index
   tabular_prediction/index
   customize/index
   nas/index

Tutorials
=========


Image Prediction
----------------
.. container:: cards

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

      In-depth tutorial on boosting accuracy and controlling model fitting.

   .. card::
      :title: Kaggle Tutorial
      :link: tabular_prediction/tabular-kaggle.html

      Using AutoGluon for Kaggle competitions with tabular data.


Text Prediction
---------------
.. container:: cards

   .. card::
      :title: Quick Start Tutorial
      :link: text_prediction/beginner.html

      10 min tutorial on fitting prediction models with text in the dataset.

   .. card::
      :title: Customize Search Space and HPO
      :link: text_prediction/customization.html

      In-depth tutorial on learning how to customize the search space and try different
      HPO algorithms.

   .. card::
      :title: Dealing with Mixed Data Types
      :link: text_prediction/heterogeneous.html

      Learning how to use AutoGluon to handle datasets with mixed data types.


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
   text_classification/index
   text_prediction/index
   tabular_prediction/index
   customize/index
   nas/index

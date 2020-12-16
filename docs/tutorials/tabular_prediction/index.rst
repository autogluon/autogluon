Tabular Prediction
====================

For standard datasets that are represented as tables (stored as CSV file, parquet from database, etc.), AutoGluon can produce models to predict the values in one column based on the values in the other columns. With just a single call to `fit()`, you can achieve high accuracy in standard supervised learning tasks (both classification and regression), without dealing with cumbersome issues like data cleaning, feature engineering, hyperparameter optimization, model selection, etc.


.. container:: cards

   .. card::
      :title: Quick Start Using FIT
      :link: tabular-quickstart.html

      Quick tutorial on fitting models with tabular datasets.

   .. card::
      :title: In-depth FIT Tutorial
      :link: tabular-indepth.html

      In-depth tutorial on controlling various aspects of model fitting.

   .. card::
      :title: Kaggle Tutorial
      :link: tabular-kaggle.html

      How to use AutoGluon for Kaggle competitions.

   .. card::
      :title: Explore Models for Data Tables with Text and Categorical Features
      :link: tabular-multimodal-text-others.html

      Tutorial about how to use autogluon to solve tasks that contain both text and categorical features.

   .. card::
      :title: FAQ
      :link: tabular-faq.html

      Frequently Asked Questions

.. toctree::
   :maxdepth: 1
   :hidden:

   tabular-quickstart
   tabular-indepth
   tabular-kaggle
   tabular-multimodal-text-others
   tabular-faq

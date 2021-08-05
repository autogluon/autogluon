Tabular Prediction
====================

For standard datasets that are represented as tables (stored as CSV file, parquet from database, etc.), AutoGluon can produce models to predict the values in one column based on the values in the other columns. With just a single call to `fit()`, you can achieve high accuracy in standard supervised learning tasks (both classification and regression), without dealing with cumbersome issues like data cleaning, feature engineering, hyperparameter optimization, model selection, etc.


.. container:: cards

   .. card::
      :title: Quick Start Using FIT
      :link: tabular-quickstart.html

      5 min tutorial on fitting models with tabular datasets.

   .. card::
      :title: In-depth FIT Tutorial
      :link: tabular-indepth.html

      In-depth tutorial on controlling various aspects of model fitting.

   .. card::
      :title: Kaggle Tutorial
      :link: tabular-kaggle.html

      Using AutoGluon for Kaggle competitions with tabular data.

   .. card::
      :title: Data Tables Containing Text
      :link: tabular-multimodal-text-others.html

      Modeling data tables with text and numeric/categorical features.

   .. card::
      :title: Multi-Label Prediction
      :link: tabular-multilabel.html

      How to predict multiple columns in a data table.

   .. card::
      :title: Adding a Custom Model
      :link: tabular-custom-model.html

      How to add a custom model to AutoGluon.

   .. card::
      :title: FAQ
      :link: tabular-faq.html

      Frequently asked questions about AutoGluon-Tabular.

.. toctree::
   :maxdepth: 1
   :hidden:

   tabular-quickstart
   tabular-indepth
   tabular-kaggle
   tabular-multimodal-text-others
   tabular-multilabel
   tabular-custom-model
   tabular-faq

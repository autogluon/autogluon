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
      :title: Data Tables Containing Image, Text, and Tabular
      :link: tabular-multimodal.html

      Modeling data tables with image, text, numeric, and categorical features.

   .. card::
      :title: Data Tables Containing Text
      :link: tabular-multimodal-text-others.html

      Modeling data tables with text and numeric/categorical features.

   .. card::
      :title: Interpretable rule-based modeling
      :link: tabular-interpretability.html

      Fitting interpretable models to data table for understanding data and predictions.

   .. card::
      :title: Training models with GPU support
      :link: tabular-gpu.html

      How to train models with GPU support.

   .. card::
      :title: Multi-Label Prediction
      :link: tabular-multilabel.html

      How to predict multiple columns in a data table.

   .. card::
      :title: Adding a Custom Model
      :link: tabular-custom-model.html

      How to add a custom model to AutoGluon.

   .. card::
      :title: Adding a Custom Model (Advanced)
      :link: tabular-custom-model-advanced.html

      How to add a custom model to AutoGluon (Advanced).

   .. card::
      :title: Adding a Custom Metric
      :link: tabular-custom-metric.html

      How to add a custom metric to AutoGluon.

   .. card::
      :title: Feature Engineering
      :link: tabular-feature-engineering.html

      AutoGluon's default feature engineering and how to extend it.

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
   tabular-multimodal
   tabular-multimodal-text-others
   tabular-interpretability
   tabular-gpu
   tabular-multilabel
   tabular-custom-model
   tabular-custom-model-advanced
   tabular-custom-metric
   tabular-feature-engineering
   tabular-faq

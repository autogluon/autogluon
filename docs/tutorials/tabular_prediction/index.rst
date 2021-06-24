Tabular Prediction
====================


Tabular datasets present data as tables. 
Each row in a table contains an example, while columns store features. 
The column data types can be numbers, strings, datetime or even customized formats such as image filepaths. 
The training dataset contains an additional column storing the labels. 
Our task is to predict this label column by using the rest feature columns. 

In this module, we will go through how to use AutoGluon's ``tabular`` module to automatically train models on tabular datasets. 

.. container:: cards

   .. card::
      :title: Quick Start
      :link: tabular-quickstart.html

      An 5 min tutorial to train models on tabular datasets.

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
   tabular-faq

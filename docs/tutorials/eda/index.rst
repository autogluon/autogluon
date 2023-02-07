Exploratory Data Analysis Tools
===============================

This section provides an overview of exploratory data analysis tools.

.. container:: cards

   .. card::
      :title: Automated Dataset Overview
      :link: eda-auto-dataset-overview.html

      Tool to get a high-level understanding of datasets including basic statistical information and feature information.

   .. card::
      :title: Automated Target Variable Analysis
      :link: eda-auto-target-analysis.html

      Automatically analyze and summarize the variable we are trying to predict and it's relationship with other variables.

   .. card::
      :title: Quick Model Fit
      :link: eda-auto-quick-fit.html

      Automatically analyze and summarize the variable we are trying to predict and it's relationship with other variables.

   .. card::
      :title: Covariate Shift Detection
      :link: eda-auto-covariate-shift.html

      Identify situations where the distribution of features in a dataset changes between the training
      and testing phases, which can lead to biased model performance.

   .. card::
      :title: Feature interaction charts
      :link: eda-auto-analyze-interaction.html

      Simple visualization tool to visualize 1/2/3-way relationships between features.
      The tool automatically picks a chart type given each property type.

.. toctree::
   :maxdepth: 1
   :hidden:

   eda-auto-dataset-overview
   eda-auto-target-analysis
   eda-auto-quick-fit
   eda-auto-covariate-shift
   eda-auto-analyze-interaction
   autogluon.eda.base-apis
   autogluon.eda.auto
   autogluon.eda.dataset
   autogluon.eda.interaction
   autogluon.eda.missing
   autogluon.eda.model
   autogluon.eda.shift
   autogluon.eda.transform

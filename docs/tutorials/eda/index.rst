Exploratory Data Analysis Tools
===============================

This section contains a high-level overview and showcases for exploratory data analysis tools.

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

Main API Reference
------------------

The section contains a reference of base constructs and composite components.

.. container:: cards

   .. card::
      :title: Auto: high-level composite components
      :link: autogluon.eda.auto.html

      Reference for high-level composite components.

   .. card::
      :title: Base APIs
      :link: autogluon.eda.base-apis.html

      Components building blocks APIs.

Low-level components API reference
----------------------------------

The section contains a reference for low-level components.

.. container:: cards

   .. card::
      :title: autogluon.eda.dataset
      :link: autogluon.eda.dataset.html

      Dataset-level APIs

   .. card::
      :title: autogluon.eda.interaction
      :link: autogluon.eda.interaction.html

      Feature-level interactions APIs

   .. card::
      :title: autogluon.eda.missing
      :link: autogluon.eda.missing.html

      Missing data APIs

   .. card::
      :title: autogluon.eda.model
      :link: autogluon.eda.model.html

      Model level APIs

   .. card::
      :title: autogluon.eda.shift
      :link: autogluon.eda.shift.html

      Distribution shift APIs

   .. card::
      :title: autogluon.eda.transform
      :link: autogluon.eda.transform.html

      Transformations APIs

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

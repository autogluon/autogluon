Tutorials
=========



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
      :title: Deployment Optimization
      :link: tabular_prediction/tabular-deployment.html

      Tutorial on optimizing the predictor artifact for production deployment.

   .. card::
      :title: Kaggle Tutorial
      :link: tabular_prediction/tabular-kaggle.html

      Using AutoGluon for Kaggle competitions with tabular data.

   .. card::
      :title: Data Tables Containing Image, Text, and Tabular
      :link: tabular_prediction/tabular-multimodal.html

      Modeling data tables with image, text, numeric, and categorical features.

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
      :link: tabular_prediction/tabular-custom-model.html

      How to add a custom model to AutoGluon.

   .. card::
      :title: Adding a Custom Metric
      :link: tabular_prediction/tabular-custom-metric.html

      How to add a custom metric to AutoGluon.

   .. card::
      :title: FAQ
      :link: tabular_prediction/tabular-faq.html

      Frequently asked questions about AutoGluon-Tabular.


Multimodal Prediction
---------------------
.. container:: cards

   .. card::
      :title: Use AutoGluon Multimodal for Text Prediction: Quick Start
      :link: multimodal/text_prediction/beginner_text.html

      How to train high-quality text prediction models with MultiModalPredictor in under 5 minutes.

   .. card::
      :title: Solving Multilingual Problems
      :link: multimodal/text_prediction/multilingual_text.html

      How to use MultiModalPredictor to build models on datasets with languages other than English.

   .. card::
      :title: Multimodal Data Tables with Text
      :link: multimodal/multimodal_prediction/multimodal_text_tabular.html

      How MultiModalPredictor can be applied to multimodal data tables with a mix of text, numerical, and categorical columns.


Time Series Forecasting
-----------------------
.. container:: cards

   .. card::
      :title: Quick Start
      :link: timeseries/forecasting-quickstart.html

      Quick start tutorial on fitting models with time series datasets.

   .. card::
      :title: In-depth Tutorial
      :link: timeseries/forecasting-indepth.html

      Detailed discussion of the time series forecasting capabilities in AutoGluon.

   .. card::
      :title: FAQ
      :link: timeseries/forecasting-faq.html

      Frequently asked questions about AutoGluon-TimeSeries.


Cloud Training and Deployment
-----------------------------
.. container:: cards

   .. card::
      :title: AutoGluon-Tabular on Amazon SageMaker Autopilot
      :link:  https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-autopilot-is-up-to-eight-times-faster-with-new-ensemble-training-mode-powered-by-autogluon/

      Checkout managed AutoGluon experience on Amazon SageMaker Autopilot

   .. card::
      :title: AutoGluon Cloud
      :link: cloud_fit_deploy/autogluon-cloud.html

      A tutorial on using AutoGluon Cloud module to train/deploy AutoGluon backed models on SageMaker.

   .. card::
      :title: Cloud Training with AWS SageMaker
      :link: cloud_fit_deploy/cloud-aws-sagemaker-training.html

      A tutorial on fitting an AutoGluon model using AWS SageMaker.


   .. card::
      :title: Deploying AutoGluon Models with AWS SageMaker
      :link: cloud_fit_deploy/cloud-aws-sagemaker-deployment.html

      A tutorial on how to deploy trained models using AWS SageMaker and Deep Learning Containers.


   .. card::
      :title: Deploying AutoGluon models with serverless templates
      :link: cloud_fit_deploy/cloud-aws-lambda-deployment.html

      A tutorial on how to deploy trained models using AWS Lambda.

Exploratory Data Analysis
-------------------------

.. container:: cards

   .. card::
      :title: Automated Dataset Overview
      :link: eda/eda-auto-dataset-overview.html

      Tool to get a high-level understanding of datasets including basic statistical information and feature information

   .. card::
      :title: Automated Target Variable Analysis
      :link: eda/eda-auto-target-analysis.html

      Automatically analyze and summarize the variable we are trying to predict and it's relationship with other variables.

   .. card::
      :title: Quick Model Fit
      :link: eda/eda-auto-quick-fit.html

      Automatically analyze and summarize the variable we are trying to predict and it's relationship with other variables.

   .. card::
      :title: Covariate Shift Detection
      :link: eda/eda-auto-covariate-shift.html

      Identify situations where the distribution of features in a dataset changes between the training
      and testing phases, which can lead to biased model performance.

   .. card::
      :title: Feature interaction charts
      :link: eda/eda-auto-analyze-interaction.html

      Simple visualization tool to visualize 1/2/3-way relationships between features.
      The tool automatically picks a chart type given each property type.


.. toctree::
   :maxdepth: 3
   :hidden:

   tabular_prediction/index
   multimodal/index
   cloud_fit_deploy/index
   eda/index

Text Prediction
===============

For predicting the label (either categorical or numerical) of the text data, AutoGluon provides
a simple `fit()` function that  automatically produces high quality text prediction models based
on pretrained networks. Here, the text data may contain sentences, short paragraphs, or multiple sentences.
A single call to `fit()` will train highly accurate neural networks on your provided text dataset,
automatically leveraging accuracy-boosting techniques such as using a
pretrained BERT/ALBERT/ELECTRA model and hyper-parameter optimization.

.. container:: cards

   .. card::
      :title: Quick Start Using FIT
      :link: beginner.html

      Quick start tutorial for text prediction problems.

   .. card::
      :title: Customize Search Space and Try HPO
      :link: customization.html

      Define your own search space and try out different HPO algorithms.

   .. card::
      :title: Text prediction for mixed data types
      :link: heterogeneous.html

      A quick tutorial on how to use TextPrediction to solve problems that have a mix of
      text, numerical, or categorical features.

.. toctree::
   :maxdepth: 1
   :hidden:

   beginner
   customization
   heterogeneous


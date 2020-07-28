Text Prediction
===============

For predicting the label (either categorical or numerical) of the text data, AutoGluon provides
a simple `fit()` function that  automatically produces high quality text prediction models based
on pretrained networks. Here, the text data may contain sentences, short paragraphs, or multiple sentences.
A single call to `fit()` will train highly accurate neural networks on your provided text dataset,
automatically leveraging accuracy-boosting techniques such as using a pretrained BERT model and
hyperparameter optimization.

.. container:: cards

   .. card::
      :title: Quick Start Using FIT
      :link: beginner.html

      Quick start tutorial for text classification.

.. toctree::
   :maxdepth: 1
   :hidden:

   beginner

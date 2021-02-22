Text Prediction
===============

For supervised learning with text data, AutoGluon provides a simple `fit()` function that automatically produces high quality text prediction models. In the data, each training example may be a sentence, short paragraph, or be comprised of multiple text fields (e.g. predicting how similar two sentences are). The target values (labels) to predict may be continuous values (regression) or discrete categories (classification).  
A single call to `fit()` will train highly accurate neural networks on your provided text dataset,
automatically leveraging accuracy-boosting techniques such as fine-tuning a
pretrained BERT/ALBERT/ELECTRA model (transfer learning) and hyperparameter optimization.

.. container:: cards

   .. card::
      :title: Quick Start Using FIT
      :link: beginner.html

      Quickly get started with text prediction problems.

   .. card::
      :title: Multimodal Data Tables with Text
      :link: heterogeneous.html

      Introduction on how TextPredictor can be used on multimodal data tables with
      a mix of text, numeric, and categorical features.

   .. card::
      :title: Customized Hyperparameter Search
      :link: customization.html

      How to do basic customization of the configuration of the network inside TextPredictor,
      and how we can use HPO.

.. toctree::
   :maxdepth: 1
   :hidden:

   beginner
   multimodal_text
   customization


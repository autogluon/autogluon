Text Prediction
===============

For supervised learning with text data, AutoGluon provides a simple `fit()` function that automatically
produces high quality text prediction models (Transformer neural networks). Each training example may be a sentence, a short paragraph, comprised of multiple text fields (e.g. predicting how similar two sentences are), or may even contain additional numeric/categorical features beyond just text. The target values (labels) to predict may be continuous values (regression) or discrete categories (classification).
A single call to `predictor.fit()` will train highly accurate neural networks on your provided text dataset, automatically leveraging accuracy-boosting techniques such as fine-tuning a pretrained NLP  model (transfer learning) and hyperparameter optimization.

.. container:: cards

   .. card::
      :title: Quick Start Using FIT
      :link: beginner.html

      How to train high-quality text prediction models in under 5 minutes.

   .. card::
      :title: Multimodal Data Tables with Text
      :link: multimodal_text.html

      How TextPredictor can be applied to multimodal data tables with
      a mix of text, numeric, and categorical columns.

   .. card::
      :title: Customize TextPredictor Configurations
      :link: customization.html

      How to specify custom hyperparameters of the TextPredictor.

   .. card::
      :title: Solving Multilingual Problems
      :link: multilingual_text.html

      Build models on datasets with languages other than English.

   .. card::
      :title: Getting Started with AutoMM
      :link: automm.html

      How to use AutoMM to train models on single modal and multimodal data.

.. toctree::
   :maxdepth: 1
   :hidden:

   beginner
   multimodal_text
   customization
   multilingual_text
   automm

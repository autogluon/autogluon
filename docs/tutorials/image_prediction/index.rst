Image Prediction (Legacy Version)
=================================

For classifying images based on their content, AutoGluon provides a simple `fit()` function that  automatically produces high quality image classification models.  A single call to `fit()` will train highly accurate neural networks on your provided image dataset, automatically leveraging accuracy-boosting techniques such as transfer learning and hyperparameter optimization on your behalf.

**Note**: AutoGluon ImagePredictor will be deprecated in v0.7. Please try our `AutoGluon Multimodal <https://auto.gluon.ai/stable/tutorials/multimodal/index.html>`_ for more functionalities and better support for your image classification need.

.. container:: cards

   .. card::
      :title: Prepare Dataset for Image Prediction
      :link: dataset.html

      Dataset preparation for Image Prediction

   .. card::
      :title: Quick Start Using FIT
      :link: beginner.html

      Quick start tutorial for image classification.

   .. card::
      :title: Customized Hyperparameter Search
      :link: hpo.html

      More in-depth image classification tutorial,
      including non-default hyperparameters and how to tune them.


.. toctree::
   :maxdepth: 1
   :hidden:

   dataset
   beginner
   hpo

Object Detection
====================

For detecting the presence and location of objects in images, AutoGluon provides a simple `fit()` function that automatically produces high quality object detection models.  A single call to `fit()` will train highly accurate neural networks on your provided image dataset, automatically leveraging accuracy-boosting techniques such as transfer learning and hyperparameter optimization on your behalf.

**Note**: AutoGluon ObjectDetector will be deprecated in v0.7. Please try our `MultiModalPredictor` for more functionalities and better support for your object detection need.

.. container:: cards

   .. card::
      :title: Prepare Dataset for Object Detector
      :link: dataset.html

      Prepare dataset for object detection.

   .. card::
      :title: Quick Start Using FIT
      :link: beginner.html

      Quick start tutorial for object detection.

.. toctree::
   :maxdepth: 1
   :hidden:

   dataset
   beginner

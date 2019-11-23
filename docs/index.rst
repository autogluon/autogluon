AutoGluon: AutoML Toolkit with MXNet Gluon
==========================================

`AutoGluon` enables easy-to-use and easy-to-extend AutoML with a focus on deep learning, and making AutoML deploy in real-world applications. Feature includes:

- Fast prototyping on deep learning applications.
- Automatic Hyper-parameter Optimization.
- Leveraging state-of-the-art deep learning techniques.
- Easy customization for advanced usage.
- Painless setup using distributed computations.

.. note::

   Placeholder for a small demo code using tabular data

Installation
------------

.. include:: install-include.rst


Quick Start
-----------

.. raw:: html
   :file: static/application.html

Tutorials
---------

.. container:: cards

   .. card::
      :title: Image Classification
      :link: tutorials/image_classification/index.html

      Dive into deep Tutorials.

   .. card::
      :title: Object Detection
      :link: tutorials/object_detection/index.html

      Dive into deep Tutorials.

Advanced Topic
~~~~~~~~~~~~~~

.. container:: cards

   .. card::
      :title: Customize AutoGluon
      :link: tutorials/course/index.html

      Advanced Usage & APIs in AutoGluon.

   .. card::
      :title: Neural Architecture Search
      :link: tutorials/nas/index.html

      Neural Architecture Search tutorials.

   .. card::
      :title: For PyTorch Users
      :link: tutorials/torch/index.html

      PyTorch HPO and NAS tutorials.

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/image_classification/index
   tutorials/object_detection/index
   tutorials/course/index
   tutorials/nas/index
   tutorials/torch/index
   api/autogluon.space
   api/autogluon.core
   api/autogluon.task
   api/autogluon.scheduler
   api/autogluon.searcher
   api/autogluon.utils
   model_zoo/index

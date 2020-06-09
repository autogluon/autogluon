Customize AutoGluon
===================

Tutorials for customizing AutoGluon using advanced APIs.

.. container:: cards
   
   .. card::
      :title: Search Space and Decorator
      :link: core.html

      Using the AutoGluon Core API for customized search spaces, searchable objects, 
      and training functions, in order to hyperparameter-tune any model.

   .. card::
      :title: Search Algorithms
      :link: algorithm.html

      Learn how to use AutoGluon's built-in hyperparameter search algorithms, 
      including early-stopping strategies.

   .. card::
      :title: Searchable Customized Objects
      :link: object.html

      Hyperparameter search in customized objects such as your own: neural network, optimizer, dataset, etc.

   .. card::
      :title: Bring Your Own Training Scripts
      :link: script.html

      Tune the hyperparameters of arbitrary Python scripts using AutoGluon.

   .. card::
      :title: Distributed Search Tutorial
      :link: distributed.html

      Easily distribute the hyperparameter search arcross multiple machines to improve efficiency.
   
   .. card::
      :title: Tune a Multi-Layer Perceptron
      :link: mlp.html

      Complete example of how to use AutoGluon and its state-of-the-art schedulers and model-based searchers to tune a MLP.

.. toctree::
   :maxdepth: 1
   :hidden:

   core
   algorithm
   object
   script
   distributed
   mlp

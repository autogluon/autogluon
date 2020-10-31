Tune Custom Models
===================

Tutorials to hyperparameter-tune any custom models or Python code.

.. container:: cards
   
   .. card::
      :title: Search Space and Decorator
      :link: core.html

      Using AutoGluon's Core APIs to hyperparameter-tune any model/code by making existing objects/training-functions 
      searchable.

   .. card::
      :title: Search Algorithms
      :link: algorithm.html

      How to use AutoGluon's built-in hyperparameter search algorithms, 
      including early-stopping strategies.

   .. card::
      :title: Searchable Objects
      :link: object.html

      Tune the hyperparameters of custom objects such as your own: neural network, optimizer, dataset, etc.

   .. card::
      :title: Tune Training Scripts
      :link: script.html

      Tune the argument values (hyperparameters) of arbitrary Python scripts using AutoGluon.

   .. card::
      :title: Distributed Search
      :link: distributed.html

      Easily distribute the hyperparameter search across multiple machines to improve efficiency.
   
   .. card::
      :title: Example: Tune a Multi-Layer Perceptron
      :link: mlp.html

      Complete example of using AutoGluon's state-of-the-art hyperparameter optimization to tune a basic MLP model.

.. toctree::
   :maxdepth: 1
   :hidden:

   core
   algorithm
   object
   script
   distributed
   mlp

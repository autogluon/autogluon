.. role:: hidden
    :class: hidden-section

autogluon.common.space
==================

Search Space
------------------

.. automodule:: autogluon.common.space
.. currentmodule:: autogluon.common.space

You can use AutoGluon search space to perform HPO. For a high-level overview, see this example:

.. code-blcok:: python

   from autogluon.common import space

   categorical_space = space.Categorical('a', 'b', 'c', 'd')  # Nested search space for hyperparameters which are categorical.
   real_space = space.Real(0.01, 0.1)  # Search space for numeric hyperparameter that takes continuous values
   int_space = space.Int(0, 100)  # Search space for numeric hyperparameter that takes integer values
   bool_space = space. Bool()  # Search space for hyperparameter that is either True or False.

.. autosummary::
   :nosignatures:

   Categorical
   Real
   Int
   Bool

:hidden:`Categorical`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Categorical
   :members: init

:hidden:`Real`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Real
   :members: init

:hidden:`Int`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Int
   :members: init

:hidden:`Bool`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Bool
   :members: init

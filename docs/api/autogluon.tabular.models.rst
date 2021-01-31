.. role:: hidden
    :class: hidden-section

autogluon.tabular.models
========================

.. note::

   This documentation is for advanced users, and is not comprehensive.

   For a stable public API, refer to TabularPredictor.

Models
------

.. automodule:: autogluon.tabular.models
.. currentmodule:: autogluon.tabular.models

.. autosummary::
   :nosignatures:

   AbstractModel
   LGBModel
   CatBoostModel
   XGBoostModel
   RFModel
   XTModel
   KNNModel
   LinearModel
   TabularNeuralNetModel
   NNFastAiTabularModel

:hidden:`AbstractModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractModel
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: AbstractModel
        :methods:

:hidden:`LGBModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LGBModel
   :members: init

:hidden:`CatBoostModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CatBoostModel
   :members: init

:hidden:`XGBoostModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: XGBoostModel
   :members: init

:hidden:`RFModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RFModel
   :members: init

:hidden:`XTModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: XTModel
   :members: init

:hidden:`KNNModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KNNModel
   :members: init

:hidden:`LinearModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearModel
   :members: init

:hidden:`TabularNeuralNetModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularNeuralNetModel
   :members: init

:hidden:`NNFastAiTabularModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NNFastAiTabularModel
   :members: init

Ensemble Models
---------------

.. automodule:: autogluon.core.models
.. currentmodule:: autogluon.core.models

.. autosummary::
   :nosignatures:

   BaggedEnsembleModel
   StackerEnsembleModel
   WeightedEnsembleModel

:hidden:`BaggedEnsembleModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BaggedEnsembleModel
   :members: init

:hidden:`StackerEnsembleModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StackerEnsembleModel
   :members: init

:hidden:`WeightedEnsembleModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: WeightedEnsembleModel
   :members: init

Experimental Models
-------------------

.. automodule:: autogluon.tabular.models
.. currentmodule:: autogluon.tabular.models

.. autosummary::
   :nosignatures:

   FastTextModel
   TextPredictionV1Model

:hidden:`FastTextModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FastTextModel
   :members: init

:hidden:`TextPredictionV1Model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextPredictionV1Model
   :members: init
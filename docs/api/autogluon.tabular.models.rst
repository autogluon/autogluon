.. role:: hidden
    :class: hidden-section

autogluon.tabular.models
========================

.. note::

   This documentation is for advanced users, and is not comprehensive.

   For a stable public API, refer to TabularPredictor.

Model Name Suffixes
-------------------

Models trained by TabularPredictor can have suffixes in their names that have special meanings.

The suffixes are as follows:

**"_Lx"**: Indicates the stack level (x) the model is trained in, such as "_L1", "_L2", etc.
A model with "_L1" suffix is a base model, meaning it does not depend on any other models.
If a model lacks this suffix, then it is a base model and is at level 1 ("_L1").

**"/Tx"**: Indicates that the model was trained via hyperparameter search (HPO). Tx is shorthand for HPO trial #x.
An example would be **"LightGBM/T8"**.

**"_BAG"**: Indicates that the model is a bagged ensemble.
A bagged ensemble contains multiple instances of the model (children) trained with different subsets of the data.
During inference, these child models each predict on the data and their predictions are averaged in the final result.
This typically achieves a stronger result than any of the individual models alone,
but slows down inference speed significantly. Refer to **"_FULL"** for instructions on how to improve inference speed.

**"_FULL"**: Indicates the model has been refit via TabularPredictor's refit_full method.
This model will have no validation score because all of the data (train and validation) was used as training data.
Usually, there will be another model with the same name as this model minus the "_FULL" suffix.
Often, this model can outperform the original model because of using more data during training,
but is usually weaker if the original was a bagged ensemble ("_BAG"), but with much faster inference speed.

**"_DSTL"**: Indicates the model was created through model distillation
via a call to TabularPredictor's distill method.
Validation scores of distilled models should only be compared against other distilled models.

**"_x"**: Indicates that the name without this added suffix already existed in a different model,
so this suffix was added to avoid overwriting the pre-existing model.
An example would be **"LightGBM_2"**.

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
   TabularNeuralNetTorchModel
   NNFastAiTabularModel
   VowpalWabbitModel
   MultiModalPredictorModel
   TextPredictorModel
   ImagePredictorModel

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

:hidden:`TabularNeuralNetTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularNeuralNetTorchModel
   :members: init

:hidden:`NNFastAiTabularModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NNFastAiTabularModel
   :members: init

:hidden:`VowpalWabbitModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VowpalWabbitModel
   :members: init

:hidden:`MultiModalPredictorModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiModalPredictorModel
   :members: init

:hidden:`TextPredictorModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextPredictorModel
   :members: init

:hidden:`ImagePredictorModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ImagePredictorModel
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

   FTTransformerModel
   TabPFNModel
   FastTextModel

:hidden:`FTTransformerModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FTTransformerModel
   :members: init

:hidden:`TabPFNModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabPFNModel
   :members: init

:hidden:`FastTextModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FastTextModel
   :members: init

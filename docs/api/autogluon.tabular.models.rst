.. role:: hidden
    :class: hidden-section

autogluon.tabular.models
========================

.. note::

   This documentation is for advanced users, and is not comprehensive.

   For a stable public API, refer to TabularPredictor.

Model Keys
-------------------

To fit a model with TabularPredictor, you must specify it in the `TabularPredictor.fit` `hyperparameters` argument.

`hyperparameters` takes in a dictionary of models, where each key is a model name, and the values are a list of dictionaries of model hyperparameters.

For example:

.. code-block:: python

    hyperparameters = {
        'NN_TORCH': {},
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
        ],
        'CAT': {},
        'XGB': {},
        'EBM': {},
        'FASTAI': {},
        'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
    }

Here is the mapping of keys to models:

.. code-block:: python
    MODEL_TYPES = {
        "RF": RFModel,
        "XT": XTModel,
        "KNN": KNNModel,
        "GBM": LGBModel,
        "CAT": CatBoostModel,
        "XGB": XGBoostModel,
        "EBM": EBMModel,
        "REALMLP": RealMLPModel,
        "MITRA": MitraModel,
        "TABICL": TabICLModel,
        "TABPFNV2": TabPFNV2Model,
        "NN_TORCH": TabularNeuralNetTorchModel,
        "LR": LinearModel,
        "FASTAI": NNFastAiTabularModel,
        "TABM": TabMModel,
        "AG_TEXT_NN": TextPredictorModel,
        "AG_IMAGE_NN": ImagePredictorModel,
        "AG_AUTOMM": MultiModalPredictorModel,
        "FT_TRANSFORMER": FTTransformerModel,
        "FASTTEXT": FastTextModel,
        "ENS_WEIGHTED": GreedyWeightedEnsembleModel,
        "SIMPLE_ENS_WEIGHTED": SimpleWeightedEnsembleModel,

        # interpretable models
        "IM_RULEFIT": RuleFitModel,
        "IM_GREEDYTREE": GreedyTreeModel,
        "IM_FIGS": FigsModel,
        "IM_HSTREE": HSTreeModel,
        "IM_BOOSTEDRULES": BoostedRulesModel,

        "DUMMY": DummyModel,
    }

Here is the mapping of model types to their default names when trained:

.. code-block:: python

    DEFAULT_MODEL_NAMES = {
        RFModel: 'RandomForest',
        XTModel: 'ExtraTrees',
        KNNModel: 'KNeighbors',
        LGBModel: 'LightGBM',
        CatBoostModel: 'CatBoost',
        XGBoostModel: 'XGBoost',
        EBMModel: 'EBM',
        RealMLPModel: 'RealMLP',
        TabMModel: 'TabM',
        MitraModel: 'Mitra',
        TabICLModel: 'TabICL',
        TabPFNV2Model: 'TabPFNv2',
        TabularNeuralNetTorchModel: 'NeuralNetTorch',
        LinearModel: 'LinearModel',
        NNFastAiTabularModel: 'NeuralNetFastAI',
        TextPredictorModel: 'TextPredictor',
        ImagePredictorModel: 'ImagePredictor',
        MultiModalPredictorModel: 'MultiModalPredictor',

        FTTransformerModel: 'FTTransformer',
        FastTextModel: 'FastText',
        GreedyWeightedEnsembleModel: 'WeightedEnsemble',
        SimpleWeightedEnsembleModel: 'WeightedEnsemble',

        # Interpretable models
        RuleFitModel: 'RuleFit',
        GreedyTreeModel: 'GreedyTree',
        FigsModel: 'Figs',
        HSTreeModel: 'HierarchicalShrinkageTree',
        BoostedRulesModel: 'BoostedRules',
    }

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
   EBMModel
   RealMLPModel
   TabMModel
   MitraModel
   TabICLModel
   TabPFNV2Model
   RFModel
   XTModel
   KNNModel
   LinearModel
   TabularNeuralNetTorchModel
   NNFastAiTabularModel
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

:hidden:`EBMModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EBMModel
   :members: init

:hidden:`RealMLPModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RealMLPModel
   :members: init

:hidden:`TabMModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabMModel
   :members: init

:hidden:`MitraModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MitraModel
   :members: init

:hidden:`TabICLModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabICLModel
   :members: init

:hidden:`TabPFNV2Model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabPFNV2Model
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
   FastTextModel

:hidden:`FTTransformerModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FTTransformerModel
   :members: init

:hidden:`FastTextModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FastTextModel
   :members: init

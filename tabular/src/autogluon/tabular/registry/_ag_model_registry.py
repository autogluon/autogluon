from autogluon.core.models import (
    DummyModel,
    GreedyWeightedEnsembleModel,
    SimpleWeightedEnsembleModel,
)

from . import ModelRegistry
from ..models import (
    BoostedRulesModel,
    CatBoostModel,
    FastTextModel,
    FigsModel,
    FTTransformerModel,
    GreedyTreeModel,
    HSTreeModel,
    ImagePredictorModel,
    KNNModel,
    LGBModel,
    LinearModel,
    MultiModalPredictorModel,
    NNFastAiTabularModel,
    RFModel,
    RuleFitModel,
    TabPFNMixModel,
    TabPFNModel,
    TabularNeuralNetTorchModel,
    TextPredictorModel,
    XGBoostModel,
    XTModel,
)


# When adding a new model officially to AutoGluon, the model class should be added to the bottom of this list.
REGISTERED_MODEL_CLS_LST = [
    RFModel,
    XTModel,
    KNNModel,
    LGBModel,
    CatBoostModel,
    XGBoostModel,
    TabularNeuralNetTorchModel,
    LinearModel,
    NNFastAiTabularModel,
    TextPredictorModel,
    ImagePredictorModel,
    MultiModalPredictorModel,
    FTTransformerModel,
    TabPFNModel,
    TabPFNMixModel,
    FastTextModel,
    GreedyWeightedEnsembleModel,
    SimpleWeightedEnsembleModel,
    RuleFitModel,
    GreedyTreeModel,
    FigsModel,
    HSTreeModel,
    BoostedRulesModel,
    DummyModel,
]

# TODO: Replace logic in `autogluon.tabular.trainer.model_presets.presets` with `ag_model_registry`
ag_model_registry = ModelRegistry(model_cls_list=REGISTERED_MODEL_CLS_LST)

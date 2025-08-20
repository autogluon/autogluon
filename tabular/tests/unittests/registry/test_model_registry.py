from __future__ import annotations

from types import MappingProxyType
from typing import Type

import pytest

from autogluon.core.models import (
    AbstractModel,
    DummyModel,
    GreedyWeightedEnsembleModel,
    SimpleWeightedEnsembleModel,
)
from autogluon.tabular.registry import ag_model_registry, ModelRegistry

from autogluon.tabular.models import (
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
    RealMLPModel,
    RFModel,
    RuleFitModel,
    TabICLModel,
    TabMModel,
    TabPFNMixModel,
    MitraModel,
    TabPFNV2Model,
    TabularNeuralNetTorchModel,
    TextPredictorModel,
    XGBoostModel,
    XTModel,
    EBMModel,
)


EXPECTED_MODEL_KEYS = {
    RFModel: "RF",
    XTModel: "XT",
    KNNModel: "KNN",
    LGBModel: "GBM",
    CatBoostModel: "CAT",
    XGBoostModel: "XGB",
    RealMLPModel: "REALMLP",
    TabularNeuralNetTorchModel: "NN_TORCH",
    LinearModel: "LR",
    NNFastAiTabularModel: "FASTAI",
    TextPredictorModel: "AG_TEXT_NN",
    ImagePredictorModel: "AG_IMAGE_NN",
    MultiModalPredictorModel: "AG_AUTOMM",
    FTTransformerModel: "FT_TRANSFORMER",
    TabICLModel: "TABICL",
    TabMModel: "TABM",
    TabPFNV2Model: "TABPFNV2",
    TabPFNMixModel: "TABPFNMIX",
    MitraModel: "MITRA",
    FastTextModel: "FASTTEXT",
    GreedyWeightedEnsembleModel: "ENS_WEIGHTED",
    SimpleWeightedEnsembleModel: "SIMPLE_ENS_WEIGHTED",
    RuleFitModel: "IM_RULEFIT",
    GreedyTreeModel: "IM_GREEDYTREE",
    FigsModel: "IM_FIGS",
    HSTreeModel: "IM_HSTREE",
    BoostedRulesModel: "IM_BOOSTEDRULES",
    DummyModel: "DUMMY",
    EBMModel: "EBM",
}

EXPECTED_MODEL_NAMES = {
    RFModel: "RandomForest",
    XTModel: "ExtraTrees",
    KNNModel: "KNeighbors",
    LGBModel: "LightGBM",
    CatBoostModel: "CatBoost",
    XGBoostModel: "XGBoost",
    RealMLPModel: "RealMLP",
    TabularNeuralNetTorchModel: "NeuralNetTorch",
    LinearModel: "LinearModel",
    NNFastAiTabularModel: "NeuralNetFastAI",
    TextPredictorModel: "TextPredictor",
    ImagePredictorModel: "ImagePredictor",
    MultiModalPredictorModel: "MultiModalPredictor",
    FTTransformerModel: "FTTransformer",
    TabICLModel: "TabICL",
    TabMModel: "TabM",
    TabPFNV2Model: "TabPFNv2",
    TabPFNMixModel: "TabPFNMix",
    MitraModel: "Mitra",
    FastTextModel: "FastText",
    GreedyWeightedEnsembleModel: "WeightedEnsemble",
    SimpleWeightedEnsembleModel: "WeightedEnsemble",
    RuleFitModel: "RuleFit",
    GreedyTreeModel: "GreedyTree",
    FigsModel: "Figs",
    HSTreeModel: "HierarchicalShrinkageTree",
    BoostedRulesModel: "BoostedRules",
    DummyModel: "Dummy",
    EBMModel: "EBMModel",
}

# Higher values indicate higher priority, priority dictates the order models are trained for a given level.
EXPECTED_MODEL_PRIORITY = {
    RFModel: 80,
    XTModel: 60,
    KNNModel: 100,
    LGBModel: 90,
    CatBoostModel: 70,
    XGBoostModel: 40,
    RealMLPModel: 75,
    TabularNeuralNetTorchModel: 25,
    LinearModel: 30,
    EBMModel: 35,
    NNFastAiTabularModel: 50,
    TextPredictorModel: 0,
    ImagePredictorModel: 0,
    MultiModalPredictorModel: 0,
    FTTransformerModel: 0,
    TabICLModel: 65,
    TabMModel: 85,
    TabPFNV2Model: 105,
    TabPFNMixModel: 45,
    MitraModel: 55,
    FastTextModel: 0,
    GreedyWeightedEnsembleModel: 0,
    SimpleWeightedEnsembleModel: 0,
    RuleFitModel: 0,
    GreedyTreeModel: 0,
    FigsModel: 0,
    HSTreeModel: 0,
    BoostedRulesModel: 0,
    DummyModel: 0,
}

EXPECTED_MODEL_PRIORITY_BY_PROBLEM_TYPE = {
    LGBModel: {
        "softclass": 100,
    },
    CatBoostModel: {
        "softclass": 60,
    },
    NNFastAiTabularModel: {
        "multiclass": 95,
    },
}

EXPECTED_REGISTERED_MODEL_CLS_LST = list(EXPECTED_MODEL_NAMES.keys())
REGISTERED_MODEL_CLS_LST = ag_model_registry.model_cls_list


@pytest.mark.parametrize(
    "model_cls",
    REGISTERED_MODEL_CLS_LST,
    ids=[c.__name__ for c in REGISTERED_MODEL_CLS_LST],
)  # noqa
def test_model_cls_key(model_cls: Type[AbstractModel]):
    expected_model_key = EXPECTED_MODEL_KEYS[model_cls]
    assert expected_model_key == model_cls.ag_key


def verify_registry(model_cls: Type[AbstractModel], model_registry: ModelRegistry):
    """
    Verifies that all methods work as intended in ModelRegistry, assuming `model_cls` is already registered.
    """
    assert model_registry.exists(model_cls)
    assert model_cls.ag_key == model_registry.key(model_cls)
    assert model_cls.ag_priority == model_registry.priority(model_cls)

    assert model_cls in model_registry.model_cls_list
    assert model_cls.ag_key in model_registry.keys

    assert model_cls == model_registry.key_to_cls(model_registry.key(model_cls))
    key_to_cls_map = model_registry.key_to_cls_map()
    assert model_cls == key_to_cls_map[model_cls.ag_key]

    name_map = model_registry.name_map()
    assert model_cls.ag_name == name_map[model_cls]
    assert model_cls.ag_name == model_registry.name(model_cls)

    priority_map = model_registry.priority_map()
    assert model_cls.ag_priority == priority_map[model_cls]
    for problem_type in model_cls.ag_priority_by_problem_type:
        priority_map_problem_type = model_registry.priority_map(problem_type=problem_type)
        assert model_cls.get_ag_priority(problem_type) == model_registry.priority(model_cls, problem_type=problem_type)
        assert model_cls.get_ag_priority(problem_type) == priority_map_problem_type[model_cls]


@pytest.mark.parametrize(
    "model_cls",
    REGISTERED_MODEL_CLS_LST,
    ids=[c.__name__ for c in REGISTERED_MODEL_CLS_LST],
)  # noqa
def test_model_registry(model_cls: Type[AbstractModel]):
    """
    Verifies that all methods work as intended in ModelRegistry.
    """
    verify_registry(model_cls=model_cls, model_registry=ag_model_registry)


def test_model_registry_new():
    """
    Verifies that all methods work as intended in a new ModelRegistry.
    """
    model_registry_new = ModelRegistry()

    assert not model_registry_new.exists(RFModel)
    assert model_registry_new.model_cls_list == []
    assert model_registry_new.keys == []
    assert model_registry_new.name_map() == {}
    assert model_registry_new.priority_map() == {}

    model_registry_new.add(RFModel)
    assert model_registry_new.model_cls_list == [RFModel]
    verify_registry(model_cls=RFModel, model_registry=model_registry_new)

    model_registry_new.remove(model_cls=RFModel)
    assert not model_registry_new.exists(RFModel)
    assert model_registry_new.model_cls_list == []
    assert model_registry_new.keys == []
    assert model_registry_new.name_map() == {}
    assert model_registry_new.priority_map() == {}


def test_no_unknown_model_cls_registered():
    assert set(ag_model_registry.model_cls_list) == set(EXPECTED_REGISTERED_MODEL_CLS_LST)


@pytest.mark.parametrize(
    "model_cls",
    REGISTERED_MODEL_CLS_LST,
    ids=[c.__name__ for c in REGISTERED_MODEL_CLS_LST],
)  # noqa
def test_model_cls_name(model_cls: Type[AbstractModel]):
    expected_model_name = EXPECTED_MODEL_NAMES[model_cls]
    assert expected_model_name == model_cls.ag_name


@pytest.mark.parametrize(
    "model_cls",
    REGISTERED_MODEL_CLS_LST,
    ids=[c.__name__ for c in REGISTERED_MODEL_CLS_LST],
)  # noqa
def test_model_cls_priority(model_cls: Type[AbstractModel]):
    expected_model_priority = EXPECTED_MODEL_PRIORITY[model_cls]
    assert expected_model_priority == model_cls.ag_priority


@pytest.mark.parametrize(
    "model_cls",
    REGISTERED_MODEL_CLS_LST,
    ids=[c.__name__ for c in REGISTERED_MODEL_CLS_LST],
)  # noqa
def test_model_cls_priority_by_problem_type(model_cls: Type[AbstractModel]):
    expected_model_priority_by_problem_type = EXPECTED_MODEL_PRIORITY_BY_PROBLEM_TYPE.get(model_cls, {})
    expected_model_priority_default = EXPECTED_MODEL_PRIORITY[model_cls]
    assert expected_model_priority_by_problem_type == model_cls.ag_priority_by_problem_type
    assert isinstance(model_cls.ag_priority_by_problem_type, MappingProxyType)
    for problem_type in ["binary", "multiclass", "regression", "quantile", "softclass"]:
        expected_model_priority = expected_model_priority_by_problem_type.get(problem_type, expected_model_priority_default)
        model_priority = model_cls.get_ag_priority(problem_type=problem_type)
        assert expected_model_priority == model_priority
    assert expected_model_priority_default == model_cls.get_ag_priority()


def test_model_cls_all_present():
    assert len(REGISTERED_MODEL_CLS_LST) == len(set(REGISTERED_MODEL_CLS_LST))
    assert set(EXPECTED_REGISTERED_MODEL_CLS_LST) == set(REGISTERED_MODEL_CLS_LST)


def test_model_cls_no_duplicate_keys():
    keys = set()
    for c in REGISTERED_MODEL_CLS_LST:
        if c.ag_key in keys:
            raise AssertionError(f"Two model classes cannot share the same key: {c.ag_key}")
        keys.add(c.ag_key)

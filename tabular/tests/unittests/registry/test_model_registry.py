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
from autogluon.tabular.models import (
    BoostedRulesModel,
    CatBoostModel,
    CausiloModel,
    EBMModel,
    FigsModel,
    FTTransformerModel,
    GreedyTreeModel,
    HSTreeModel,
    ImagePredictorModel,
    KNNModel,
    LGBModel,
    LinearModel,
    MitraModel,
    MultiModalPredictorModel,
    NNFastAiTabularModel,
    NoriModel,
    PrepLGBModel,
    RealMLPModel,
    RealTabPFNv2Model,
    RealTabPFNv25Model,
    RFModel,
    RuleFitModel,
    TabDPTModel,
    TabDPTTurboModel,
    TabFMModel,
    TabICLModel,
    TabMModel,
    TabPFN3Model,
    TabPFN35Model,
    TabPFNMixModel,
    TabPFNv26Model,
    TabularNeuralNetTorchModel,
    TextPredictorModel,
    XGBoostModel,
    XTModel,
)
from autogluon.tabular.registry import ModelRegistry, ag_model_registry

EXPECTED_MODEL_KEYS = {
    RFModel: "RF",
    XTModel: "XT",
    KNNModel: "KNN",
    LGBModel: "GBM",
    PrepLGBModel: "GBM_PREP",
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
    TabDPTModel: "TABDPT",
    TabDPTTurboModel: "TABDPT-TURBO",
    TabICLModel: "TABICL",
    TabMModel: "TABM",
    TabPFNMixModel: "TABPFNMIX",
    TabPFNv26Model: "TABPFN-2.6",
    TabPFN3Model: "TABPFN-3",
    TabPFN35Model: "TABPFN-3.5",
    CausiloModel: "CAUSILO",
    TabFMModel: "TABFM",
    MitraModel: "MITRA",
    NoriModel: "NORI",
    GreedyWeightedEnsembleModel: "ENS_WEIGHTED",
    SimpleWeightedEnsembleModel: "SIMPLE_ENS_WEIGHTED",
    RuleFitModel: "IM_RULEFIT",
    GreedyTreeModel: "IM_GREEDYTREE",
    FigsModel: "IM_FIGS",
    HSTreeModel: "IM_HSTREE",
    BoostedRulesModel: "IM_BOOSTEDRULES",
    DummyModel: "DUMMY",
    EBMModel: "EBM",
    RealTabPFNv25Model: "REALTABPFN-V2.5",
    RealTabPFNv2Model: "REALTABPFN-V2",
}

EXPECTED_MODEL_NAMES = {
    RFModel: "RandomForest",
    XTModel: "ExtraTrees",
    KNNModel: "KNeighbors",
    LGBModel: "LightGBM",
    PrepLGBModel: "LightGBMPrep",
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
    TabDPTModel: "TabDPT",
    TabDPTTurboModel: "TabDPT-Turbo",
    TabICLModel: "TabICL",
    TabMModel: "TabM",
    TabPFNMixModel: "TabPFNMix",
    TabPFNv26Model: "TabPFN-2.6",
    TabPFN3Model: "TabPFN-3",
    TabPFN35Model: "TabPFN-3.5",
    CausiloModel: "Causilo",
    TabFMModel: "TabFM",
    MitraModel: "Mitra",
    NoriModel: "Nori",
    GreedyWeightedEnsembleModel: "WeightedEnsemble",
    SimpleWeightedEnsembleModel: "WeightedEnsemble",
    RuleFitModel: "RuleFit",
    GreedyTreeModel: "GreedyTree",
    FigsModel: "Figs",
    HSTreeModel: "HierarchicalShrinkageTree",
    BoostedRulesModel: "BoostedRules",
    DummyModel: "Dummy",
    EBMModel: "EBM",
    RealTabPFNv25Model: "RealTabPFN-v2.5",
    RealTabPFNv2Model: "RealTabPFN-v2",
}

# Higher values indicate higher priority, priority dictates the order models are trained for a given level.
EXPECTED_MODEL_PRIORITY = {
    RFModel: 80,
    XTModel: 60,
    KNNModel: 100,
    LGBModel: 90,
    PrepLGBModel: 90,
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
    TabDPTModel: 50,
    TabDPTTurboModel: 50,
    TabICLModel: 65,
    TabMModel: 85,
    TabPFNMixModel: 45,
    TabPFNv26Model: 40,
    TabPFN3Model: 40,
    TabPFN35Model: 40,
    CausiloModel: 40,
    TabFMModel: 40,
    MitraModel: 55,
    NoriModel: 40,
    GreedyWeightedEnsembleModel: 0,
    SimpleWeightedEnsembleModel: 0,
    RuleFitModel: 0,
    GreedyTreeModel: 0,
    FigsModel: 0,
    HSTreeModel: 0,
    BoostedRulesModel: 0,
    DummyModel: 0,
    RealTabPFNv25Model: 40,
    RealTabPFNv2Model: 40,
}

EXPECTED_MODEL_PRIORITY_BY_PROBLEM_TYPE = {
    LGBModel: {
        "softclass": 100,
    },
    PrepLGBModel: {
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
        expected_model_priority = expected_model_priority_by_problem_type.get(
            problem_type, expected_model_priority_default
        )
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


def test_model_cls_no_duplicate_aliases():
    keys_and_aliases = set()
    for c in REGISTERED_MODEL_CLS_LST:
        assert c.ag_key not in keys_and_aliases
        keys_and_aliases.add(c.ag_key)
        for alias in ag_model_registry.aliases(c):
            assert alias not in keys_and_aliases
            keys_and_aliases.add(alias)


def test_model_registry_aliases():
    assert ag_model_registry.aliases(RealTabPFNv2Model) == ("TABPFNV2",)
    assert ag_model_registry.aliases_map.get("TABPFNV2") == "REALTABPFN-V2"
    assert ag_model_registry.key_to_cls("TABPFNV2") == RealTabPFNv2Model
    assert "TABPFNV2" in ag_model_registry.all_keys

    # Test custom model with aliases in an isolated registry
    class CustomModel(AbstractModel):
        ag_key = "CUSTOM_PRIMARY"
        ag_name = "CustomModel"
        ag_priority = 10
        ag_aliases = ("CUSTOM_ALIAS_1", "CUSTOM_ALIAS_2")

    registry = ModelRegistry()
    registry.add(CustomModel)
    assert registry.key_to_cls("CUSTOM_PRIMARY") == CustomModel
    assert registry.key_to_cls("CUSTOM_ALIAS_1") == CustomModel
    assert registry.key_to_cls("CUSTOM_ALIAS_2") == CustomModel
    assert registry.aliases_map["CUSTOM_ALIAS_1"] == "CUSTOM_PRIMARY"
    assert registry.aliases_map["CUSTOM_ALIAS_2"] == "CUSTOM_PRIMARY"

    # Test alias collision with existing key
    class CollidingKeyModel(AbstractModel):
        ag_key = "CUSTOM_ALIAS_1"
        ag_name = "Colliding"
        ag_priority = 10

    with pytest.raises(AssertionError):
        registry.add(CollidingKeyModel)

    # Test alias collision with existing alias
    class CollidingAliasModel(AbstractModel):
        ag_key = "ANOTHER_KEY"
        ag_name = "CollidingAlias"
        ag_priority = 10
        ag_aliases = ("CUSTOM_ALIAS_2",)

    with pytest.raises(AssertionError):
        registry.add(CollidingAliasModel)

    registry.remove(CustomModel)
    assert not registry.exists(CustomModel)
    assert "CUSTOM_PRIMARY" not in registry.all_keys
    assert "CUSTOM_ALIAS_1" not in registry.all_keys
    assert "CUSTOM_ALIAS_2" not in registry.all_keys


def test_presets_with_tabpfnv2_alias():
    from autogluon.tabular.trainer.model_presets.presets import get_preset_models

    models, _ = get_preset_models(
        path="dummy_path",
        problem_type="binary",
        eval_metric="accuracy",
        hyperparameters={"TABPFNV2": {}},
    )
    assert len(models) == 1
    model = models[0]
    assert isinstance(model, RealTabPFNv2Model)
    assert model.name.startswith("RealTabPFN-v2")

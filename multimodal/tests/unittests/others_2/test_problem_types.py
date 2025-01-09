import numpy as np
import pandas as pd
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BINARY, CLASSIFICATION, MULTICLASS, NER, OBJECT_DETECTION, REGRESSION
from autogluon.multimodal.data import infer_problem_type
from autogluon.multimodal.utils.problem_types import PROBLEM_TYPES_REG


@pytest.mark.parametrize(
    "name",
    [
        "classification",
        "multiclass",
        "binary",
        "regression",
        "ner",
        "named_entity_recognition",
        "object_detection",
        "text_similarity",
        "image_similarity",
        "image_text_similarity",
        "feature_extraction",
        "zero_shot_image_classification",
        "few_shot_classification",
    ],
)
def test_get_problem_type(name):
    problem_prop = PROBLEM_TYPES_REG.get(name)
    assert problem_prop.name == PROBLEM_TYPES_REG.get(problem_prop.name).name


@pytest.mark.parametrize("name", PROBLEM_TYPES_REG.list_keys())
def test_problem_type_in_init(name):
    if name != OBJECT_DETECTION:
        predictor = MultiModalPredictor(problem_type=name)
        assert predictor.problem_type == PROBLEM_TYPES_REG.get(name).name


@pytest.mark.parametrize(
    "y_data,provided_problem_type,gt_problem_type",
    [
        (pd.Series([0, 1, 0, 1, 1, 0]), None, BINARY),
        (pd.Series(["a", "b", "c"]), None, MULTICLASS),
        (pd.Series(["a", "b", "c"]), CLASSIFICATION, MULTICLASS),
        (pd.Series(np.linspace(0.0, 1.0, 100)), None, REGRESSION),
        (pd.Series(["0", "1", "2", 3, 4, 5, 5, 5, 0]), None, MULTICLASS),
        (None, NER, NER),
        (None, OBJECT_DETECTION, OBJECT_DETECTION),
    ],
)
def test_infer_problem_type(y_data, provided_problem_type, gt_problem_type):
    inferred_problem_type = infer_problem_type(
        y_train_data=y_data,
        provided_problem_type=provided_problem_type,
    )
    assert inferred_problem_type == gt_problem_type

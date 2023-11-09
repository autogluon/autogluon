import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG


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
def test_problem_type_in_predictor(name):
    predictor = MultiModalPredictor(problem_type=name)
    assert predictor.problem_type == PROBLEM_TYPES_REG.get(name).name

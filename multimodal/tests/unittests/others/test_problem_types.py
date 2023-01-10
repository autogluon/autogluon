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
        "few_shot_text_classification",
        "ocr_text_detection",
        "ocr_text_recognition",
    ],
)
def test_get_problem_type(name):
    problem_prop = PROBLEM_TYPES_REG.get(name)
    print(problem_prop)
    assert problem_prop.name == PROBLEM_TYPES_REG.get(problem_prop.name).name


@pytest.mark.parametrize("name", PROBLEM_TYPES_REG.list_keys())
def fetch_predictor_via_problem_types(name):
    predictor = MultiModalPredictor(problem_type=name)

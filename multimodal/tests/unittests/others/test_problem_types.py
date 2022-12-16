import pytest
from autogluon.multimodal.problem_types import problem_type_reg


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
    problem_type = problem_type_reg.get(name)
    print(problem_type)

import numpy.testing as npt
import pandas as pd
import pytest

from autogluon.multimodal import MultiModalPredictor


@pytest.mark.parametrize(
    "model_name", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
)
def test_sentence_transformer_embedding(model_name):
    predictor = MultiModalPredictor(
        problem_type="feature_extraction", hyperparameters={"model.hf_text.checkpoint_name": model_name}
    )
    case1 = {"sentence": ["Hello world"]}
    case2 = {"sentence": ["Hello world", "Test Hello World"]}

    outputs_case1_from_df = predictor.extract_embedding(pd.DataFrame(case1))
    outputs_case1_from_dict = predictor.extract_embedding(case1)

    npt.assert_allclose(outputs_case1_from_dict["sentence"], outputs_case1_from_df["sentence"])

    outputs_case2_from_df = predictor.extract_embedding(pd.DataFrame(case2))
    outputs_case2_from_dict = predictor.extract_embedding(case2)
    npt.assert_allclose(outputs_case2_from_df["sentence"], outputs_case2_from_dict["sentence"], 1e-3, 1e-3)
    npt.assert_allclose(outputs_case2_from_df["sentence"][:1], outputs_case1_from_df["sentence"], 1e-3, 1e-3)

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from autogluon.eda.analysis import ShapAnalysis
from autogluon.eda.auto import analyze, quick_fit

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


@pytest.mark.parametrize(
    "label, expected_task_type",
    [
        ("class", "binary"),
        ("fnlwgt", "regression"),
    ],
)
def test_ShapAnalysis(label, expected_task_type, monkeypatch):
    df_train = pd.read_csv(os.path.join(RESOURCE_PATH, "adult", "train_data.csv")).sample(30, random_state=0)
    with tempfile.TemporaryDirectory() as path:
        state = quick_fit(
            estimator_args=dict(path=path),
            train_data=df_train,
            label=label,
            return_state=True,
            save_model_to_state=True,
            render_analysis=False,
        )
        assert state.model.problem_type == expected_task_type

        rows = state.model_evaluation.highest_error[:2]
        s = analyze(train_data=df_train, model=state.model, return_state=True, anlz_facets=[ShapAnalysis(rows)])

        assert len(s.explain.shapley) == 2
        for i, (_, row) in enumerate(rows.iterrows()):
            shap_data = s.explain.shapley[i]
            assert_frame_equal(shap_data["row"], pd.DataFrame([row]))
            assert shap_data["feature_names"] is None
            np.array_equal(shap_data["features"], row.values[: len(shap_data["features"])])
            assert sorted(list(shap_data.keys())) == sorted(
                [
                    "row",
                    "expected_value",
                    "shap_values",
                    "features",
                    "feature_names",
                ]
            )

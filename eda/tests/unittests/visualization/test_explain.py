from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest
from pandas import RangeIndex

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import ExplainForcePlot, ExplainWaterfallPlot
from autogluon.eda.visualization.jupyter import JupyterMixin


@pytest.mark.parametrize(
    "display_rows",
    [
        (True),
        (False),
    ],
)
def test_ExplainForcePlot(display_rows, monkeypatch):
    state = AnalysisState()
    state.explain = {
        "shapley": [
            AnalysisState(
                row="row",
                expected_value="expected_value",
                shap_values="shap_values",
                features="features",
                feature_names="feature_names",
            )
        ]
    }
    with monkeypatch.context() as m:
        call_shap_force_plot = MagicMock()
        call_display_obj = MagicMock()
        m.setattr("shap.force_plot", call_shap_force_plot)
        m.setattr(JupyterMixin, "display_obj", call_display_obj)

        ExplainForcePlot(display_rows=display_rows, text_rotation=40, extra_arg="extra_arg").render(state)

        if display_rows:
            call_display_obj.assert_called_with("row")
        else:
            call_display_obj.assert_not_called()

        call_shap_force_plot.assert_called_with(
            "expected_value",
            "shap_values",
            "features",
            feature_names="feature_names",
            text_rotation=40,
            matplotlib=True,
            extra_arg="extra_arg",
        )


@pytest.mark.parametrize(
    "display_rows",
    [
        (True),
        (False),
    ],
)
def test_ExplainWaterfallPlot(display_rows, monkeypatch):
    state = AnalysisState()
    state.explain = {
        "shapley": [
            AnalysisState(
                row="row",
                expected_value="expected_value",
                shap_values="shap_values",
                features=pd.DataFrame({"feature_names": ["features"]}),
                feature_names="feature_names",
            )
        ]
    }
    with monkeypatch.context() as m:
        call_shap_waterfall_plot = MagicMock()
        call_shap_explanation = MagicMock(return_value="explanation_mock")
        call_display_obj = MagicMock()
        m.setattr("shap.waterfall_plot", call_shap_waterfall_plot)
        m.setattr("shap.Explanation", call_shap_explanation)
        m.setattr(JupyterMixin, "display_obj", call_display_obj)

        ExplainWaterfallPlot(display_rows=display_rows, extra_arg="extra_arg").render(state)

        if display_rows:
            call_display_obj.assert_called_with("row")
        else:
            call_display_obj.assert_not_called()

        call_shap_waterfall_plot.assert_called_with("explanation_mock")

        assert call_shap_explanation.call_args.args == ("shap_values",)
        kwargs = call_shap_explanation.call_args.kwargs
        output_names = kwargs.pop("output_names")
        assert output_names.equals(pd.DataFrame({"feature_names": ["features"]}))
        assert kwargs == dict(base_values="expected_value", feature_names=RangeIndex(start=0, stop=1, step=1))

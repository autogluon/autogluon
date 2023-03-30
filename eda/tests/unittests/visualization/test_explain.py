from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

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
        call_display_obj = MagicMock()
        m.setattr("shap.plots.waterfall", call_shap_waterfall_plot)
        m.setattr(JupyterMixin, "display_obj", call_display_obj)

        ExplainWaterfallPlot(display_rows=display_rows, extra_arg="extra_arg").render(state)

        if display_rows:
            call_display_obj.assert_called_with("row")
        else:
            call_display_obj.assert_not_called()

        assert call_shap_waterfall_plot.call_args.args[0].__dict__ == {
            "base_values": "expected_value",
            "display_data": np.array([["features"]], dtype=object),
            "feature_names": [0],
            "values": "shap_values",
        }
        assert call_shap_waterfall_plot.call_args.kwargs == {"extra_arg": "extra_arg"}

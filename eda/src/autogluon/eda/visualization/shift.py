from typing import Optional

from .. import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin

__all__ = ["XShiftSummary"]


class XShiftSummary(AbstractVisualization, JupyterMixin):
    """
    Summarize the results of the XShiftDetector.  It will render the results as markdown in jupyter.
    This will contain the detection status (True if detected), the details of the hypothesis test (test
    statistic, pvalue), and the feature importances for the detection.
    """

    MAX_FEATURES_TO_DISPLAY = 5

    def __init__(self, headers: bool = False, namespace: Optional[str] = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def _summary(self, results: dict) -> str:
        """Output the results of C2ST in a human readable format"""
        if not results["detection_status"]:
            ret_md = "We did not detect a substantial difference between the training and test X distributions."
            return ret_md
        else:
            ret_md = (
                f"We detected a substantial difference between the training and test X distributions,\n"
                f"a type of distribution shift.\n"
                f"\n"
                f"**Test results**: "
                f"We can predict whether a sample is in the test vs. training set with a `{results['eval_metric']}` of\n"
                f"`{results['test_statistic']:.4f}` with a p-value of `{results['pvalue']:.4f}` "
                f"(smaller than the threshold of `{results['pvalue_threshold']:.4f})`.\n"
                f"\n"
            )
        return ret_md

    def _render_feature_importance_if_needed(self, state):
        if "feature_importance" in state and state["detection_status"]:
            fi = state["feature_importance"]
            fi = fi[fi.p_value <= state["pvalue_threshold"]]
            if len(fi) > 0:
                self.render_markdown(
                    "**Feature importances**: The variables that are the most responsible for this shift are those with high feature importance:\n\n"
                )
                self.display_obj(fi[: self.MAX_FEATURES_TO_DISPLAY])

    def can_handle(self, state: AnalysisState) -> bool:
        return self.at_least_one_key_must_be_present(state, "xshift_results")

    def _render(self, state: AnalysisState) -> None:
        header_text = "Detecting distribution shift"
        self.render_header_if_needed(state, header_text)
        self.render_markdown(self._summary(state.xshift_results))
        self._render_feature_importance_if_needed(state.xshift_results)

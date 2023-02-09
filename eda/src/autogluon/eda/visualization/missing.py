import logging
from typing import Optional

import matplotlib.pyplot as plt
import missingno as msno

from .. import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin

__all__ = ["MissingValues"]
logger = logging.getLogger(__name__)


class MissingValues(AbstractVisualization, JupyterMixin):
    """
    Renders visualization of missingness for datasets using one of the methods specified in `graph_type'.

    This visualization depends on :py:class:`~autogluon.eda.analysis.missing.MissingValuesAnalysis` analysis.

    See also `missingno <https://github.com/ResidentMario/missingno>`_ documentation

    Parameters
    ----------
    graph_type: str, default = 'matrix'
        One of the following visualization types:
        - matrix - nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion
            This visualization will comfortably accommodate up to 50 labelled variables.
            Past that range labels begin to overlap or become unreadable, and by default large displays omit them.
        - bar - visualizes how many rows are non-null vs null in the column. Logarithmic scale can by specifying `log=True` in `kwargs`
        - heatmap - correlation heatmap measures nullity correlation: how strongly the presence or absence of one
            variable affects the presence of another. Nullity correlation ranges from -1
            (if one variable appears the other definitely does not) to 0 (variables appearing or not appearing have no effect on one another)
            to 1 (if one variable appears the other definitely also does).
            Entries marked <1 or >-1 have a correlation that is close to being exactingly negative or positive but is still not quite perfectly so.
        - dendrogram - the dendrogram allows to more fully correlate variable completion, revealing trends deeper than the pairwise ones
            visible in the correlation heatmap. The dendrogram uses a hierarchical clustering algorithm (courtesy of scipy) to bin variables
            against one another by their nullity correlation (measured in terms of binary distance).
            At each step of the tree the variables are split up based on which combination minimizes the distance of the remaining clusters.
            The more monotone the set of variables, the closer their total distance is to zero, and the closer their average distance (the y-axis) is to zero.
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.missing.MissingValuesAnalysis`
    """

    __OPERATIONS_MAPPING = {
        "matrix": msno.matrix,
        "bar": msno.bar,
        "heatmap": msno.heatmap,
        "dendrogram": msno.dendrogram,
    }

    MAX_MATRIX_VARIABLES_NUMBER = 50

    def __init__(
        self, graph_type: str = "matrix", headers: bool = False, namespace: Optional[str] = None, **kwargs
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.graph_type = graph_type
        assert (
            self.graph_type in self.__OPERATIONS_MAPPING
        ), f"{self.graph_type} must be one of {self.__OPERATIONS_MAPPING.keys()}"
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        can_handle = self.all_keys_must_be_present(state, "missing_statistics")
        if can_handle and self.graph_type == "matrix" and self._has_too_many_variables_for_matrix(state):
            logging.warning(
                f"The dataset has more than {self.MAX_MATRIX_VARIABLES_NUMBER} variables; "
                f"matrix visualization will comfortably accommodate up to {self.MAX_MATRIX_VARIABLES_NUMBER} labelled variables. "
                f"Past that range labels begin to overlap or become unreadable, and by default large displays omit them."
            )
        return can_handle

    def _render(self, state: AnalysisState) -> None:
        for ds, ds_state in state.missing_statistics.items():
            self.render_header_if_needed(state, f"`{ds}` missing values analysis")
            widget = self._get_operation(self.graph_type)
            self._internal_render(widget, ds_state.data, **self._kwargs)

    def _has_too_many_variables_for_matrix(self, state: AnalysisState):
        for _, ds_state in state.missing_statistics.items():
            if len(ds_state.data.columns) > self.MAX_MATRIX_VARIABLES_NUMBER:
                return True
        return False

    @staticmethod
    def _internal_render(widget, data, **kwargs):
        fig = widget(data, fontsize=10, **kwargs)
        plt.show(fig)

    def _get_operation(self, graph_type):
        return self.__OPERATIONS_MAPPING[graph_type]

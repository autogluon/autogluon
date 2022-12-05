from typing import List, Optional

import phik  # noqa - required for significance_matrix instrumentation on pandas dataframes

from .base import AbstractAnalysis
from .. import AnalysisState

__all__ = ["Correlation", "CorrelationSignificance"]


class Correlation(AbstractAnalysis):
    """
    Correlation analysis.

    Parameters
    ----------
    method: str  {'pearson', 'kendall', 'spearman', 'phik'}, default='spearman'
        Method of correlation:
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * phik : phi_k correlation
                Correlation matrix of bivariate gaussian derived from chi2-value Chi2-value
                gets converted into correlation coefficient of bivariate gauss with correlation
                value rho, assuming given binning and number of records. Correlation coefficient
                value is between 0 and 1. Bivariate gaussian's range is set to [-5,5] by construction.
                See Also `phik <https://github.com/KaveIO/PhiK>`_ documentation.

    focus_field: Optional[str], default = None
        field name to focus. Specifying a field would filter all correlations only when they are >= `focus_field_threshold`
        This is helpful when dealing with a large number of variables.
    focus_field_threshold: float, default = 0.5
        a cut-off threshold when `focus_field` is specified
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: List[AbstractAnalysis], default = []
        wrapped analyses; these will receive sampled `args` during `fit` call

    See Also
    --------
    `phik <https://github.com/KaveIO/PhiK>`_ documentation

    """

    def __init__(
        self,
        method: str = "spearman",
        focus_field: Optional[str] = None,
        focus_field_threshold: float = 0.5,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        assert method in ["pearson", "kendall", "spearman", "phik"]
        self.method = method
        self.focus_field = focus_field
        self.focus_field_threshold = focus_field_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.correlations = {}
        state.correlations_method = self.method
        for (ds, df) in self.available_datasets(args):
            if self.method == "phik":
                state.correlations[ds] = df.phik_matrix(**self.args, verbose=False)
            else:
                state.correlations[ds] = df.corr(method=self.method, **self.args)

            if self.focus_field is not None and self.focus_field in state.correlations[ds].columns:
                state.correlations_focus_field = self.focus_field
                state.correlations_focus_field_threshold = self.focus_field_threshold
                state.correlations_focus_high_corr = {}
                df_corr = state.correlations[ds]
                df_corr = df_corr[df_corr[self.focus_field].abs() >= self.focus_field_threshold]
                keep_cols = df_corr.index.values
                state.correlations[ds] = df_corr[keep_cols]

                high_corr = (
                    state.correlations[ds][[self.focus_field]]
                    .sort_values(self.focus_field, ascending=False)
                    .drop(self.focus_field)
                )
                state.correlations_focus_high_corr[ds] = high_corr


class CorrelationSignificance(AbstractAnalysis):
    """
    Significance of correlation of all variable combinations in the DataFrame.
    See :py:meth:`~phik.significance.significance_matrix` for more details.
    This analysis requires :py:class:`~autogluon.eda.analysis.interaction.Correlation` results to be
    available in the state.

    See Also
    --------
    `phik <https://github.com/KaveIO/PhiK>`_ documentation
    :py:meth:`~phik.significance.significance_matrix`
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "correlations", "correlations_method")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.significance_matrix = {}
        for (ds, df) in self.available_datasets(args):
            state.significance_matrix[ds] = df[state.correlations[ds].columns].significance_matrix(
                **self.args, verbose=False
            )

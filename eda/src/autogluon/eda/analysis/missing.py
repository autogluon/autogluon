from typing import Any, Dict

from .base import AbstractAnalysis, AnalysisState

__all__ = ["MissingValuesAnalysis"]


class MissingValuesAnalysis(AbstractAnalysis):
    """
    Analyze dataset's missing value counts and frequencies

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=...,
    >>>     anlz_facets=[
    >>>         eda.missing.MissingValuesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>         viz.missing.MissingValues()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    :py:class:`~autogluon.eda.visualization.missing.MissingValues`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        s: Dict[str, Any] = {}
        for ds, df in self.available_datasets(args):
            s[ds] = {
                "count": {},
                "ratio": {},
            }
            na = df.isna().sum()
            na = na[na > 0]
            s[ds]["count"] = na.to_dict()
            s[ds]["ratio"] = (na / len(df)).to_dict()
            s[ds]["data"] = df

        state.missing_statistics = s

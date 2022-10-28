from typing import Union, List, Dict, Any

import pandas as pd
from pandas import DataFrame

from .base import AbstractVisualization
from .jupyter import JupyterMixin
from ..state import AnalysisState, StateCheckMixin

__all__ = ['DatasetStatistics', 'DatasetTypeMismatch']


class DatasetStatistics(AbstractVisualization, JupyterMixin):
    """
    Display aggregate dataset statistics and dataset-level information.

    The report is a composite view of combination of performed analyses: :py:class:`~autogluon.eda.analysis.dataset.DatasetSummary`,
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`, :py:class:`~autogluon.eda.analysis.dataset.VariableTypeAnalysis`,
    :py:class:`~autogluon.eda.analysis.dataset.SpecialTypesAnalysis`.
    The components can be present in any combination (assuming their dependencies are satisfied).

    The report requires at lest one of the analyses present to be rendered.

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.DatasetSummary(),
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>         eda.dataset.VariableTypeAnalysis(),
    >>>         eda.dataset.SpecialTypesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.DatasetSummary`
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`
    :py:class:`~autogluon.eda.analysis.dataset.VariableTypeAnalysis`
    :py:class:`~autogluon.eda.analysis.dataset.SpecialTypesAnalysis`
    """

    def __init__(self,
                 headers: bool = False,
                 namespace: str = None,
                 sort_by: Union[None, str] = None,
                 sort_asc: bool = True,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.sort_by = sort_by
        self.sort_asc = sort_asc

    def can_handle(self, state: AnalysisState) -> bool:
        return self.at_least_one_key_must_be_present(state, 'dataset_stats', 'missing_statistics', 'raw_type', 'special_types')

    def _render(self, state: AnalysisState) -> None:
        datasets = []
        for k in ['dataset_stats', 'missing_statistics', 'raw_type', 'variable_type', 'special_types']:
            if k in state:
                datasets = state[k].keys()
                break

        for ds in datasets:
            # Merge different metrics
            stats = self._merge_analysis_facets(ds, state)
            # Fix counts
            df = pd.DataFrame(stats)
            if 'dataset_stats' in state:
                df = self._fix_counts(df, ['unique', 'freq'])
            if 'missing_statistics' in state:
                df = self._fix_counts(df, ['missing_count'])
            df = df.fillna('')

            self.render_header_if_needed(state, f'{ds} dataset summary')
            if self.sort_by in df.columns:
                df = df.sort_values(by=self.sort_by, ascending=self.sort_asc)
            self.display_obj(df)

    @staticmethod
    def _merge_analysis_facets(ds: str, state: AnalysisState):
        stats: Dict[str, Any] = {}
        if 'dataset_stats' in state:
            stats = state.dataset_stats[ds].copy()
        if 'missing_statistics' in state:
            stats = {**stats, **{f'missing_{k}': v for k, v in state.missing_statistics[ds].items() if k in ['count', 'ratio']}}
        if 'raw_type' in state:
            stats['raw_type'] = state.raw_type[ds]
        if 'variable_type' in state:
            stats['variable_type'] = state.variable_type[ds]
        if 'special_types' in state:
            stats['special_types'] = state.special_types[ds]
        return stats

    @staticmethod
    def _fix_counts(df: DataFrame, cols: List[str]) -> DataFrame:
        for k in cols:
            if k in df.columns:
                df[k] = df[k].fillna(-1).astype(int).replace({-1: ''})
        return df


class DatasetTypeMismatch(AbstractVisualization, JupyterMixin, StateCheckMixin):
    """
    Display mismatch between raw types between datasets provided. In case if mismatch found, mark the row with a warning.

    The report requires :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis` analysis present.

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> auto.analyze(
    >>>     train_data=..., test_data=...,
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetTypeMismatch()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`
    """

    def __init__(self, headers: bool = False, namespace: str = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, 'raw_type')

    def _render(self, state: AnalysisState) -> None:
        df = pd.DataFrame(state.raw_type).sort_index()
        warnings = df.eq(df.iloc[:, 0], axis=0)
        df['warnings'] = warnings.all(axis=1).map({True: '', False: 'warning'})
        df.fillna('--', inplace=True)

        self.render_header_if_needed(state, 'Types warnings summary')
        self.display_obj(df)

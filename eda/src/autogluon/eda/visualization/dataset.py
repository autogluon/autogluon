from typing import Any, Dict, Union, List

import ipywidgets as wgts
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
from IPython.display import display
from pandas import DataFrame

from .base import AbstractVisualization
from .jupyter import JupyterMixin
from .. import AnalysisState


class DatasetStatistics(AbstractVisualization, JupyterMixin):

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
        return self.at_least_one_key_must_be_present(state, ['dataset_stats', 'missing_statistics', 'raw_type'])

    def _render(self, state: AnalysisState) -> None:
        datasets = []
        for k in ['dataset_stats', 'missing_statistics', 'raw_type', 'variable_type']:
            if k in state:
                datasets = state[k].keys()

        for ds in datasets:
            # Merge different metrics
            stats = {}
            if 'dataset_stats' in state:
                stats = {**stats, **state.dataset_stats[ds]}
            if 'missing_statistics' in state:
                stats = {**stats, **{f'missing_{k}': v for k, v in state.missing_statistics[ds].items() if k in ['count', 'ratio']}}
            if 'raw_type' in state:
                stats['raw_type'] = state.raw_type[ds]
            if 'variable_type' in state:
                stats['variable_type'] = state.variable_type[ds]
            if 'special_types' in state:
                stats['special_types'] = state.special_types[ds]
            # Fix counts
            df = pd.DataFrame(stats)
            if 'dataset_stats' in state:
                df = self.__fix_counts(df, ['unique', 'freq'])
            if 'missing_statistics' in state:
                df = self.__fix_counts(df, ['missing_count'])

            df = df.fillna('')

            self.render_header_if_needed(state, f'{ds} dataset summary')
            if self.sort_by in df.columns:
                df = df.sort_values(by=self.sort_by, ascending=self.sort_asc)
            self.display_obj(df)

    def __fix_counts(self, df: DataFrame, cols: List[str]) -> DataFrame:
        for k in cols:
            if k in df.columns:
                df[k] = df[k].fillna(-1).astype(int).replace({-1: ''})
        return df


class DatasetTypeMismatch(AbstractVisualization, JupyterMixin):

    def __init__(self, headers: bool = False, namespace: str = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        return 'raw_types' in state

    def _render(self, state: AnalysisState) -> None:
        df = pd.DataFrame(state.raw_type).sort_index()
        warnings = df.eq(df.iloc[:, 0], axis=0)
        df['warnings'] = warnings.all(axis=1).map({True: '', False: '⚠️'})
        df.fillna('--', inplace=True)

        self.render_header_if_needed(state, 'Types warnings summary')
        self.display_obj(df)


class MissingValues(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 headers: bool = False,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return 'missing_statistics' in state

    def _render(self, state: AnalysisState) -> None:
        for ds, data in state.missing_statistics.items():
            self.render_header_if_needed(state, f'{ds} missing values analysis')
            widgets = [msno.matrix, msno.bar, msno.heatmap, msno.dendrogram]
            outs = [wgts.Output() for _ in widgets]
            tab = wgts.Tab(children=outs)
            for i, c in enumerate([w.__name__ for w in widgets]):
                tab.set_title(i, c)
            display(tab)
            for widget, out in zip(widgets, outs):
                with out:
                    ax = widget(data.data, **self._kwargs)
                    plt.show(ax)

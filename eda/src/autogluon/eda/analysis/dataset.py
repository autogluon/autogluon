from __future__ import annotations

from typing import List, Union

import pandas as pd

from autogluon.common.features.infer_types import get_type_group_map_special, get_type_map_raw
from .base import AbstractAnalysis
from .. import AnalysisState


class Sampler(AbstractAnalysis):

    def __init__(self,
                 sample: Union[None, int] = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [], **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.sample = sample

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.sample_size = self.sample
        if self.sample is not None:
            for (ds, df) in self.available_datasets(args):
                self.args[ds] = df.sample(self.sample)


class DatasetSummary(AbstractAnalysis):

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        s = {}
        for (ds, df) in self.available_datasets(args):
            summary = df.describe(include='all').T
            summary = summary.join(pd.DataFrame({'dtypes': df.dtypes}))
            summary['unique'] = args[ds].nunique()
            summary['count'] = summary['count'].astype(int)
            summary = summary.sort_index()
            s[ds] = summary.to_dict()
        state.dataset_stats = s


class RawTypesAnalysis(AbstractAnalysis):

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.raw_types = {}
        for (ds, df) in self.available_datasets(args):
            state.raw_types[ds] = get_type_map_raw(df)


class SpecialTypesAnalysis(AbstractAnalysis):

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.special_types = {}
        for (ds, df) in self.available_datasets(args):
            state.special_types[ds] = self.infer_special_types(df)

    def infer_special_types(self, ds):
        special_types = {}
        for t, cols in get_type_group_map_special(ds).items():
            for col in cols:
                if col not in special_types:
                    special_types[col] = set()
                special_types[col].add(t)
        for col, types in special_types.items():
            special_types[col] = ', '.join(sorted(types))
        return special_types


class MissingValuesAnalysis(AbstractAnalysis):
    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        s = {}
        for (ds, df) in self.available_datasets(args):
            s[ds] = {
                'count': {},
                'ratio': {},
            }
            na = df.isna().sum()
            na = na[na > 0]
            s[ds]['count'] = na.to_dict()
            s[ds]['ratio'] = (na / len(df)).to_dict()
            s[ds]['data'] = df
        state.missing_statistics = s

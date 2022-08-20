from __future__ import annotations

from typing import List, Union

from autogluon.common.features.infer_types import get_type_group_map_special, get_type_map_raw
from .base import AbstractAnalysis
from .. import AnalysisState

DATASET_ARGS = ['train_data', 'test_data', 'tuning_data']


class Sampler(AbstractAnalysis):

    def __init__(self,
                 namespace: str = 'sample',
                 sample: Union[None, int] = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [], **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.namespace = namespace
        self.sample = sample

    def _get_state_from_parent(self) -> AnalysisState:
        state = super()._get_state_from_parent()
        if self.sample is not None:
            state[self.namespace] = {}
            state = state[self.namespace]
        return state

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.sample_size = self.sample
        if self.sample is not None:
            for ds in DATASET_ARGS:
                if ds in args:
                    self.args[ds] = args[ds].sample(self.sample)


class RawTypesAnalysis(AbstractAnalysis):

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.raw_types = {}
        for ds in DATASET_ARGS:
            if ds in args:
                state.raw_types[ds] = get_type_map_raw(args[ds])


class SpecialTypesAnalysis(AbstractAnalysis):

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.special_types = {}
        for ds in DATASET_ARGS:
            if ds in args:
                state.special_types[ds] = self.infer_special_types(args[ds])

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
        s = {
            'count': {},
            'ratio': {},
        }

        for ds in DATASET_ARGS:
            if ds in args:
                na = args[ds].isna().sum()
                na = na[na > 0]
                s['count'][ds] = na.to_dict()
                s['ratio'][ds] = (na / len(args[ds])).to_dict()

        state.missing_statistics = s

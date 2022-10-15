from typing import Union, List

from . import AnalysisState
from .base import AbstractAnalysis

__all__ = ['MissingValuesAnalysis']


class MissingValuesAnalysis(AbstractAnalysis):

    def __init__(self,
                 low_missing_threshold: Union[int, float] = 0.01,
                 high_missing_threshold: Union[int, float] = 0.9,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)

        if isinstance(low_missing_threshold, float):
            assert 0.0 < low_missing_threshold < 1.0, \
                'low_missing_threshold must either be int to represent number of rows for the threshold or be within range (0, 1) to specify cut-off frequency'
        self.low_missing_threshold = low_missing_threshold

        if isinstance(high_missing_threshold, float):
            assert 0.0 < high_missing_threshold < 1.0, \
                'high_missing_threshold must either be int to represent number of rows for the threshold or be within range (0, 1) to specify cut-off frequency'
        self.high_missing_threshold = high_missing_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

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

            low_field = 'count' if isinstance(self.low_missing_threshold, int) else 'ratio'
            high_field = 'count' if isinstance(self.high_missing_threshold, int) else 'ratio'
            s[ds]['low_missing_counts'] = {k: v for k, v in s[ds][low_field].items() if v <= self.low_missing_threshold}
            s[ds]['high_missing_counts'] = {k: v for k, v in s[ds][high_field].items() if
                                            (v >= self.high_missing_threshold) and (k not in s[ds]['low_missing_counts'])}
            s[ds]['mid_missing_counts'] = {k: v for k, v in s[ds][high_field].items() if
                                           (k not in s[ds]['low_missing_counts']) and (k not in s[ds]['high_missing_counts'])}

        state.missing_statistics = s

from __future__ import annotations

import warnings
from typing import List, Union

import pandas as pd
from scipy import stats

from autogluon.common.features.infer_types import get_type_group_map_special, get_type_map_raw
from .base import AbstractAnalysis
from ..state import AnalysisState
from ..util.types import map_raw_type_to_feature_type

__all__ = ['DatasetSummary', 'DistributionFit', 'RawTypesAnalysis', 'Sampler', 'SpecialTypesAnalysis', 'VariableTypeAnalysis']


class Sampler(AbstractAnalysis):

    def __init__(self,
                 sample: Union[None, int] = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [], **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.sample = sample

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.sample_size = self.sample
        if self.sample is not None:
            for (ds, df) in self.available_datasets(args):
                self.args[ds] = df.sample(self.sample, random_state=0)


class DatasetSummary(AbstractAnalysis):

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

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

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.raw_type = {}
        for (ds, df) in self.available_datasets(args):
            state.raw_type[ds] = get_type_map_raw(df)


class VariableTypeAnalysis(AbstractAnalysis):

    def __init__(self,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 numeric_as_categorical_threshold: int = 20,
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, 'raw_type')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.variable_type = {}
        for (ds, df) in self.available_datasets(args):
            state.variable_type[ds] = {c: map_raw_type_to_feature_type(c, t, df, self.numeric_as_categorical_threshold)
                                       for c, t in state.raw_type[ds].items()}


class SpecialTypesAnalysis(AbstractAnalysis):

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

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


class DistributionFit(AbstractAnalysis):

    def __init__(self,
                 columns=Union[str, List[str]],
                 pvalue_min=0.01,
                 keep_top_n: int = 3,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.pvalue_min = pvalue_min
        self.keep_top_n = keep_top_n

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.distributions_fit = {}
        for (ds, df) in self.available_datasets(args):
            state.distributions_fit[ds] = {}
            for c in self.columns:
                if c in df.columns:
                    state.distributions_fit[ds][c] = self._fit_dist(df[c], self.pvalue_min)

    def _fit_dist(self, series, pvalue_min=0.01):
        results = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            list_of_dists = [
                'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull',
                'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'genlogistic', 'genpareto',
                'gennorm', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l',
                'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu',
                'kstwobign', 'laplace', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami',
                'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice',
                'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald',
                'weibull_min', 'weibull_max'
            ]
            for i in list_of_dists:
                dist = getattr(stats, i)
                param = dist.fit(series)
                statistic, pvalue = stats.kstest(series, i, args=param)
                if pvalue >= pvalue_min:
                    results[i] = {
                        'param': param,
                        'statistic': statistic,
                        'pvalue': pvalue,
                    }
            results = pd.DataFrame(results).T.sort_values('pvalue', ascending=False)[:self.keep_top_n].T.to_dict()
            return results

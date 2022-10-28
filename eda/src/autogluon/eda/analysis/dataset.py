from __future__ import annotations

from typing import List, Union, Optional

import pandas as pd

from autogluon.common.features.infer_types import get_type_group_map_special, get_type_map_raw
from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT, R_CATEGORY, R_BOOL
from .base import AbstractAnalysis
from ..state import AnalysisState

__all__ = ['DatasetSummary', 'RawTypesAnalysis', 'Sampler', 'SpecialTypesAnalysis', 'VariableTypeAnalysis']


class Sampler(AbstractAnalysis):
    """
    Sampler is a wrapper that provides sampling capabilites for the wrapped analyses.
    The sampling is performed for all datasets in `args` and passed to all `children` during `fit` call.

    Parameters
    ----------
    sample: Union[None, int, float], default = None
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: List[AbstractAnalysis], default []
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    Examples
    --------
    >>> from autogluon.eda.analysis.base import BaseAnalysis
    >>> from autogluon.eda.analysis import Sampler
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df_train = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list('ABCD'))
    >>> df_test = pd.DataFrame(np.random.randint(0, 100, size=(20, 4)), columns=list('EFGH'))
    >>> analysis = BaseAnalysis(train_data=df_train, test_data=df_test, children=[
    >>>     Sampler(sample=5, children=[
    >>>         # Analysis here will be performed on a sample of 5 for both train_data and test_data
    >>>     ])
    >>> ])
    """

    def __init__(self,
                 sample: Union[None, int, float] = None,
                 parent: Optional[AbstractAnalysis] = None,
                 children: Optional[List[AbstractAnalysis]] = None,
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        if sample is not None and isinstance(sample, float):
            assert 0.0 < sample < 1.0, 'sample must be within the range (0.0, 1.0)'
        self.sample = sample

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        if self.sample is not None:
            state.sample_size = self.sample
            for (ds, df) in self.available_datasets(args):
                arg = 'n'
                if self.sample is not None and isinstance(self.sample, float):
                    arg = 'frac'
                self.args[ds] = df.sample(**{arg: self.sample}, random_state=0)


class DatasetSummary(AbstractAnalysis):
    """
    Generates dataset summary including counts, number of unique elements, most frequent, dtypes and 7-figure summary (std/mean/min/max/quartiles)

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.DatasetSummary(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

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
    """
    Infers autogluon raw types for the column.

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.raw_type = {}
        for (ds, df) in self.available_datasets(args):
            state.raw_type[ds] = get_type_map_raw(df)


class VariableTypeAnalysis(AbstractAnalysis):
    """
    Infers variable types for the column: numeric vs category.

    This analysis depends on :func:`RawTypesAnalysis`.

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>         eda.dataset.VariableTypeAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

    def __init__(self,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: Optional[List[AbstractAnalysis]] = None,
                 numeric_as_categorical_threshold: int = 20,
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, 'raw_type')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.variable_type = {}
        for (ds, df) in self.available_datasets(args):
            state.variable_type[ds] = {c: self.map_raw_type_to_feature_type(c, t, df, self.numeric_as_categorical_threshold)
                                       for c, t in state.raw_type[ds].items()}

    @staticmethod
    def map_raw_type_to_feature_type(col: Optional[str], raw_type: str, df: pd.DataFrame, numeric_as_categorical_threshold: int = 20) -> Union[None, str]:
        if col is None:
            return None
        elif df[col].nunique() <= numeric_as_categorical_threshold:
            return 'category'
        elif raw_type in [R_INT, R_FLOAT]:
            return 'numeric'
        elif raw_type in [R_OBJECT, R_CATEGORY, R_BOOL]:
            return 'category'
        else:
            return None


class SpecialTypesAnalysis(AbstractAnalysis):
    """
    Infers autogluon special types for the column (i.e. text).

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.SpecialTypesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.special_types = {}
        for (ds, df) in self.available_datasets(args):
            state.special_types[ds] = self.infer_special_types(df)

    @staticmethod
    def infer_special_types(ds):
        special_types = {}
        for t, cols in get_type_group_map_special(ds).items():
            for col in cols:
                if col not in special_types:
                    special_types[col] = set()
                special_types[col].add(t)
        for col, types in special_types.items():
            special_types[col] = ', '.join(sorted(types))
        return special_types

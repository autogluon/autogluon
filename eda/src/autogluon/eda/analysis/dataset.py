from __future__ import annotations

from typing import List, Union, Optional

from .base import AbstractAnalysis
from ..state import AnalysisState

__all__ = ['Sampler']


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

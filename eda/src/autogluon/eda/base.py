from __future__ import annotations

from typing import List, Union

from autogluon.eda.analysis.univariate import UnivariateAnalysis


class Analysis:
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __init__(self, **kwargs) -> None:
        self.facets = []
        self.ctx = {**kwargs}

    def with_columns(self, columns: Union[str, List[str]] = '__all__') -> Analysis:
        if columns:
            self.ctx['columns'] = columns
        else:
            self.ctx.pop('columns')
        return self

    @property
    def univariate(self) -> UnivariateAnalysis:
        analysis = UnivariateAnalysis(self.ctx, parent=self)
        return analysis

    def __str__(self) -> str:
        return '\n\t'.join([str(facet) for facet in self.facets])

    def fit(self):
        for f in self.facets:
            f.fit()

    def render(self, engine='vega', **kwargs):
        for f in self.facets:
            f.render(engine, **kwargs)

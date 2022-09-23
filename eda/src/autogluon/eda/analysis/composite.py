from __future__ import annotations

from typing import List, Union

from .base import AbstractAnalysis
from .. import AnalysisState


class BaseAnalysis(AbstractAnalysis):

    def __init__(self,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        pass

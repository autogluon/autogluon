from typing import Union, List, Any, Optional, Callable
from PyOD.models import hbos, iforest
from .. import AnalysisState
from .base import AbstractAnalysis


class AnomalyDetector(AbstractAnalysis):
    """An anomaly detector for tabular data.  It provides anomaly scores for each row in the test dataframe. The
    default methods are isolation forests (high quality) and histogram based outlier score (medium quality - faster
    computation).
    """

    def __init__(self,
                 preset: str='high_quality',
                 OD_method: Callable=None,
                 OD_kwargs: dict={},
                 parent: Union[None,AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        preset_list = ['high_quality', 'medium_quality']
        assert preset in preset_list, 'preset must be one of ' + ', '.join(preset_list)
        self.preset = preset
        self.OD_method = OD_method
        self.OD_kwargs = OD_kwargs

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        if self.OD_method is None:
            if self.preset == 'high_quality':
                self.OD_method = iforest
            if self.preset == 'medium_quality':
                self.OD_method = hbos
        clf = self.OD_method(**OD_kwargs)
        feature_generator = AutoMLPipelineFeatureGenerator()
        train_trans = feature_generator.fit_transform(X=train_cs)
        test_trans = feature_generator.transform(test_cs)
        clf.fit(train_trans, **fit_kwargs)
        state.test_ano_scores = clf.decision_function(test_trans)
        state.test_ano_pred = clf.predict(test_trans)

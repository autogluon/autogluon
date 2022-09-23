from typing import Union, List, Callable
from pyod.models import hbos, iforest
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from .. import AnalysisState
from .base import AbstractAnalysis


class AnomalyDetector(AbstractAnalysis):
    """An anomaly detector for tabular data.  It provides anomaly scores for each row in the test dataframe. The
    default methods are isolation forests (high quality) and histogram based outlier score (medium quality - faster
    computation).

    Parameters
    ----------
    preset: str, default = 'high_quality'
        'high_quality' for more powerful but computationally expensive detector, 'medium_quality' otherwise
    OD_method: Callable, default = None
        Custom anomaly detector from pyod.models, if you don't want the defaults - will override preset
    OD_kwargs: dict, default = {}
        kwargs for the OD method

    State attributes
    ---------------
    state.test_ano_scores: 1D array
        The anomaly scores from the pyod anomaly detector
    state.test_ano_pred: 1D array
        The anomaly predictions (1 = anomaly) from the pyod anomaly detector
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
                self.OD_method = iforest.IForest
            if self.preset == 'medium_quality':
                self.OD_method = hbos.HBOS
        clf = self.OD_method(**self.OD_kwargs)
        feature_generator = AutoMLPipelineFeatureGenerator()
        if args['label'] is not None:
            if args['label'] in args['train_data'].columns:
                X_train = args['train_data'].drop(columns=[args['label']])
            if args['label'] in args['test_data'].columns:
                X_test = args['test_data'].drop(columns=[args['label']])
        train_trans = feature_generator.fit_transform(X=X_train)
        test_trans = feature_generator.transform(X_test)
        clf.fit(train_trans, **fit_kwargs)
        state.test_ano_scores = clf.decision_function(test_trans)
        state.test_ano_pred = clf.predict(test_trans)

from typing import Union, List, Callable

import pandas as pd
import shap
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
    fit_train: bool, default = False
        True to find anomalies in the training set
    OD_method: Callable, default = None
        Custom anomaly detector from pyod.models, if you don't want the defaults - will override preset
    OD_kwargs: dict, default = {}
        kwargs for the OD method
    shap_sub_samp: float or int, default = 0.1
        The amount of subsampling for shap permutation
    num_anomalies: int, default = 5
        The number of top anomalies when returning SHAP features

    State attributes
    ---------------
    state.top_test_anomalies: pd.Series
        The test set anomalies with SHAP results from the pyod anomaly detector
    state.top_train_anomalies: pd.Series
        The test set anomalies with SHAP results from the pyod anomaly detector, if fit_train==True
    """

    def __init__(self,
                 preset: str='high_quality',
                 fit_train: bool=False,
                 OD_method: Callable=None,
                 OD_kwargs: dict={},
                 shap_sub_samp: int=40, #fix this!
                 num_anomalies: int=5,
                 parent: Union[None,AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        preset_list = ['high_quality', 'medium_quality']
        assert preset in preset_list, 'preset must be one of ' + ', '.join(preset_list)
        self.preset = preset
        self.OD_method = OD_method
        self.OD_kwargs = OD_kwargs
        self.shap_sub_samp = shap_sub_samp
        self.num_anomalies = num_anomalies
        self.fit_train = fit_train

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        if self.OD_method is None:
            if self.preset == 'high_quality':
                self.OD_method = iforest.IForest
            if self.preset == 'medium_quality':
                self.OD_method = hbos.HBOS
        X_train = args['train_data'].copy()
        if args['label'] is not None:
            if args['label'] in args['train_data'].columns:
                X_train = X_train.drop(columns=[args['label']])
        feature_generator = AutoMLPipelineFeatureGenerator()
        train_trans = feature_generator.fit_transform(X=X_train)
        if self.fit_train:
            state.top_train_anomalies = self._fit_train(train_trans, **fit_kwargs)
        if 'test_data' in args:
            X_test = args['test_data'].copy()
            if args['label'] is not None:
                if args['label'] in args['test_data'].columns:
                    X_test = X_test.drop(columns=[args['label']])
            test_trans = feature_generator.transform(X_test)
            state.top_test_anomalies = self._fit_test(train_trans, test_trans, **fit_kwargs)

    def _fit_train(self, train_trans, **fit_kwargs):
        X_tr_1 = train_trans.sample(frac = 0.5)
        X_tr_2 = train_trans.drop(X_tr_1.index)
        ano_model_1 = self.OD_method(**self.OD_kwargs).fit(X_tr_1, **fit_kwargs)
        ano_model_2 = self.OD_method(**self.OD_kwargs).fit(X_tr_2, **fit_kwargs)
        scores_1 = ano_model_2.decision_function(X_tr_1.values)
        scores_2 = ano_model_1.decision_function(X_tr_2.values)
        scores = [(score, 1, i) for i, score in enumerate(scores_1)]
        scores += [(score, 2, i) for i, score in enumerate(scores_2)]
        scores.sort(key=lambda x: x[0], reverse=True)
        scores_top = scores[:self.num_anomalies]
        top_ano_ids1 = [id for _, s, id in scores_top if s == 1]
        top_ano_ids2 = [id for _, s, id in scores_top if s == 2]
        scores_top_1 = [sco for sco, s, id in scores_top if s == 1]
        scores_top_2 = [sco for sco, s, id in scores_top if s == 2]
        shap_vals = []
        if len(top_ano_ids1) > 0:
            shap_vals += [(a,b,c) for a, (b,c) in zip(scores_top_1,
                    self._compute_anomaly_shap(ano_model_2, top_ano_ids1, X_tr_1))]
        if len(top_ano_ids2) > 0:
            shap_vals += [(a,b,c) for a, (b,c) in zip(scores_top_2,
                self._compute_anomaly_shap(ano_model_1, top_ano_ids2, X_tr_2))]
        return shap_vals

    def _fit_test(self, train_trans, test_trans, **fit_kwargs):
        clf = self.OD_method(**self.OD_kwargs)
        clf.fit(train_trans, **fit_kwargs)
        scores = clf.decision_function(test_trans.values)
        top_score_ids = scores.argsort()[:-self.num_anomalies-1:-1]
        scores_top = scores[top_score_ids]
        return [(a,b,c) for a, (b,c) in zip(scores_top,
                    self._compute_anomaly_shap(clf, top_score_ids, test_trans))]

    def _compute_anomaly_shap(self, clf, top_score_ids, test_trans):
        test_sampler = shap.utils.sample(test_trans.values, self.shap_sub_samp)
        explainer = shap.Explainer(clf.decision_function, test_sampler)
        shap_values = explainer(test_trans.values[top_score_ids, :])
        top_anomalies = [(test_trans.iloc[aid, :], sv) for aid, sv in zip(top_score_ids, shap_values)]
        return top_anomalies

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return 'train_data' in args

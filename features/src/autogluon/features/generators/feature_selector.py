import logging
from typing import Tuple
import numpy as np
from pandas import DataFrame, Series
from .abstract import AbstractFeatureGenerator
from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models import KNNModel, RFModel, CatBoostModel, LGBModel

logger = logging.getLogger(__name__)


class ProxyModelFeatureSelector(AbstractFeatureGenerator):
    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        if model == 'LGB':
            self.model_type = LGBModel
        elif model == 'RF':
            self.model_type = RFModel
        elif model == 'KNN':
            self.model_type = KNNModel
        else:
            self.model_type = CatBoostModel
        self.model = None
        self.kept_features = []
        self.pruned_features = []
        self.num_train_samples = kwargs.get('num_train_samples', 50000)
        self.n_repeats = kwargs.get('n_repeats', 1)
        self.seed = kwargs.get('random_seed', 0)
        # fit_with_prune kwargs
        self.max_num_fit = kwargs.get('max_num_fit', 5)
        self.stop_threshold = kwargs.get('stop_threshold', 3)
        self.prune_ratio = kwargs.get('prune_ratio', 0.1)
        self.num_resource = kwargs.get('num_resource', None)
        self.fi_strategy = kwargs.get('fi_strategy', 'uniform')
        self.fp_strategy = kwargs.get('fp_strategy', 'percentage')
        self.subsample_size = kwargs.get('subsample_size', 100)
        self.prune_threshold = kwargs.get('prune_threshold', None)

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> Tuple[DataFrame, dict]:
        """
        Call self.model.fit_with_prune() then call best_model.get_features().
        Try running powersetting with lots of resources since we call this just
        once.
        """
        features = self.feature_metadata_in.get_features()
        base_model = self.model_type(name='proxy_base_model', problem_type=kwargs.get('problem_type', None), eval_metric=kwargs.get('eval_metric', None))
        self.model = BaggedEnsembleModel(base_model, name='proxy_model', random_state=0)
        self.num_train_samples = min(len(X), self.num_train_samples)
        indexes = np.random.default_rng(self.seed).choice(len(X), replace=False, size=self.num_train_samples)
        X_train, y_train = X.iloc[indexes], y.iloc[indexes]
        best_model, _ = self.model.fit_with_prune(X=X_train, y=y_train, X_val=None, y_val=None, max_num_fit=self.max_num_fit,
                                                  stop_threshold=self.stop_threshold, prune_ratio=self.prune_ratio, num_resource=self.num_resource,
                                                  fi_strategy=self.fi_strategy, fp_strategy=self.fp_strategy, subsample_size=self.subsample_size,
                                                  prune_threshold=self.prune_threshold, k_fold=3)
        self.kept_features = best_model.get_features()
        self.pruned_features = [feature for feature in features if feature not in self.kept_features]
        return self._transform(X), self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return X[self.kept_features]

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

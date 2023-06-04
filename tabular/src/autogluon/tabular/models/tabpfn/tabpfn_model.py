import numpy as np

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator


# TODO: Max classes = 10
# TODO: Max rows = 1000
# TODO: Max features = 100
class TabPFNModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X)
        hyp = self._get_model_params()
        N_ensemble_configurations = hyp.get('N_ensemble_configurations', 1)
        from tabpfn import TabPFNClassifier
        self.model = TabPFNClassifier(device='cpu', N_ensemble_configurations=N_ensemble_configurations).fit(X, y, overwrite_warning=True)

    # TODO: Should we fillna 0? what about -1?
    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _more_tags(self):
        tags = {'can_refit_full': True}
        return tags

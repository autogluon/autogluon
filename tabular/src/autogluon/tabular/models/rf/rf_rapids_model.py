import logging

from autogluon.core.constants import REGRESSION, SOFTCLASS
from autogluon.core.utils.try_import import try_import_rapids_cuml

from .rf_model import RFModel

logger = logging.getLogger(__name__)


# TODO: Improve memory safety
# TODO: Respect time limit
# TODO: Significantly less accurate than RFModel with same hyperparameters.
#  Refer to https://github.com/rapidsai/cuml/issues/2518
class RFRapidsModel(RFModel):
    """
    RAPIDS Random Forest model : https://rapids.ai/start.html

    NOTE: This code is experimental, it is recommend to not use this unless you are a developer.
    This was tested on rapids-0.18 via:

    conda create -n rapids-0.18 -c rapidsai -c nvidia -c conda-forge -c defaults rapids-blazing=0.18 python=3.7 cudatoolkit=10.1 -y
    conda activate rapids-0.18
    pip install --pre autogluon.tabular[all]
    """
    def _get_model_type(self):
        try_import_rapids_cuml()
        from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
        if self.problem_type in [REGRESSION, SOFTCLASS]:
            return RandomForestRegressor
        else:
            return RandomForestClassifier

    def _set_default_params(self):
        default_params = {
            'n_estimators': 300,
            'max_depth': 99,  # RAPIDS does not allow unlimited depth, so this approximates it.
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _fit(self, X, y, **kwargs):
        logger.warning('\tWarning: Training RAPIDS RandomForest model... There is a known bug that lowers model quality compared to sklearn RandomForest. '
                       'Consider using CPU instead if model quality is not sufficient.\n'
                       '\t\tLink to issue: https://github.com/rapidsai/cuml/issues/2518')
        X = self.preprocess(X)
        self.model = self._get_model_type()(**self._get_model_params())
        self.model = self.model.fit(X, y)
        self.params_trained['n_estimators'] = self.model.n_estimators

    # FIXME: Efficient OOF doesn't work in RAPIDS
    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {'use_child_oof': False}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self):
        return {'valid_oof': False}

import logging
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from ..rf.rf_quantile import ExtraTreesQuantileRegressor
from ..rf.rf_model import RFModel
from autogluon.core.constants import REGRESSION, QUANTILE

logger = logging.getLogger(__name__)


class XTModel(RFModel):
    """
    Extra Trees model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    """
    def _get_model_type(self):
        if self.problem_type == REGRESSION:
            return ExtraTreesRegressor
        elif self.problem_type == QUANTILE:
            logger.warning('\tWarning: sklearn forest models are experimental for quantile regression. '
                           'They may change or be removed without warning in future releases.')
            return ExtraTreesQuantileRegressor
        else:
            return ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from ..rf.rf_model import RFModel
from autogluon.core.constants import REGRESSION


class XTModel(RFModel):
    """
    Extra Trees model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    """
    def _get_model_type(self):
        if self.problem_type == REGRESSION:
            return ExtraTreesRegressor
        else:
            return ExtraTreesClassifier

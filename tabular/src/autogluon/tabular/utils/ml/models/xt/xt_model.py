from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from ..rf.rf_model import RFModel
from ...constants import REGRESSION


class XTModel(RFModel):
    def _get_model_type(self):
        if self.problem_type == REGRESSION:
            return ExtraTreesRegressor
        else:
            return ExtraTreesClassifier

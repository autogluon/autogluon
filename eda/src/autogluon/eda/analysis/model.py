from typing import Union, List

from sklearn.metrics import confusion_matrix

from autogluon.core.constants import REGRESSION
from autogluon.eda import AnalysisState
from autogluon.eda.analysis import AbstractAnalysis
from autogluon.tabular import TabularPredictor


class AutoGluonModelEvaluator(AbstractAnalysis):

    def __init__(self,
                 normalize: Union[None, str] = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.normalize = normalize

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        if 'model' in args and 'val_data' in args:
            predictor: TabularPredictor = args.model
            val_data = args.val_data
            problem_type = predictor.problem_type
            label = predictor.label
            y_true = val_data[label]
            y_pred = predictor.predict(val_data)
            importance = predictor.feature_importance(val_data.reset_index(drop=True))

            s = {
                'problem_type': predictor.problem_type,
                'y_true': y_true,
                'y_pred': y_pred,
                'importance': importance,
            }
            if problem_type == REGRESSION:
                pass  # TODO: add scatterplot for regression
            else:
                cm = confusion_matrix(y_true, y_pred, normalize=self.normalize, labels=y_true.unique())
                s['confusion_matrix_normalized'] = self.normalize is not None
                s['confusion_matrix'] = cm

            state.model_evaluation = s

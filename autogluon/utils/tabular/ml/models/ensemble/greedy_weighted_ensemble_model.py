import logging
from ..abstract.abstract_model import AbstractModel
from ....metrics import accuracy
from ...constants import BINARY, MULTICLASS, REGRESSION
from ...tuning.ensemble_selection import EnsembleSelection

logger = logging.getLogger(__name__)


class GreedyWeightedEnsembleModel(AbstractModel):
    def __init__(self, path: str, name: str, problem_type: str, objective_func, num_classes, base_model_names, model_base=EnsembleSelection, hyperparameters=None, features=None, feature_types_metadata=None, debug=0):
        super().__init__(path, name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features, feature_types_metadata=feature_types_metadata, debug=debug)
        self.model_base = model_base
        self.num_classes = num_classes
        self.base_model_names = base_model_names
        self.features, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)

    def _get_default_searchspace(self, problem_type):
        spaces = {}

        return spaces

    def _set_default_params(self):
        default_params = {'ensemble_size': 100}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def preprocess(self, X):
        X = self.convert_pred_probas_df_to_list(X)
        return X

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, **kwargs):
        X_train = self.preprocess(X_train)

        self.model = self.model_base(ensemble_size=100, problem_type=self.problem_type, metric=self.objective_func)
        self.model = self.model.fit(X_train, Y_train)
        self.base_model_names, self.model.weights_ = self.remove_zero_weight_models(self.base_model_names, self.model.weights_)
        self.features, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)

    def convert_pred_probas_df_to_list(self, pred_probas_df) -> list:
        pred_probas = []
        for i, model in enumerate(self.base_model_names):
            index_start = i * self.num_pred_cols_per_model
            index_end = (i + 1) * self.num_pred_cols_per_model
            model_cols = self.features[index_start:index_end]
            pred_proba = pred_probas_df[model_cols].values
            if self.num_pred_cols_per_model == 1:
                pred_proba = pred_proba.flatten()
            pred_probas.append(pred_proba)
        return pred_probas

    @staticmethod
    def remove_zero_weight_models(base_model_names, base_model_weights):
        base_models_to_keep = []
        base_model_weights_to_keep = []
        for i, weight in enumerate(base_model_weights):
            if weight != 0:
                base_models_to_keep.append(base_model_names[i])
                base_model_weights_to_keep.append(weight)
        return base_models_to_keep, base_model_weights_to_keep

    def set_stack_columns(self, base_model_names):
        if self.problem_type == MULTICLASS:
            stack_columns = [model_name + '_' + str(cls) for model_name in base_model_names for cls in range(self.num_classes)]
            num_pred_cols_per_model = self.num_classes
        else:
            stack_columns = base_model_names
            num_pred_cols_per_model = 1
        return stack_columns, num_pred_cols_per_model

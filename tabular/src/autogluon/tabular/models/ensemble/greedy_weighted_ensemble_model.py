import logging

from ..abstract.abstract_model import AbstractModel
from ...constants import MULTICLASS
from ...tuning.ensemble_selection import EnsembleSelection

logger = logging.getLogger(__name__)


class GreedyWeightedEnsembleModel(AbstractModel):
    def __init__(self, base_model_names, model_base=EnsembleSelection, **kwargs):
        super().__init__(**kwargs)
        self.model_base = model_base
        self.base_model_names = base_model_names
        self.weights_ = None
        self.features, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)

    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def _set_default_params(self):
        default_params = {'ensemble_size': 100}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Consider moving convert_pred_probas_df_to_list into inner model to ensure X remains a dataframe after preprocess is called
    def _preprocess_nonadaptive(self, X, **kwargs):
        # TODO: super() call?
        X = self.convert_pred_probas_df_to_list(X)
        return X

    # TODO: Check memory after loading best model predictions, only load top X model predictions that fit in memory
    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, **kwargs):
        X_train = self.preprocess(X_train)

        self.model = self.model_base(ensemble_size=self.params['ensemble_size'], problem_type=self.problem_type, metric=self.stopping_metric)
        self.model = self.model.fit(X_train, y_train, time_limit=time_limit)
        self.base_model_names, self.model.weights_ = self.remove_zero_weight_models(self.base_model_names, self.model.weights_)
        self.weights_ = self.model.weights_
        self.features, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)
        self.params_trained['ensemble_size'] = self.model.ensemble_size

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

    def _get_model_weights(self):
        num_models = len(self.base_model_names)
        model_weight_dict = {self.base_model_names[i]: self.weights_[i] for i in range(num_models)}
        return model_weight_dict

    def get_info(self):
        info = super().get_info()
        info['model_weights'] = self._get_model_weights()
        return info

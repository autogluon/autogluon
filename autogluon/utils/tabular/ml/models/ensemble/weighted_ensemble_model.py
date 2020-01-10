import logging

from .stacker_ensemble_model import StackerEnsembleModel
from .greedy_weighted_ensemble_model import GreedyWeightedEnsembleModel

logger = logging.getLogger(__name__)


# TODO: Optimize predict speed when fit on kfold, can simply sum weights
class WeightedEnsembleModel(StackerEnsembleModel):
    def __init__(self, path, name, base_model_names, base_model_paths_dict, base_model_types_dict, base_model_types_inner_dict=None, base_model_performances_dict=None, num_classes=None, hyperparameters=None, random_state=0, debug=0):
        model_0 = base_model_types_dict[base_model_names[0]].load(path=base_model_paths_dict[base_model_names[0]], verbose=False)
        super().__init__(path=path, name=name, model_base=model_0, base_model_names=base_model_names, base_model_paths_dict=base_model_paths_dict, base_model_types_dict=base_model_types_dict, base_model_types_inner_dict=base_model_types_inner_dict, base_model_performances_dict=base_model_performances_dict, use_orig_features=False, num_classes=num_classes, hyperparameters=hyperparameters, random_state=random_state, debug=debug)
        self.model_base = GreedyWeightedEnsembleModel(path='', name='greedy_ensemble', num_classes=self.num_classes, base_model_names=self.base_model_names, problem_type=self.problem_type, objective_func=self.objective_func)
        self.low_memory = False

    def fit(self, X, y, k_fold=5, n_repeats=1, n_repeat_start=0, compute_base_preds=True, time_limit=None, **kwargs):
        super().fit(X, y, k_fold=k_fold, n_repeats=n_repeats, n_repeat_start=n_repeat_start, compute_base_preds=compute_base_preds, time_limit=time_limit, **kwargs)
        base_model_names = []
        for model in self.models:
            model = self.load_child(model, verbose=False)
            base_model_names = base_model_names + [base_model_name for base_model_name in model.base_model_names if base_model_name not in base_model_names]
        self.base_model_names = [base_model_name for base_model_name in self.base_model_names if base_model_name in base_model_names]
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)

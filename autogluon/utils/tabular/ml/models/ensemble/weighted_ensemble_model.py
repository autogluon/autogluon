import logging

from .stacker_ensemble_model import StackerEnsembleModel
from .greedy_weighted_ensemble_model import GreedyWeightedEnsembleModel

logger = logging.getLogger(__name__)


# TODO: Optimize predict speed when fit on kfold, can simply sum weights
class WeightedEnsembleModel(StackerEnsembleModel):
    def __init__(self, path, name, base_model_names, base_model_paths_dict, base_model_types_dict, base_model_performances_dict=None, num_classes=None, hyperparameters=None, debug=0):
        model_0 = base_model_types_dict[base_model_names[0]].load(path=base_model_paths_dict[base_model_names[0]], verbose=False)
        super().__init__(path=path, name=name, model_base=model_0, base_model_names=base_model_names, base_model_paths_dict=base_model_paths_dict, base_model_types_dict=base_model_types_dict, base_model_performances_dict=base_model_performances_dict, use_orig_features=False, num_classes=num_classes, hyperparameters=hyperparameters, debug=debug)
        self.model_base = GreedyWeightedEnsembleModel(path='', name='greedy_ensemble', num_classes=self.num_classes, base_model_names=self.base_model_names, problem_type=self.problem_type, objective_func=self.objective_func)

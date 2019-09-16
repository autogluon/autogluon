
from tabular.ml.trainer.abstract_trainer import AbstractTrainer
from tabular.ml.models.model_presets.presets import get_preset_models


# Trainer handles model training details
class AutoTrainer(AbstractTrainer):
    def __init__(self, path, problem_type, objective_func=None, num_classes=None, low_memory=False, feature_types_metadata={}, searcher=None, scheduler=None):
        super().__init__(path=path, problem_type=problem_type, objective_func=objective_func, num_classes=num_classes, low_memory=low_memory, 
                         feature_types_metadata=feature_types_metadata, searcher=searcher, scheduler=scheduler)
        self.num_boost_round = 100000

    def train(self, X_train, X_test, y_train, y_test, hyperparameter_tune=False, feature_prune=False):
        models = get_preset_models(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, num_boost_round=self.num_boost_round, num_classes=self.num_classes)
        self.train_multi_and_ensemble(X_train, X_test, y_train, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)

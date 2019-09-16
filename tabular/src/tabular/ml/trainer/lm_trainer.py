
from tabular.ml.trainer.abstract_trainer import AbstractTrainer
from tabular.ml.models.model_presets.presets import get_preset_models


# Trainer handles language model training details
class LMTrainer(AbstractTrainer):
    def __init__(self, path, problem_type, objective_func=None, num_classes=None, low_memory=False, feature_types_metadata={}):
        super().__init__(path=path, problem_type=problem_type, objective_func=objective_func, num_classes=num_classes, low_memory=low_memory, feature_types_metadata=feature_types_metadata)

    def train_single(self, X_train, X_test, y_train, y_test, model):
        print('fitting', model.name, '...')
        model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
        print('LM Training finished')

    def train(self, X_train, X_test, y_train, y_test):
        models = get_preset_models(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func)
        for i, model in enumerate(models):
            print(f'Training model {i+1}/{len(models)}...')
            self.train_single(X_train, X_test, y_train, y_test, model)
            print(f'Training model {i+1}/{len(models)} - COMPLETED')

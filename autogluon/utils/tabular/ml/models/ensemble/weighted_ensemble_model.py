import numpy as np

from ..abstract.abstract_model import AbstractModel


class WeightedEnsembleModel(AbstractModel):
    def __init__(self, path, name, base_model_names, base_model_paths_dict, base_model_types_dict, base_model_weights, debug=0):
        self.base_model_names = base_model_names
        self.base_model_paths_dict = base_model_paths_dict  # TODO: Make compatible with movement of files
        self.base_model_types_dict = base_model_types_dict
        self.base_model_weights = base_model_weights
        self.oof_pred_proba = None  # TODO: Remove this? Move it internally into trainer

        model_0 = self.load_model(self.base_model_names[0])
        problem_type = model_0.problem_type
        objective_func = model_0.objective_func

        super().__init__(path=path, name=name, model=None, problem_type=problem_type, objective_func=objective_func, debug=debug)

    # TODO: Currently, if this is a weighted ensemble of stackers, it will be very slow due to each stacker needing to repeat computation on the base models.
    #  To solve this, this model must know full context of stacker, and only get preds once for each required model
    #  This is already done in trainer, but could be moved internally.
    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)

        model_index_to_ignore = []
        for index, weight in enumerate(self.base_model_weights):
            if weight == 0:
                model_index_to_ignore.append(index)
        models_to_predict_on = [model for index, model in enumerate(self.base_model_names) if index not in model_index_to_ignore]
        models_to_predict_on_weights = [weight for index, weight in enumerate(self.base_model_weights) if index not in model_index_to_ignore]

        pred_probas = []
        for model in models_to_predict_on:
            if type(model) is str:
                model = self.load_model(model)
            model_pred = model.predict_proba(X)
            pred_probas.append(model_pred)
        return self.weight_pred_probas(pred_probas, weights=models_to_predict_on_weights)

    @staticmethod
    def weight_pred_probas(pred_probas, weights):
        preds_norm = [pred * weight for pred, weight in zip(pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        return preds_ensemble

    # TODO: Make compatible with movement of files
    def load_model(self, model_name: str, reset_paths=False):
        return self.base_model_types_dict[model_name].load(path=self.base_model_paths_dict[model_name], reset_paths=reset_paths)

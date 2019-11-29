import numpy as np

from .stacker_ensemble_model import StackerEnsembleModel


class WeightedEnsembleModel(StackerEnsembleModel):
    def __init__(self, path, name, base_model_names, base_model_paths_dict, base_model_types_dict, base_model_weights, debug=0):
        self.base_model_weights = base_model_weights

        base_models_to_keep = []
        base_model_weights_to_keep = []
        # TODO: MOVE TO FIT!
        for i, weight in enumerate(self.base_model_weights):
            if weight != 0:
                base_models_to_keep.append(base_model_names[i])
                base_model_weights_to_keep.append(weight)

        self.base_model_weights = base_model_weights_to_keep

        model_0 = base_model_types_dict[base_model_names[0]].load(path=base_model_paths_dict[base_model_names[0]], verbose=False)
        super().__init__(path=path, name=name, model_base=model_0, base_model_names=base_models_to_keep, base_model_paths_dict=base_model_paths_dict, base_model_types_dict=base_model_types_dict, use_orig_features=False, debug=debug)
        self.model_base = None

    # TODO: Currently, if this is a weighted ensemble of stackers, it will be very slow due to each stacker needing to repeat computation on the base models.
    #  To solve this, this model must know full context of stacker, and only get preds once for each required model
    #  This is already done in trainer, but could be moved internally.
    def predict_proba(self, X, preprocess=True):
        # TODO: FILTER BASE_MODEL_NAMES / COLUMNS BASED ON WEIGHTS, if 0 WEIGHT THEN DON'T LOAD! DO THIS IN FIT ONCE CORE LOGIC IS MOVED INSIDE
        if preprocess:
            X = self.preprocess(X, preprocess=False, model=None)
        model_index_to_ignore = []
        models_to_predict_on_weights = [weight for index, weight in enumerate(self.base_model_weights) if index not in model_index_to_ignore]

        pred_probas = []
        for i, model in enumerate(self.base_model_names):
            index_start = i * self.num_pred_cols_per_model
            index_end = (i+1) * self.num_pred_cols_per_model
            model_cols = self.stack_columns[index_start:index_end]
            pred_probas.append(X[model_cols].values)

        return self.weight_pred_probas(pred_probas, weights=models_to_predict_on_weights)

    @staticmethod
    def weight_pred_probas(pred_probas, weights):
        preds_norm = [pred * weight for pred, weight in zip(pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        return preds_ensemble

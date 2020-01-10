import copy, logging, time
import numpy as np
import pandas as pd
from collections import defaultdict

from ...utils import generate_kfold
from ..abstract.abstract_model import AbstractModel
from .bagged_ensemble_model import BaggedEnsembleModel
from ...constants import MULTICLASS, REGRESSION

logger = logging.getLogger(__name__)


# TODO: Currently, if this is a stacker above level 1, it will be very slow taking raw input due to each stacker needing to repeat computation on the base models.
#  To solve this, this model must know full context of stacker, and only get preds once for each required model
#  This is already done in trainer, but could be moved internally.
class StackerEnsembleModel(BaggedEnsembleModel):
    def __init__(self, path, name, model_base: AbstractModel, base_model_names=None, base_model_paths_dict=None, base_model_types_dict=None, base_model_types_inner_dict=None, base_model_performances_dict=None, use_orig_features=True, num_classes=None, hyperparameters=None, random_state=0, debug=0):
        super().__init__(path=path, name=name, model_base=model_base, hyperparameters=hyperparameters, random_state=random_state, debug=debug)
        if base_model_names is None:
            base_model_names = []
        if base_model_paths_dict is None:
            base_model_paths_dict = {}
        if base_model_types_dict is None:
            base_model_types_dict = {}
        self.base_model_names = base_model_names
        self.base_model_paths_dict = base_model_paths_dict
        self.base_model_types_dict = base_model_types_dict
        self.bagged_mode = None
        self.use_orig_features = use_orig_features
        self.num_classes = num_classes

        if (base_model_performances_dict is not None) and (base_model_types_inner_dict is not None):
            if self.params['max_models_per_type'] > 0:
                self.base_model_names = self.limit_models_per_type(models=self.base_model_names, model_types=base_model_types_inner_dict, model_scores=base_model_performances_dict, max_models_per_type=self.params['max_models_per_type'])
            if self.params['max_models'] > 0:
                self.base_model_names = self.limit_models(models=self.base_model_names, model_scores=base_model_performances_dict, max_models=self.params['max_models'])
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)

    @staticmethod
    def limit_models_per_type(models, model_types, model_scores, max_models_per_type):
        model_type_groups = defaultdict(list)
        for model in models:
            model_type = model_types[model]
            model_type_groups[model_type].append((model, model_scores[model]))
        for key in model_type_groups:
            model_type_groups[key] = sorted(model_type_groups[key], key=lambda x: x[1], reverse=True)
        for key in model_type_groups:
            model_type_groups[key] = model_type_groups[key][:max_models_per_type]
        models_remain = []
        for key in model_type_groups:
            models_remain += model_type_groups[key]
        models_valid = [model for model, score in models_remain]
        return models_valid

    def limit_models(self, models, model_scores, max_models):
        model_types = {model: '' for model in models}
        return self.limit_models_per_type(models=models, model_types=model_types, model_scores=model_scores, max_models_per_type=max_models)

    def _set_default_params(self):
        default_params = {'max_models': 25, 'max_models_per_type': 5}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def preprocess(self, X, preprocess=True, fit=False, compute_base_preds=True, infer=True, model=None):
        if infer:
            if (set(self.stack_columns).issubset(set(list(X.columns)))):
                compute_base_preds = False  # TODO: Consider removing, this can be dangerous but the code to make this work otherwise is complex (must rewrite predict_proba)
        if compute_base_preds:
            X_stacker = []
            for model_name in self.base_model_names:
                model_type = self.base_model_types_dict[model_name]
                model_path = self.base_model_paths_dict[model_name]
                model_loaded = model_type.load(model_path)
                if fit:
                    y_pred_proba = model_loaded.oof_pred_proba
                else:
                    y_pred_proba = model_loaded.predict_proba(X)
                X_stacker.append(y_pred_proba)  # TODO: This could get very large on a high class count problem. Consider capping to top N most frequent classes and merging least frequent
            X_stacker = self.pred_probas_to_df(X_stacker)
            X_stacker.index = X.index
            if self.use_orig_features:
                X = pd.concat([X_stacker, X], axis=1)
            else:
                X = X_stacker
        elif not self.use_orig_features:
            X = X[self.stack_columns]
        if preprocess:
            X = super().preprocess(X, model=model)
        return X

    def pred_probas_to_df(self, pred_proba: list) -> pd.DataFrame:
        if self.problem_type == MULTICLASS:
            pred_proba = np.concatenate(pred_proba, axis=1)
            pred_proba = pd.DataFrame(pred_proba, columns=self.stack_columns)
        else:
            pred_proba = pd.DataFrame(data=np.asarray(pred_proba).T, columns=self.stack_columns)
        return pred_proba

    def fit(self, X, y, k_fold=5, k_fold_start=0, k_fold_end=None, n_repeats=1, n_repeat_start=0, compute_base_preds=True, time_limit=None, **kwargs):
        start_time = time.time()
        X = self.preprocess(X=X, preprocess=False, fit=True, compute_base_preds=compute_base_preds)
        if time_limit is not None:
            time_limit = time_limit - (time.time() - start_time)
        if len(self.models) == 0:
            if self.feature_types_metadata is None:  # TODO: This is probably not the best way to do this
                self.feature_types_metadata = {'float': self.stack_columns}
            else:
                self.feature_types_metadata = copy.deepcopy(self.feature_types_metadata)
                if 'float' in self.feature_types_metadata.keys():
                    self.feature_types_metadata['float'] += self.stack_columns
                else:
                    self.feature_types_metadata['float'] = self.stack_columns
        if k_fold >= 2:
            super().fit(X=X, y=y, k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, time_limit=time_limit)
            self.bagged_mode = True
        else:
            self.models = [copy.deepcopy(self.model_base)]
            self.model_base = None
            self.bagged_mode = False
            self.models[0].set_contexts(path_context=self.path + self.models[0].name + '/')
            self.models[0].feature_types_metadata = self.feature_types_metadata  # TODO: Move this
            self.models[0].fit(X_train=X, Y_train=y)
            self._oof_pred_proba = self.models[0].predict_proba(X=X)  # TODO: Cheater value, will be overfit to valid set
            self._oof_pred_model_repeats = np.ones(shape=len(X))
            self._n_repeats = 1
            self._n_repeats_finished = 1

    def set_stack_columns(self, base_model_names):
        if self.problem_type == MULTICLASS:
            stack_columns = [model_name + '_' + str(cls) for model_name in base_model_names for cls in range(self.num_classes)]
            num_pred_cols_per_model = self.num_classes
        else:
            stack_columns = base_model_names
            num_pred_cols_per_model = 1
        return stack_columns, num_pred_cols_per_model

    # TODO: Currently double disk usage, saving model in HPO and also saving model in stacker
    def hyperparameter_tune(self, X, y, k_fold, scheduler_options=None, compute_base_preds=True, **kwargs):
        if len(self.models) != 0:
            raise ValueError('self.models must be empty to call hyperparameter_tune, value: %s' % self.models)
        if self.problem_type == REGRESSION:
            stratified = False
        else:
            stratified = True
        # TODO: Preprocess data here instead of repeatedly
        X = self.preprocess(X=X, preprocess=False, fit=True, compute_base_preds=compute_base_preds)
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=stratified, random_state=self._random_state, n_repeats=1)

        train_index, test_index = kfolds[0]
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        orig_time = scheduler_options[1]['time_out']
        scheduler_options[1]['time_out'] = orig_time * 0.8  # TODO: Scheduler doesn't early stop on final model, this is a safety net. Scheduler should be updated to early stop
        hpo_models, hpo_model_performances, hpo_results = self.model_base.hyperparameter_tune(X_train=X_train, X_test=X_test, Y_train=y_train, Y_test=y_test, scheduler_options=scheduler_options, **kwargs)
        scheduler_options[1]['time_out'] = orig_time

        stackers = {}
        stackers_performance = {}
        for i, model_name in enumerate(hpo_models.keys()):

            model_path = hpo_models[model_name]
            model_performance = hpo_model_performances[model_name]
            child = self._child_type.load(path=model_path)
            pred_proba = child.predict_proba(X_test)

            # TODO: Create new StackerEnsemble Here
            stacker = copy.deepcopy(self)

            if self.problem_type == MULTICLASS:
                oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())))
            else:
                oof_pred_proba = np.zeros(shape=len(X))
            oof_pred_model_repeats = np.zeros(shape=len(X))
            oof_pred_proba[test_index] += pred_proba
            oof_pred_model_repeats[test_index] += 1

            stacker.set_contexts(self.path + str(i) + '/')
            stacker.name = stacker.name + '/' + str(i)
            stacker._k = k_fold
            stacker._k_fold_end = 1
            stacker._n_repeats = 1
            stacker._oof_pred_proba = oof_pred_proba
            stacker._oof_pred_model_repeats = oof_pred_model_repeats
            child.name = child.name + '_fold_0'
            child.path = child.create_contexts(stacker.path + child.name + '/')
            if stacker.low_memory:
                stacker.save_child(child, verbose=False)
                stacker.models.append(child.name)
            else:
                stacker.models.append(child)

            stacker.model_base = None
            stacker.save_model_base(child.convert_to_template())
            stacker.save()
            stackers[stacker.name] = stacker.path
            stackers_performance[stacker.name] = model_performance

        # TODO: hpo_results likely not correct because no renames
        return stackers, stackers_performance, hpo_results

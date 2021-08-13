import copy, time, traceback, logging
import os
from typing import List, Union, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import psutil
from collections import defaultdict

from autogluon.core.constants import AG_ARGS, AG_ARGS_FIT, BINARY, MULTICLASS, REGRESSION, QUANTILE, REFIT_FULL_NAME, REFIT_FULL_SUFFIX
from autogluon.core.models import AbstractModel, BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel
from autogluon.core.features.feature_metadata import FeatureMetadata
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils import default_holdout_frac, get_pred_from_proba, generate_train_test_split, infer_eval_metric, compute_permutation_feature_importance, extract_column, compute_weighted_metric
from autogluon.core.utils.exceptions import TimeLimitExceeded, NotEnoughMemoryError, NoValidFeatures, NoGPUError
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_json, save_pkl

from .utils import process_hyperparameters
from ..augmentation.distill_utils import format_distillation_labels, augment_data

logger = logging.getLogger(__name__)


# FIXME: Below is major defect!
#  Weird interaction for metrics like AUC during bagging.
#  If kfold = 5, scores are 0.9, 0.85, 0.8, 0.75, and 0.7, the score is not 0.8! It is much lower because probs are combined together and AUC is recalculated
#  Do we want this to happen? Should we calculate score by 5 separate scores and then averaging instead?

# TODO: Dynamic model loading for ensemble models during prediction, only load more models if prediction is uncertain. This dynamically reduces inference time.
# TODO: Try midstack Semi-Supervised. Just take final models and re-train them, use bagged preds for SS rows. This would be very cheap and easy to try.
# TODO: Move to autogluon.core
class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'
    trainer_info_name = 'info.pkl'
    trainer_info_json_name = 'info.json'
    distill_stackname = 'distill'  # name of stack-level for distilled student models

    def __init__(self, path: str, problem_type: str, eval_metric=None,
                 num_classes=None, quantile_levels=None, low_memory=False, feature_metadata=None, k_fold=0, n_repeats=1,
                 sample_weight=None, weight_evaluation=False, save_data=False, random_state=0, verbosity=2):
        self.path = path
        self.problem_type = problem_type
        self.feature_metadata = feature_metadata
        self.save_data = save_data
        self.random_state = random_state  # Integer value added to the stack level to get the random_state for kfold splits or the train/val split if bagging is disabled
        self.verbosity = verbosity
        self.sample_weight = sample_weight  # TODO: consider redesign where Trainer doesnt need sample_weight column name and weights are separate from X
        self.weight_evaluation = weight_evaluation
        if eval_metric is not None:
            self.eval_metric = eval_metric
        else:
            self.eval_metric = infer_eval_metric(problem_type=self.problem_type)

        logger.log(25, f"AutoGluon will gauge predictive performance using evaluation metric: '{self.eval_metric.name}'")
        if not (self.eval_metric.needs_pred or self.eval_metric.needs_quantile):
            logger.log(25, "\tThis metric expects predicted probabilities rather than predicted class labels, so you'll need to use predict_proba() instead of predict()")

        logger.log(20, "\tTo change this, specify the eval_metric argument of fit()")
        self.num_classes = num_classes
        self.quantile_levels = quantile_levels
        self.feature_prune = False  # will be set to True if feature-pruning is turned on.
        self.low_memory = low_memory
        self.bagged_mode = True if k_fold >= 2 else False
        if self.bagged_mode:
            self.k_fold = k_fold  # int number of folds to do model bagging, < 2 means disabled
            self.n_repeats = n_repeats
        else:
            self.k_fold = 0
            self.n_repeats = 1

        self.model_best = None

        self.models = {}  # Dict of model name -> model object. A key, value pair only exists if a model is persisted in memory.  # TODO: v0.1 Rename and consider making private
        self.model_graph = nx.DiGraph()  # Directed Acyclic Graph (DAG) of model interactions. Describes how certain models depend on the predictions of certain other models. Contains numerous metadata regarding each model.
        self.model_full_dict = {}  # Dict of normal model -> FULL model. FULL models are produced by self.refit_single_full() and self.refit_ensemble_full().
        self._model_full_dict_val_score = {}  # Dict of FULL model -> normal model validation score in case the normal model had been deleted.
        self.reset_paths = False

        self._time_limit = None  # Internal float of the total time limit allowed for a given fit call. Used in logging statements.
        self._time_train_start = None  # Internal timestamp of the time training started for a given fit call. Used in logging statements.

        self._num_rows_train = None
        self._num_cols_train = None

        self.is_data_saved = False
        self._X_saved = False
        self._y_saved = False
        self._X_val_saved = False
        self._y_val_saved = False

        self._groups = None  # custom split indices

        self._regress_preds_asprobas = False  # whether to treat regression predictions as class-probabilities (during distillation)

        self._extra_banned_names = set()  # Names which are banned but are not used by a trained model.

        # self._exceptions_list = []  # TODO: Keep exceptions list for debugging during benchmarking.

    # path_root is the directory containing learner.pkl
    @property
    def path_root(self) -> str:
        return self.path.rsplit(os.path.sep, maxsplit=2)[0] + os.path.sep

    @property
    def path_utils(self) -> str:
        return self.path_root + 'utils' + os.path.sep

    @property
    def path_data(self) -> str:
        return self.path_utils + 'data' + os.path.sep

    def load_X(self):
        if self._X_saved:
            path = self.path_data + 'X.pkl'
            return load_pkl.load(path=path)
        return None

    def load_X_val(self):
        if self._X_val_saved:
            path = self.path_data + 'X_val.pkl'
            return load_pkl.load(path=path)
        return None

    def load_y(self):
        if self._y_saved:
            path = self.path_data + 'y.pkl'
            return load_pkl.load(path=path)
        return None

    def load_y_val(self):
        if self._y_val_saved:
            path = self.path_data + 'y_val.pkl'
            return load_pkl.load(path=path)
        return None

    def load_data(self):
        X = self.load_X()
        y = self.load_y()
        X_val = self.load_X_val()
        y_val = self.load_y_val()

        return X, y, X_val, y_val

    def save_X(self, X, verbose=True):
        path = self.path_data + 'X.pkl'
        save_pkl.save(path=path, object=X, verbose=verbose)
        self._X_saved = True

    def save_X_val(self, X, verbose=True):
        path = self.path_data + 'X_val.pkl'
        save_pkl.save(path=path, object=X, verbose=verbose)
        self._X_val_saved = True

    def save_y(self, y, verbose=True):
        path = self.path_data + 'y.pkl'
        save_pkl.save(path=path, object=y, verbose=verbose)
        self._y_saved = True

    def save_y_val(self, y, verbose=True):
        path = self.path_data + 'y_val.pkl'
        save_pkl.save(path=path, object=y, verbose=verbose)
        self._y_val_saved = True

    def get_model_names(self, stack_name: Union[List[str], str] = None, level: Union[List[int], int] = None, can_infer: bool = None, models: List[str] = None) -> List[str]:
        if models is None:
            models = list(self.model_graph.nodes)
        if stack_name is not None:
            if not isinstance(stack_name, list):
                stack_name = [stack_name]
            node_attributes: dict = self.get_models_attribute_dict(attribute='stack_name')
            models = [model_name for model_name in models if node_attributes[model_name] in stack_name]
        if level is not None:
            if not isinstance(level, list):
                level = [level]
            node_attributes: dict = self.get_models_attribute_dict(attribute='level')
            models = [model_name for model_name in models if node_attributes[model_name] in level]
        # TODO: can_infer is technically more complicated, if an ancestor can't infer then the model can't infer.
        if can_infer is not None:
            node_attributes = self.get_models_attribute_dict(attribute='can_infer')
            models = [model for model in models if node_attributes[model] == can_infer]
        return models

    def get_max_level(self, stack_name: str = None, models: List[str] = None) -> int:
        models = self.get_model_names(stack_name=stack_name, models=models)
        models_attribute_dict = self.get_models_attribute_dict(attribute='level', models=models)
        if models_attribute_dict:
            return max(list(models_attribute_dict.values()))
        else:
            return -1

    def construct_model_templates(self, hyperparameters: dict, **kwargs) -> Tuple[List[AbstractModel], dict]:
        """Constructs a list of unfit models based on the hyperparameters dict."""
        raise NotImplementedError

    def construct_model_templates_distillation(self, hyperparameters: dict, **kwargs) -> Tuple[List[AbstractModel], dict]:
        """Constructs a list of unfit models based on the hyperparameters dict for softclass distillation."""
        raise NotImplementedError

    def get_model_level(self, model_name: str) -> int:
        return self.get_model_attribute(model=model_name, attribute='level')

    def set_contexts(self, path_context):
        self.path, model_paths = self.create_contexts(path_context)
        for model, path in model_paths.items():
            self.set_model_attribute(model=model, attribute='path', val=path)

    def create_contexts(self, path_context: str) -> (str, dict):
        path = path_context
        model_paths = self.get_models_attribute_dict(attribute='path')
        for model, prev_path in model_paths.items():
            model_local_path = prev_path.split(self.path, 1)[1]
            new_path = path + model_local_path
            model_paths[model] = new_path

        return path, model_paths

    def fit(self, X, y, hyperparameters: dict, X_val=None, y_val=None, **kwargs):
        raise NotImplementedError

    # TODO: Enable easier re-mapping of trained models -> hyperparameters input (They don't share a key since name can change)
    def train_multi_levels(self, X, y, hyperparameters: dict, X_val=None, y_val=None, X_unlabeled=None, base_model_names: List[str] = None,
                           feature_prune=False, core_kwargs: dict = None, aux_kwargs: dict = None,
                           level_start=1, level_end=1, time_limit=None, name_suffix: str = None, relative_stack=True, level_time_modifier=0.333) -> List[str]:
        """
        Trains a multi-layer stack ensemble using the input data on the hyperparameters dict input.
            hyperparameters is used to determine the models used in each stack layer.
        If continuing a stack ensemble with level_start>1, ensure that base_model_names is set to the appropriate base models that will be used by the level_start level models.
        Trains both core and aux models.
            core models are standard models which are fit on the data features. Core models will also use model predictions if base_model_names was specified or if level != 1.
            aux models are ensemble models which only use the predictions of core models as features. These models never use the original features.

        level_time_modifier : float, default 0.333
            The amount of extra time given relatively to early stack levels compared to later stack levels.
            If 0, then all stack levels are given 100%/L of the time, where L is the number of stack levels.
            If 1, then all stack levels are given 100% of the time, meaning if the first level uses all of the time given to it, the other levels won't train.
            Time given to a level = remaining_time / remaining_levels * (1 + level_time_modifier), capped by total remaining time.

        Returns a list of the model names that were trained from this method call, in order of fit.
        """
        self._time_limit = time_limit
        self._time_train_start = time.time()
        time_train_start = self._time_train_start

        hyperparameters = self._process_hyperparameters(hyperparameters=hyperparameters)

        if relative_stack:
            if level_start != 1:
                raise AssertionError(f'level_start must be 1 when `relative_stack=True`. (level_start = {level_start})')
            level_add = 0
            if base_model_names:
                max_base_model_level = self.get_max_level(models=base_model_names)
                level_start = max_base_model_level + 1
                level_add = level_start - 1
                level_end += level_add
            if level_start != 1:
                hyperparameters_relative = {}
                for key in hyperparameters:
                    if isinstance(key, int):
                        hyperparameters_relative[key+level_add] = hyperparameters[key]
                    else:
                        hyperparameters_relative[key] = hyperparameters[key]
                hyperparameters = hyperparameters_relative

        core_kwargs = {} if core_kwargs is None else core_kwargs.copy()
        aux_kwargs = {} if aux_kwargs is None else aux_kwargs.copy()

        model_names_fit = []
        if level_start != level_end:
            logger.log(20, f'AutoGluon will fit {level_end - level_start + 1} stack levels (L{level_start} to L{level_end}) ...')
        for level in range(level_start, level_end + 1):
            core_kwargs_level = core_kwargs.copy()
            aux_kwargs_level = aux_kwargs.copy()
            if time_limit is not None:
                time_train_level_start = time.time()
                levels_left = level_end - level + 1
                time_left = time_limit - (time_train_level_start - time_train_start)
                time_limit_for_level = min(time_left / levels_left * (1 + level_time_modifier), time_left)
                time_limit_core = time_limit_for_level
                time_limit_aux = max(time_limit_for_level * 0.1, min(time_limit, 360))  # Allows aux to go over time_limit, but only by a small amount
                core_kwargs_level['time_limit'] = core_kwargs_level.get('time_limit', time_limit_core)
                aux_kwargs_level['time_limit'] = aux_kwargs_level.get('time_limit', time_limit_aux)
            if level != 1:
                feature_prune = False  # TODO: Enable feature prune on levels > 1
            base_model_names, aux_models = self.stack_new_level(
                X=X, y=y, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled,
                models=hyperparameters, level=level, base_model_names=base_model_names,
                feature_prune=feature_prune,
                core_kwargs=core_kwargs_level, aux_kwargs=aux_kwargs_level, name_suffix=name_suffix,
            )
            model_names_fit += base_model_names + aux_models
        self._time_limit = None
        self.save()
        return model_names_fit

    def stack_new_level(self, X, y, models: Union[List[AbstractModel], dict], X_val=None, y_val=None, X_unlabeled=None, level=1, base_model_names: List[str] = None,
                        feature_prune=False, core_kwargs: dict = None, aux_kwargs: dict = None, name_suffix: str = None) -> (List[str], List[str]):
        """
        Similar to calling self.stack_new_level_core, except auxiliary models will also be trained via a call to self.stack_new_level_aux, with the models trained from self.stack_new_level_core used as base models.
        """
        if base_model_names is None:
            base_model_names = []
        if level < 1:
            raise AssertionError(f'Stack level must be >= 1, but level={level}.')
        elif not base_model_names and level > 1:
            logger.log(30, f'Warning: Training models at stack level {level}, but no base models were specified.')
        elif base_model_names and level == 1:
            raise AssertionError(f'Stack level 1 models cannot have base models, but base_model_names={base_model_names}.')
        core_kwargs = {} if core_kwargs is None else core_kwargs.copy()
        aux_kwargs = {} if aux_kwargs is None else aux_kwargs.copy()
        if name_suffix:
            core_kwargs['name_suffix'] = core_kwargs.get('name_suffix', '') + name_suffix
            aux_kwargs['name_suffix'] = aux_kwargs.get('name_suffix', '') + name_suffix
        core_models = self.stack_new_level_core(X=X, y=y, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled, models=models,
                                                level=level, base_model_names=base_model_names, feature_prune=feature_prune, **core_kwargs)

        if X_val is None:
            aux_models = self.stack_new_level_aux(X=X, y=y, base_model_names=core_models, level=level+1, **aux_kwargs)
        else:
            aux_models = self.stack_new_level_aux(X=X_val, y=y_val, fit=False, base_model_names=core_models, level=level+1, **aux_kwargs)
        return core_models, aux_models

    def stack_new_level_core(self, X, y, models: Union[List[AbstractModel], dict], X_val=None, y_val=None, X_unlabeled=None,
                             level=1, base_model_names: List[str] = None, stack_name='core',
                             ag_args=None, ag_args_fit=None, ag_args_ensemble=None, excluded_model_types=None, ensemble_type=StackerEnsembleModel,
                             name_suffix: str = None, get_models_func=None, refit_full=False, **kwargs) -> List[str]:
        """
        Trains all models using the data provided.
        If level > 1, then the models will use base model predictions as additional features.
            The base models used can be specified via base_model_names.
        If self.bagged_mode, then models will be trained as StackerEnsembleModels.
        The data provided in this method should not contain stack features, as they will be automatically generated if necessary.
        """
        if get_models_func is None:
            get_models_func = self.construct_model_templates
        if base_model_names is None:
            base_model_names = []
        if not self.bagged_mode and level != 1:
            raise ValueError('Stack Ensembling is not valid for non-bagged mode.')

        if isinstance(models, dict):
            get_models_kwargs = dict(
                level=level,
                name_suffix=name_suffix,
                ag_args=ag_args,
                ag_args_fit=ag_args_fit,
                excluded_model_types=excluded_model_types,
            )

            if self.bagged_mode:
                if level == 1:
                    (base_model_names, base_model_paths, base_model_types) = (None, None, None)
                elif level > 1:
                    base_model_names, base_model_paths, base_model_types = self._get_models_load_info(model_names=base_model_names)
                    if len(base_model_names) == 0:
                        logger.log(20, 'No base models to train on, skipping stack level...')
                        return []
                else:
                    raise AssertionError(f'Stack level cannot be less than 1! level = {level}')

                ensemble_kwargs = {
                    'base_model_names': base_model_names,
                    'base_model_paths_dict': base_model_paths,
                    'base_model_types_dict': base_model_types,
                    'random_state': level + self.random_state,
                }
                get_models_kwargs.update(dict(
                    ag_args_ensemble=ag_args_ensemble,
                    ensemble_type=ensemble_type,
                    ensemble_kwargs=ensemble_kwargs,
                ))
            models, model_args_fit = get_models_func(hyperparameters=models, **get_models_kwargs)
            if model_args_fit:
                hyperparameter_tune_kwargs = {
                    model_name: model_args_fit[model_name]['hyperparameter_tune_kwargs']
                    for model_name in model_args_fit if 'hyperparameter_tune_kwargs' in model_args_fit[model_name]
                }
                kwargs['hyperparameter_tune_kwargs'] = hyperparameter_tune_kwargs
        logger.log(20, f'Fitting {len(models)} L{level} models ...')
        X_init = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=True)
        if X_val is not None:
            X_val = self.get_inputs_to_stacker(X_val, base_models=base_model_names, fit=False)
        if refit_full and X_val is not None:
            X_init = pd.concat([X_init, X_val])
            y = pd.concat([y, y_val])
            X_val = None
            y_val = None
        if X_unlabeled is not None:
            X_unlabeled = self.get_inputs_to_stacker(X_unlabeled, base_models=base_model_names, fit=False)

        fit_kwargs = dict(num_classes=self.num_classes)

        # FIXME: TODO: v0.1 X_unlabeled isn't cached so it won't be available during refit_full or fit_extra.
        return self._train_multi(X=X_init, y=y, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled,
                                 models=models, level=level, stack_name=stack_name, fit_kwargs=fit_kwargs, **kwargs)

    # TODO: Consider making level be auto-determined based off of max(base_model_levels)+1
    # TODO: Remove name_suffix, hacked in
    # TODO: X can be optional because it isn't needed if fit=True
    def stack_new_level_aux(self, X, y, base_model_names: List[str], level, fit=True, stack_name='aux1', time_limit=None, name_suffix: str = None, get_models_func=None, check_if_best=True) -> List[str]:
        """
        Trains auxiliary models (currently a single weighted ensemble) using the provided base models.
        Level must be greater than the level of any of the base models.
        Auxiliary models never use the original features and only train with the predictions of other models as features.
        """
        X_stack_preds = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=fit, use_orig_features=False)
        if self.weight_evaluation:
            X, w = extract_column(X, self.sample_weight)  # TODO: consider redesign with w as separate arg instead of bundled inside X
            if w is not None:
                X_stack_preds[self.sample_weight] = w.values/w.mean()
        return self.generate_weighted_ensemble(X=X_stack_preds, y=y, level=level, base_model_names=base_model_names, k_fold=1, n_repeats=1, stack_name=stack_name, time_limit=time_limit, name_suffix=name_suffix, get_models_func=get_models_func, check_if_best=check_if_best)

    def predict(self, X, model=None):
        if model is None:
            model = self._get_best()
        return self._predict_model(X, model)

    def predict_proba(self, X, model=None):
        if model is None:
            model = self._get_best()
        return self._predict_proba_model(X, model)

    def _get_best(self):
        if self.model_best is not None:
            return self.model_best
        else:
            return self.get_model_best()

    # Note: model_pred_proba_dict is mutated in this function to minimize memory usage
    def get_inputs_to_model(self, model, X, model_pred_proba_dict=None, fit=False, preprocess_nonadaptive=False):
        """
        For output X:
            If preprocess_nonadaptive=False, call model.predict(X)
            If preprocess_nonadaptive=True, call model.predict(X, preprocess_nonadaptive=False)
        """
        if isinstance(model, str):
            # TODO: Remove unnecessary load when no stacking
            model = self.load_model(model)
        model_level = self.get_model_level(model.name)
        if model_level > 1 and isinstance(model, StackerEnsembleModel):
            if fit:
                model_pred_proba_dict = None
            else:
                model_set = self.get_minimum_model_set(model)
                model_set = [m for m in model_set if m != model.name]  # TODO: Can probably be faster, get this result from graph
                model_pred_proba_dict = self.get_model_pred_proba_dict(X=X, models=model_set, model_pred_proba_dict=model_pred_proba_dict, fit=fit)
            X = model.preprocess(X=X, preprocess_nonadaptive=preprocess_nonadaptive, fit=fit, model_pred_proba_dict=model_pred_proba_dict)
        elif preprocess_nonadaptive:
            X = model.preprocess(X=X, preprocess_stateful=False)
        return X

    def score(self, X, y, model=None, weights=None) -> float:
        if self.eval_metric.needs_pred or self.eval_metric.needs_quantile:
            y_pred = self.predict(X=X, model=model)
        else:
            y_pred = self.predict_proba(X=X, model=model)
        return compute_weighted_metric(y, y_pred, self.eval_metric, weights, weight_evaluation=self.weight_evaluation,
                                       quantile_levels=self.quantile_levels)

    def score_with_y_pred_proba(self, y, y_pred_proba, weights=None) -> float:
        if self.eval_metric.needs_pred or self.eval_metric.needs_quantile:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        else:
            y_pred = y_pred_proba
        return compute_weighted_metric(y, y_pred, self.eval_metric, weights, weight_evaluation=self.weight_evaluation,
                                       quantile_levels=self.quantile_levels)

    # TODO: Consider adding persist to disk functionality for pred_proba dictionary to lessen memory burden on large multiclass problems.
    #  For datasets with 100+ classes, this function could potentially run the system OOM due to each pred_proba numpy array taking significant amounts of space.
    #  This issue already existed in the previous level-based version but only had the minimum required predictions in memory at a time, whereas this has all model predictions in memory.
    # TODO: Add memory optimal topological ordering -> Minimize amount of pred_probas in memory at a time, delete pred probas that are no longer required
    # Optimally computes pred_probas for each model in `models`. Will compute each necessary model only once and store its predictions in a dictionary.
    # Note: Mutates model_pred_proba_dict and model_pred_time_dict input if present to minimize memory usage
    # fit = get oof pred proba
    # if record_pred_time is `True`, outputs tuple of dicts (model_pred_proba_dict, model_pred_time_dict), else output only model_pred_proba_dict
    def get_model_pred_proba_dict(self, X, models, model_pred_proba_dict=None, model_pred_time_dict=None, fit=False, record_pred_time=False):
        if model_pred_proba_dict is None:
            model_pred_proba_dict = {}
        if model_pred_time_dict is None:
            model_pred_time_dict = {}

        if fit:
            model_pred_order = [model for model in models if model not in model_pred_proba_dict.keys()]
        else:
            model_set = set()
            for model in models:
                if model in model_set:
                    continue
                min_model_set = set(self.get_minimum_model_set(model))
                model_set = model_set.union(min_model_set)
            model_set = model_set.difference(set(model_pred_proba_dict.keys()))
            models_to_load = list(model_set)
            subgraph = nx.subgraph(self.model_graph, models_to_load)

            # For model in model_pred_proba_dict, remove model node from graph and all ancestors that have no remaining descendants and are not in `models`
            models_to_ignore = [model for model in models_to_load if (model not in models) and (not list(subgraph.successors(model)))]
            while models_to_ignore:
                model = models_to_ignore[0]
                predecessors = list(subgraph.predecessors(model))
                subgraph.remove_node(model)
                models_to_ignore = models_to_ignore[1:]
                for predecessor in predecessors:
                    if (predecessor not in models) and (not list(subgraph.successors(predecessor))) and (predecessor not in models_to_ignore):
                        models_to_ignore.append(predecessor)

            # Get model prediction order
            model_pred_order = list(nx.lexicographical_topological_sort(subgraph))

        # Compute model predictions in topological order
        for model_name in model_pred_order:
            if record_pred_time:
                time_start = time.time()

            if fit:
                model_type = self.get_model_attribute(model=model_name, attribute='type')
                if issubclass(model_type, BaggedEnsembleModel):
                    model_path = self.get_model_attribute(model=model_name, attribute='path')
                    model_pred_proba_dict[model_name] = model_type.load_oof(path=model_path)
                else:
                    raise AssertionError(f'Model {model_name} must be a BaggedEnsembleModel to return oof_pred_proba')
            else:
                model = self.load_model(model_name=model_name)
                if isinstance(model, StackerEnsembleModel):
                    preprocess_kwargs = dict(infer=False, model_pred_proba_dict=model_pred_proba_dict)
                    model_pred_proba_dict[model_name] = model.predict_proba(X, **preprocess_kwargs)
                else:
                    model_pred_proba_dict[model_name] = model.predict_proba(X)

            if record_pred_time:
                time_end = time.time()
                model_pred_time_dict[model_name] = time_end - time_start

        if record_pred_time:
            return model_pred_proba_dict, model_pred_time_dict
        else:
            return model_pred_proba_dict

    # TODO: Remove _get_inputs_to_stacker_legacy eventually, move logic internally into this function instead
    def get_inputs_to_stacker(self, X, base_models, model_pred_proba_dict=None, fit=False, use_orig_features=True):
        if base_models is None:
            base_models = []
        if not fit:
            model_pred_proba_dict = self.get_model_pred_proba_dict(X=X, models=base_models, model_pred_proba_dict=model_pred_proba_dict)
            model_pred_proba_list = [model_pred_proba_dict[model] for model in base_models]
        else:
            # TODO: After _get_inputs_to_stacker_legacy is removed, this if/else is not necessary, instead pass fit param to get_model_pred_proba_dict()
            model_pred_proba_list = None

        X_stacker_input = self._get_inputs_to_stacker_legacy(X=X, level_start=1, level_end=2, model_levels={1: base_models}, y_pred_probas=model_pred_proba_list, fit=fit)
        if not use_orig_features:
            X_stacker_input = X_stacker_input.drop(columns=X.columns)
        return X_stacker_input

    # TODO: Legacy code, still used during training because it is technically slightly faster and more memory efficient than get_model_pred_proba_dict()
    #  Remove in future as it limits flexibility in stacker inputs during training
    def _get_inputs_to_stacker_legacy(self, X, level_start, level_end, model_levels, y_pred_probas=None, fit=False):
        if level_start > level_end:
            raise AssertionError(f'level_start cannot be greater than level end: ({level_start}, {level_end})')
        if (level_start == 1) and (level_end == 1):
            return X
        if fit:
            if level_start > 1:
                dummy_stacker_start = self._get_dummy_stacker(level=level_start, model_levels=model_levels, use_orig_features=True)
                cols_to_drop = dummy_stacker_start.stack_columns
                X = X.drop(cols_to_drop, axis=1)
            dummy_stacker = self._get_dummy_stacker(level=level_end, model_levels=model_levels, use_orig_features=True)
            X = dummy_stacker.preprocess(X=X, preprocess_nonadaptive=False, fit=True, compute_base_preds=True)
        elif y_pred_probas is not None:
            if y_pred_probas == []:
                return X
            dummy_stacker = self._get_dummy_stacker(level=level_end, model_levels=model_levels, use_orig_features=True)
            X_stacker = dummy_stacker.pred_probas_to_df(pred_proba=y_pred_probas, index=X.index)
            if dummy_stacker.params['use_orig_features']:
                if level_start > 1:
                    dummy_stacker_start = self._get_dummy_stacker(level=level_start, model_levels=model_levels, use_orig_features=True)
                    cols_to_drop = dummy_stacker_start.stack_columns
                    X = X.drop(cols_to_drop, axis=1)
                X = pd.concat([X_stacker, X], axis=1)
            else:
                X = X_stacker
        else:
            dummy_stackers = {}
            for level in range(level_start, level_end+1):
                if level > 1:
                    dummy_stackers[level] = self._get_dummy_stacker(level=level, model_levels=model_levels, use_orig_features=True)
            for level in range(level_start, level_end):
                if level > 1:
                    cols_to_drop = dummy_stackers[level].stack_columns
                else:
                    cols_to_drop = []
                X = dummy_stackers[level+1].preprocess(X=X, preprocess_nonadaptive=False, fit=False, compute_base_preds=True)
                if len(cols_to_drop) > 0:
                    X = X.drop(cols_to_drop, axis=1)
        return X

    # You must have previously called fit() with cache_data=True
    # Fits _FULL versions of specified models, but does NOT link them (_FULL stackers will still use normal models as input)
    def refit_single_full(self, X=None, y=None, X_val=None, y_val=None, X_unlabeled=None, models=None) -> List[str]:
        if X is None:
            X = self.load_X()
        if X_val is None:
            X_val = self.load_X_val()
        if y is None:
            y = self.load_y()
        if y_val is None:
            y_val = self.load_y_val()

        if models is None:
            models = self.get_model_names()

        model_levels = dict()
        ignore_models = []
        ignore_stack_names = [REFIT_FULL_NAME]
        for stack_name in ignore_stack_names:
            ignore_models += self.get_model_names(stack_name=stack_name)  # get_model_names returns [] if stack_name does not exist
        models = [model for model in models if model not in ignore_models]
        for model in models:
            model_level = self.get_model_level(model)
            if model_level not in model_levels:
                model_levels[model_level] = []
            model_levels[model_level].append(model)

        levels = sorted(model_levels.keys())
        models_trained_full = []
        model_full_dict = {}
        for level in levels:
            models_level = model_levels[level]
            for model in models_level:
                model = self.load_model(model)
                model_name = model.name
                model_full = model.convert_to_refit_full_template()
                # Mitigates situation where bagged models barely had enough memory and refit requires more. Worst case results in OOM, but this lowers chance of failure.
                model_full._user_params_aux['max_memory_usage_ratio'] = model.params_aux['max_memory_usage_ratio'] * 1.15
                # TODO: Do it for all models in the level at once to avoid repeated processing of data?
                base_model_names = self.get_base_model_names(model_name)
                stacker_type = type(model)
                if issubclass(stacker_type, WeightedEnsembleModel):
                    # TODO: Technically we don't need to re-train the weighted ensemble, we could just copy the original and re-use the weights.
                    w = None
                    if X_val is None:
                        if self.weight_evaluation:
                            X, w = extract_column(X, self.sample_weight)
                        X_stack_preds = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=True, use_orig_features=False)
                        y_input = y
                    else:
                        if self.weight_evaluation:
                            X_val, w = extract_column(X_val, self.sample_weight)
                        X_stack_preds = self.get_inputs_to_stacker(X_val, base_models=base_model_names, fit=False, use_orig_features=False)  # TODO: May want to cache this during original fit, as we do with OOF preds
                        y_input = y_val
                    if w is not None:
                        X_stack_preds[self.sample_weight] = w.values/w.mean()

                    orig_weights = model._get_model_weights()
                    base_model_names = list(orig_weights.keys())
                    weights = list(orig_weights.values())

                    child_hyperparameters = {
                        AG_ARGS: {'model_type': 'SIMPLE_ENS_WEIGHTED'},
                        'weights': weights,
                    }

                    # TODO: stack_name=REFIT_FULL_NAME_AUX?
                    models_trained = self.generate_weighted_ensemble(X=X_stack_preds, y=y_input, level=level, stack_name=REFIT_FULL_NAME, k_fold=1, n_repeats=1,
                                                                     base_model_names=base_model_names, name_suffix=REFIT_FULL_SUFFIX, save_bag_folds=True,
                                                                     check_if_best=False, child_hyperparameters=child_hyperparameters)
                    # TODO: Do the below more elegantly, ideally as a parameter to the trainer train function to disable recording scores/pred time.
                    for model_weighted_ensemble in models_trained:
                        model_loaded = self.load_model(model_weighted_ensemble)
                        model_loaded.val_score = None
                        model_loaded.predict_time = None
                        self.set_model_attribute(model=model_weighted_ensemble, attribute='val_score', val=None)
                        self.save_model(model_loaded)
                else:
                    models_trained = self.stack_new_level_core(X=X, y=y, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled, models=[model_full], base_model_names=base_model_names, level=level, stack_name=REFIT_FULL_NAME,
                                                               hyperparameter_tune_kwargs=None, feature_prune=False, k_fold=0, n_repeats=1, ensemble_type=stacker_type, refit_full=True)
                if len(models_trained) == 1:
                    model_full_dict[model_name] = models_trained[0]
                for model_trained in models_trained:
                    self._model_full_dict_val_score[model_trained] = self.get_model_attribute(model_name, 'val_score')
                models_trained_full += models_trained

        keys_to_del = []
        for model in model_full_dict.keys():
            if model_full_dict[model] not in models_trained_full:
                keys_to_del.append(model)
        for key in keys_to_del:
            del model_full_dict[key]
        self.model_full_dict.update(model_full_dict)
        self.save()  # TODO: This could be more efficient by passing in arg to not save if called by refit_ensemble_full since it saves anyways later.
        return models_trained_full

    # Fits _FULL models and links them in the stack so _FULL models only use other _FULL models as input during stacking
    # If model is specified, will fit all _FULL models that are ancestors of the provided model, automatically linking them.
    # If no model is specified, all models are refit and linked appropriately.
    def refit_ensemble_full(self, model='all') -> dict:
        if model == 'all':
            ensemble_set = self.get_model_names()
        else:
            if model == 'best':
                model = self.get_model_best()
            ensemble_set = self.get_minimum_model_set(model)
        existing_models = self.get_model_names()
        ensemble_set_valid = []
        for model in ensemble_set:
            if model in self.model_full_dict and self.model_full_dict[model] in existing_models:
                logger.log(20, f"Model '{model}' already has a refit _FULL model: '{self.model_full_dict[model]}', skipping refit...")
            else:
                ensemble_set_valid.append(model)
        if ensemble_set_valid:
            models_trained_full = self.refit_single_full(models=ensemble_set_valid)
        else:
            models_trained_full = []

        for model_full in models_trained_full:
            # TODO: Consider moving base model info to a separate pkl file so that it can be edited without having to load/save the model again
            #  Downside: Slower inference speed when models are not persisted in memory prior.
            model_loaded = self.load_model(model_full)
            if isinstance(model_loaded, StackerEnsembleModel):
                for stack_column_prefix in model_loaded.stack_column_prefix_lst:
                    base_model = model_loaded.stack_column_prefix_to_model_map[stack_column_prefix]
                    new_base_model = self.model_full_dict[base_model]
                    new_base_model_type = self.get_model_attribute(model=new_base_model, attribute='type')
                    new_base_model_path = self.get_model_attribute(model=new_base_model, attribute='path')

                    model_loaded.base_model_paths_dict[new_base_model] = new_base_model_path
                    model_loaded.base_model_types_dict[new_base_model] = new_base_model_type
                    model_loaded.base_model_names.append(new_base_model)
                    model_loaded.stack_column_prefix_to_model_map[stack_column_prefix] = new_base_model

            model_loaded.save()  # TODO: Avoid this!

            # Remove old edges and add new edges
            edges_to_remove = list(self.model_graph.in_edges(model_loaded.name))
            self.model_graph.remove_edges_from(edges_to_remove)
            if isinstance(model_loaded, StackerEnsembleModel):
                for stack_column_prefix in model_loaded.stack_column_prefix_lst:
                    base_model_name = model_loaded.stack_column_prefix_to_model_map[stack_column_prefix]
                    self.model_graph.add_edge(base_model_name, model_loaded.name)

        self.save()
        return copy.deepcopy(self.model_full_dict)

    # TODO: Take best performance model with lowest inference
    def get_model_best(self, can_infer=None, allow_full=True):
        models = self.get_model_names(can_infer=can_infer)
        if not models:
            raise AssertionError('Trainer has no fit models that can infer.')
        model_performances = self.get_models_attribute_dict(attribute='val_score')
        perfs = [(m, model_performances[m]) for m in models if model_performances[m] is not None]
        if not perfs:
            model_full_dict_inverse = {full: orig for orig, full in self.model_full_dict.items()}
            models = [m for m in models if m in model_full_dict_inverse]
            perfs = [(m, self._get_full_model_val_score(m)) for m in models]
            if not perfs:
                raise AssertionError('No fit models that can infer exist with a validation score to choose the best model.')
            elif not allow_full:
                raise AssertionError('No fit models that can infer exist with a validation score to choose the best model, but refit_full models exist. Set `allow_full=True` to get the best refit_full model.')
        return max(perfs, key=lambda i: i[1])[0]

    def save_model(self, model, reduce_memory=True):
        # TODO: In future perhaps give option for the reduce_memory_size arguments, perhaps trainer level variables specified by user?
        if reduce_memory:
            model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if self.low_memory:
            model.save()
        else:
            self.models[model.name] = model

    def save(self):
        models = self.models
        if self.low_memory:
            self.models = {}
        save_pkl.save(path=self.path + self.trainer_file_name, object=self)
        if self.low_memory:
            self.models = models

    def persist_models(self, model_names='all', with_ancestors=False, max_memory=None) -> List[str]:
        if model_names == 'all':
            model_names = self.get_model_names()
        elif model_names == 'best':
            if self.model_best is not None:
                model_names = [self.model_best]
            else:
                model_names = [self.get_model_best(can_infer=True)]
        if not isinstance(model_names, list):
            raise ValueError(f'model_names must be a list of model names. Invalid value: {model_names}')
        if with_ancestors:
            model_names = self.get_minimum_models_set(model_names)
        model_names_already_persisted = [model_name for model_name in model_names if model_name in self.models]
        if model_names_already_persisted:
            logger.log(30, f'The following {len(model_names_already_persisted)} models were already persisted and will be ignored in the model loading process: {model_names_already_persisted}')
        model_names = [model_name for model_name in model_names if model_name not in model_names_already_persisted]
        if not model_names:
            logger.log(30, f'No valid unpersisted models were specified to be persisted, so no change in model persistence was performed.')
            return []
        if max_memory is not None:
            info = self.get_models_info(model_names)
            model_mem_size_map = {model: info[model]['memory_size'] for model in model_names}
            for model in model_mem_size_map:
                if 'children_info' in info[model]:
                    for child in info[model]['children_info'].values():
                        model_mem_size_map[model] += child['memory_size']
            total_mem_required = sum(model_mem_size_map.values())
            available_mem = psutil.virtual_memory().available
            memory_proportion = total_mem_required / available_mem
            if memory_proportion > max_memory:
                logger.log(30, f'Models will not be persisted in memory as they are expected to require {round(memory_proportion * 100, 2)}% of memory, which is greater than the specified max_memory limit of {round(max_memory*100, 2)}%.')
                logger.log(30, f'\tModels will be loaded on-demand from disk to maintain safe memory usage, increasing inference latency. If inference latency is a concern, try to use smaller models or increase the value of max_memory.')
                return []
            else:
                logger.log(20, f'Persisting {len(model_names)} models in memory. Models will require {round(memory_proportion*100, 2)}% of memory.')

        models = []
        for model_name in model_names:
            model = self.load_model(model_name)
            self.models[model.name] = model
            models.append(model)

        for model in models:
            # TODO: Move this to model code
            if isinstance(model, BaggedEnsembleModel):
                for fold, fold_model in enumerate(model.models):
                    if isinstance(fold_model, str):
                        model.models[fold] = model.load_child(fold_model)
        return model_names

    # TODO: model_name change to model in params
    def load_model(self, model_name: str, path: str = None, model_type=None) -> AbstractModel:
        if isinstance(model_name, AbstractModel):
            return model_name
        if model_name in self.models.keys():
            return self.models[model_name]
        else:
            if path is None:
                path = self.get_model_attribute(model=model_name, attribute='path')
            if model_type is None:
                model_type = self.get_model_attribute(model=model_name, attribute='type')
            return model_type.load(path=path, reset_paths=self.reset_paths)

    def unpersist_models(self, model_names='all') -> list:
        if model_names == 'all':
            model_names = list(self.models.keys())
        if not isinstance(model_names, list):
            raise ValueError(f'model_names must be a list of model names. Invalid value: {model_names}')
        unpersisted_models = []
        for model in model_names:
            if model in self.models:
                self.models.pop(model)
                unpersisted_models.append(model)
        if unpersisted_models:
            logger.log(20, f'Unpersisted {len(unpersisted_models)} models: {unpersisted_models}')
        else:
            logger.log(30, f'No valid persisted models were specified to be unpersisted, so no change in model persistence was performed.')
        return unpersisted_models

    def generate_weighted_ensemble(self, X, y, level, base_model_names, k_fold=1, n_repeats=1, stack_name=None, hyperparameters=None,
                                   time_limit=None, name_suffix: str = None, save_bag_folds=None, check_if_best=True, child_hyperparameters=None,
                                   get_models_func=None) -> List[str]:
        if get_models_func is None:
            get_models_func = self.construct_model_templates
        if len(base_model_names) == 0:
            logger.log(20, 'No base models to train on, skipping weighted ensemble...')
            return []

        if child_hyperparameters is None:
            child_hyperparameters = {}

        if save_bag_folds is None:
            can_infer_dict = self.get_models_attribute_dict('can_infer', models=base_model_names)
            if False in can_infer_dict.values():
                save_bag_folds = False
            else:
                save_bag_folds = True

        weighted_ensemble_model, _ = get_models_func(
            hyperparameters={
                'default': {
                    'ENS_WEIGHTED': [child_hyperparameters],
                }
            },
            ensemble_type=WeightedEnsembleModel,
            ensemble_kwargs=dict(
                base_model_names=base_model_names,
                base_model_paths_dict=self.get_models_attribute_dict(attribute='path', models=base_model_names),
                base_model_types_dict=self.get_models_attribute_dict(attribute='type', models=base_model_names),
                base_model_types_inner_dict=self.get_models_attribute_dict(attribute='type_inner', models=base_model_names),
                base_model_performances_dict=self.get_models_attribute_dict(attribute='val_score', models=base_model_names),
                hyperparameters=hyperparameters,
                random_state=level + self.random_state,
            ),
            ag_args={'name_bag_suffix': ''},
            ag_args_ensemble={'save_bag_folds': save_bag_folds},
            name_suffix=name_suffix,
            level=level,
        )
        weighted_ensemble_model = weighted_ensemble_model[0]
        w = None
        if self.weight_evaluation:
            X, w = extract_column(X, self.sample_weight)
        models = self._train_multi(
            X=X,
            y=y,
            X_val=None,
            y_val=None,
            models=[weighted_ensemble_model],
            k_fold=k_fold,
            n_repeats=n_repeats,
            hyperparameter_tune_kwargs=None,
            feature_prune=False,
            stack_name=stack_name,
            level=level,
            time_limit=time_limit,
            ens_sample_weight=w,
            fit_kwargs=dict(num_classes=self.num_classes, groups=None),  # FIXME: Is this the right way to do this?
        )
        for weighted_ensemble_model_name in models:
            if check_if_best and weighted_ensemble_model_name in self.get_model_names():
                if self.model_best is None:
                    self.model_best = weighted_ensemble_model_name
                else:
                    best_score = self.get_model_attribute(self.model_best, 'val_score')
                    cur_score = self.get_model_attribute(weighted_ensemble_model_name, 'val_score')
                    if cur_score > best_score:
                        # new best model
                        self.model_best = weighted_ensemble_model_name
        return models

    def _train_single(self, X, y, model: AbstractModel, X_val=None, y_val=None, **model_fit_kwargs) -> AbstractModel:
        """
        Trains model but does not add the trained model to this Trainer.
        Returns trained model object.
        """
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **model_fit_kwargs)
        return model

    def _train_and_save(self, X, y, model: AbstractModel, X_val=None, y_val=None, stack_name='core', level=1, **model_fit_kwargs) -> List[str]:
        """
        Trains model and saves it to disk, returning a list with a single element: The name of the model, or no elements if training failed.
        If the model name is returned:
            The model can be accessed via self.load_model(model.name).
            The model will have metadata information stored in self.model_graph.
            The model's name will be appended to self.models_level[stack_name][level]
            The model will be accessible and usable through any Trainer function that takes as input 'model' or 'model_name'.
        Note: self._train_and_save should not be used outside of self._train_single_full
        """
        fit_start_time = time.time()
        time_limit = model_fit_kwargs.get('time_limit', None)
        model_names_trained = []
        try:
            fit_log_message = f'Fitting model: {model.name} ...'
            if time_limit is not None:
                if time_limit <= 0:
                    logger.log(15, f'Skipping {model.name} due to lack of time remaining.')
                    return model_names_trained
                if self._time_limit is not None and self._time_train_start is not None:
                    time_left_total = self._time_limit - (fit_start_time - self._time_train_start)
                else:
                    time_left_total = time_limit
                fit_log_message += f' Training model for up to {round(time_limit, 2)}s of the {round(time_left_total, 2)}s of remaining time.'
            logger.log(20, fit_log_message)
            model = self._train_single(X, y, model, X_val, y_val, **model_fit_kwargs)
            fit_end_time = time.time()
            if self.weight_evaluation:
                w = model_fit_kwargs.get('sample_weight', None)
                w_val = model_fit_kwargs.get('sample_weight_val', None)
            else:
                w = None
                w_val = None
            if isinstance(model, BaggedEnsembleModel):
                if X_val is not None and y_val is not None:
                    score = model.score(X=X_val, y=y_val, sample_weight=w_val)
                elif model.is_valid_oof() or isinstance(model, WeightedEnsembleModel):
                    score = model.score_with_oof(y=y, sample_weight=w)
                else:
                    score = None
            else:
                if X_val is not None and y_val is not None:
                    score = model.score(X=X_val, y=y_val, sample_weight=w_val)
                else:
                    score = None
            pred_end_time = time.time()
            if model.fit_time is None:
                model.fit_time = fit_end_time - fit_start_time
            if model.predict_time is None:
                if score is None:
                    model.predict_time = None
                else:
                    model.predict_time = pred_end_time - fit_end_time
            model.val_score = score
            # TODO: Add recursive=True to avoid repeatedly loading models each time this is called for bagged ensembles (especially during repeated bagging)
            self.save_model(model=model)
        except TimeLimitExceeded:
            logger.log(20, f'\tTime limit exceeded... Skipping {model.name}.')
            # logger.log(20, '\tTime wasted: ' + str(time.time() - fit_start_time))
            del model
        except NotEnoughMemoryError:
            logger.warning(f'\tNot enough memory to train {model.name}... Skipping this model.')
            del model
        except NoValidFeatures:
            logger.warning(f'\tNo valid features to train {model.name}... Skipping this model.')
            del model
        except NoGPUError:
            logger.warning(f'\tNo GPUs available to train {model.name}... Skipping this model.')
            del model
        except ImportError as err:
            logger.error(f'\tWarning: Exception caused {model.name} to fail during training (ImportError)... Skipping this model.')
            logger.error(f'\t\t{err}')
            if self.verbosity > 2:
                logger.exception('Detailed Traceback:')
        except Exception as err:
            logger.error(f'\tWarning: Exception caused {model.name} to fail during training... Skipping this model.')
            logger.error(f'\t\t{err}')
            if self.verbosity > 0:
                logger.exception('Detailed Traceback:')
            del model
        else:
            self._add_model(model=model, stack_name=stack_name, level=level)
            model_names_trained.append(model.name)
            if self.low_memory:
                del model
        return model_names_trained

    def _add_model(self, model: AbstractModel, stack_name: str = 'core', level: int = 1) -> bool:
        """
        Registers the fit model in the Trainer object. Stores information such as model performance, save path, model type, and more.
        To use a model in Trainer, self._add_model must be called.
        If self.low_memory, then the model object will be deleted after this call. Use Trainer directly to leverage the model further.

        Parameters
        ----------
        model : AbstractModel
            Model which has been fit. This model will be registered to the Trainer.
        stack_name : str, default 'core'
            Stack name to assign the model to. This is used for advanced functionality.
        level : int, default 1
            Stack level of the stack name to assign the model to. This is used for advanced functionality.
            The model's name is appended to self.models_level[stack_name][level]
            The model's base_models (if it has any) must all be a lower level than the model.

        Returns
        -------
        boolean, True if model was registered, False if model was found to be invalid and not registered.
        """
        if model.val_score is not None:
            if model.eval_metric.name != self.eval_metric.name:
                logger.log(20, f'\tNote: model has different eval_metric than default.')
            logger.log(20, f'\t{round(model.val_score, 4)}\t = Validation score   ({model.eval_metric.name})')
        if model.fit_time is not None:
            logger.log(20, f'\t{round(model.fit_time, 2)}s\t = Training   runtime')
        if model.predict_time is not None:
            logger.log(20, f'\t{round(model.predict_time, 2)}s\t = Validation runtime')
        if model.val_score is not None and np.isnan(model.val_score):
            logger.warning(f'WARNING: {model.name} has a val_score of {model.val_score} (NaN)! This should never happen. The model will not be saved to avoid instability.')
            return False
        # TODO: Add to HPO
        if isinstance(model, BaggedEnsembleModel):
            type_inner = model._child_type
        else:
            type_inner = type(model)
        self.model_graph.add_node(
            model.name,
            fit_time=model.fit_time,
            predict_time=model.predict_time,
            val_score=model.val_score,
            path=model.path,
            type=type(model),  # Outer type, can be BaggedEnsemble, StackEnsemble (Type that is able to load the model)
            type_inner=type_inner,  # Inner type, if Ensemble then it is the type of the inner model (May not be able to load with this type)
            can_infer=model.can_infer(),
            can_fit=model.can_fit(),
            is_valid=model.is_valid(),
            stack_name=stack_name,
            level=level,
            **model._fit_metadata,
        )
        if isinstance(model, StackerEnsembleModel):
            prior_models = self.get_model_names()
            # TODO: raise exception if no base models and level != 1?
            for stack_column_prefix in model.stack_column_prefix_lst:
                base_model_name = model.stack_column_prefix_to_model_map[stack_column_prefix]
                if base_model_name not in prior_models:
                    raise AssertionError(f"Model '{model.name}' depends on model '{base_model_name}', but '{base_model_name}' is not registered as a trained model! Valid models: {prior_models}")
                elif level <= self.model_graph.nodes[base_model_name]['level']:
                    raise AssertionError(f"Model '{model.name}' depends on model '{base_model_name}', but '{base_model_name}' is not in a lower stack level. ('{model.name}' level: {level}, '{base_model_name}' level: {self.model_graph.nodes[base_model_name]['level']})")
                self.model_graph.add_edge(base_model_name, model.name)
        if self.low_memory:
            del model
        return True

    # TODO: Split this to avoid confusion, HPO should go elsewhere?
    def _train_single_full(self, X, y, model: AbstractModel, X_unlabeled=None, X_val=None, y_val=None, feature_prune=False, hyperparameter_tune_kwargs=None,
                           stack_name='core', k_fold=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=1, time_limit=None, fit_kwargs=None, **kwargs) -> List[str]:
        """
        Trains a model, with the potential to train multiple versions of this model with hyperparameter tuning and feature pruning.
        Returns a list of successfully trained and saved model names.
        Models trained from this method will be accessible in this Trainer.
        """
        if k_fold is None:
            k_fold = self.k_fold
        if n_repeats is None:
            n_repeats = self.n_repeats
        if fit_kwargs is None:
            fit_kwargs = dict()
        model_fit_kwargs = dict(
            time_limit=time_limit,
            verbosity=self.verbosity,
        )
        model_fit_kwargs.update(fit_kwargs)
        if self.sample_weight is not None:
            X, w_train = extract_column(X, self.sample_weight)
            if w_train is not None:  # may be None for ensemble
                # TODO: consider moving weight normalization into AbstractModel.fit()
                model_fit_kwargs['sample_weight'] = w_train.values/w_train.mean()  # normalization can affect gradient algorithms like boosting
            if X_val is not None:
                X_val, w_val = extract_column(X_val, self.sample_weight)
                if self.weight_evaluation and w_val is not None:  # ignore validation sample weights unless weight_evaluation specified
                    model_fit_kwargs['sample_weight_val'] = w_val.values/w_val.mean()
            ens_sample_weight = kwargs.get('ens_sample_weight', None)
            if ens_sample_weight is not None:
                model_fit_kwargs['sample_weight'] = ens_sample_weight  # sample weights to use for weighted ensemble only
        if self._groups is not None and 'groups' not in model_fit_kwargs:
            if k_fold == self.k_fold:  # don't do this on refit full
                model_fit_kwargs['groups'] = self._groups

        #######################
        # FIXME: This section is a hack, compute genuine feature_metadata for each stack level instead
        #  Don't do this here, do this upstream so it isn't recomputed for each model
        #  Add feature_metadata to model_fit_kwargs
        # FIXME: Sample weight `extract_column` is a hack, have to compute feature_metadata here because sample weight column could be in X upstream, extract sample weight column upstream instead.
        # FIXME: This doesn't assign proper special types to stack features, relying on a hack in StackerEnsembleModel to assign S_STACK to feature metadata, don't do this.
        #  Remove hack in StackerEnsembleModel
        feature_metadata = self.feature_metadata
        features_base = self.feature_metadata.get_features()
        features_new = [feature for feature in X.columns if feature not in features_base]
        if features_new:
            feature_metadata_new = FeatureMetadata.from_df(X[features_new])
            feature_metadata = feature_metadata.join_metadata(feature_metadata_new).keep_features(list(X.columns))
        model_fit_kwargs['feature_metadata'] = feature_metadata
        #######################

        if hyperparameter_tune_kwargs:
            if n_repeat_start != 0:
                raise ValueError(f'n_repeat_start must be 0 to hyperparameter_tune, value = {n_repeat_start}')
            elif k_fold_start != 0:
                raise ValueError(f'k_fold_start must be 0 to hyperparameter_tune, value = {k_fold_start}')
            if not isinstance(hyperparameter_tune_kwargs, tuple):
                num_trials = 1 if time_limit is None else 1000
                hyperparameter_tune_kwargs = scheduler_factory(hyperparameter_tune_kwargs, num_trials=num_trials, nthreads_per_trial='auto', ngpus_per_trial='auto')
            # hpo_models (dict): keys = model_names, values = model_paths
            logger.log(20, f'Hyperparameter tuning model: {model.name} ...')
            try:
                if isinstance(model, BaggedEnsembleModel):
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X=X, y=y, k_fold=k_fold, scheduler_options=hyperparameter_tune_kwargs, **model_fit_kwargs)
                else:
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X=X, y=y, X_val=X_val, y_val=y_val, scheduler_options=hyperparameter_tune_kwargs, **model_fit_kwargs)
            except Exception as err:
                logger.exception(f'Warning: Exception caused {model.name} to fail during hyperparameter tuning... Skipping this model.')
                logger.warning(err)
                del model
                model_names_trained = []
            else:
                # Commented out because it takes too much space (>>5 GB if run for an hour on a small-medium sized dataset)
                # self.hpo_results[model.name] = hpo_results
                model_names_trained = []
                self._extra_banned_names.add(model.name)
                for model_hpo_name, model_path in hpo_models.items():
                    model_hpo = self.load_model(model_hpo_name, path=model_path, model_type=type(model))
                    logger.log(20, f'Fitted model: {model_hpo.name} ...')
                    if self._add_model(model=model_hpo, stack_name=stack_name, level=level):
                        model_names_trained.append(model_hpo.name)
        else:
            if isinstance(model, BaggedEnsembleModel):
                model_fit_kwargs.update(dict(
                    k_fold=k_fold,
                    k_fold_start=k_fold_start,
                    k_fold_end=k_fold_end,
                    n_repeats=n_repeats,
                    n_repeat_start=n_repeat_start,
                    compute_base_preds=False,
                ))
            model_names_trained = self._train_and_save(X, y, model, X_val, y_val, X_unlabeled=X_unlabeled, stack_name=stack_name, level=level, **model_fit_kwargs)
        self.save()
        return model_names_trained

    # TODO: How to deal with models that fail during this? They have trained valid models before, but should we still use those models or remove the entire model? Currently we still use models.
    # TODO: Time allowance can be made better by only using time taken during final model training and not during HPO and feature pruning.
    # TODO: Time allowance not accurate if running from fit_continue
    # TODO: Remove level and stack_name arguments, can get them automatically
    # TODO: Make sure that pretraining on X_unlabeled only happens 1 time rather than every fold of bagging. (Do during pretrain API work?)
    def _train_multi_repeats(self, X, y, models: list, n_repeats, n_repeat_start=1, time_limit=None, time_limit_total_level=None, **kwargs) -> List[str]:
        """
        Fits bagged ensemble models with additional folds and/or bagged repeats.
        Models must have already been fit prior to entering this method.
        This method should only be called in self._train_multi
        Returns a list of successfully trained and saved model names.
        """
        if time_limit_total_level is None:
            time_limit_total_level = time_limit
        models_valid = models
        models_valid_next = []
        repeats_completed = 0
        time_start = time.time()
        for n in range(n_repeat_start, n_repeats):
            if not models_valid:
                break  # No models to repeat
            if time_limit is not None:
                time_start_repeat = time.time()
                time_left = time_limit - (time_start_repeat - time_start)
                if n == n_repeat_start:
                    time_required = time_limit_total_level * 0.575  # Require slightly over 50% to be safe
                else:
                    time_required = (time_start_repeat - time_start) / repeats_completed * (0.575/0.425)
                if time_left < time_required:
                    logger.log(15, 'Not enough time left to finish repeated k-fold bagging, stopping early ...')
                    break
            logger.log(20, f'Repeating k-fold bagging: {n+1}/{n_repeats}')
            for i, model in enumerate(models_valid):
                if not self.get_model_attribute(model=model, attribute='can_fit'):
                    if isinstance(model, str):
                        models_valid_next.append(model)
                    else:
                        models_valid_next.append(model.name)
                    continue

                if isinstance(model, str):
                    model = self.load_model(model)
                if not isinstance(model, BaggedEnsembleModel):
                    raise AssertionError(f'{model.name} must inherit from BaggedEnsembleModel to perform repeated k-fold bagging. Model type: {type(model).__name__}')
                if time_limit is None:
                    time_left = None
                else:
                    time_start_model = time.time()
                    time_left = time_limit - (time_start_model - time_start)

                models_valid_next += self._train_single_full(X=X, y=y, model=model, k_fold_start=0, k_fold_end=None, n_repeats=n + 1, n_repeat_start=n, time_limit=time_left, **kwargs)
            models_valid = copy.deepcopy(models_valid_next)
            models_valid_next = []
            repeats_completed += 1
        logger.log(20, f'Completed {n_repeat_start + repeats_completed}/{n_repeats} k-fold bagging repeats ...')
        return models_valid

    def _train_multi_initial(self, X, y, models: List[AbstractModel], k_fold, n_repeats, hyperparameter_tune_kwargs=None, feature_prune=False, time_limit=None, **kwargs) -> List[str]:
        """
        Fits models that have not previously been fit.
        This method should only be called in self._train_multi
        Returns a list of successfully trained and saved model names.
        """
        fit_args = dict(
            X=X,
            y=y,
            k_fold=k_fold,
        )
        fit_args.update(kwargs)
        hpo_enabled = False
        if hyperparameter_tune_kwargs:
            for key in hyperparameter_tune_kwargs:
                if hyperparameter_tune_kwargs[key] is not None:
                    hpo_enabled = True
                    break

        hpo_time_ratio = 0.9
        if hpo_enabled:
            time_split = True
        else:
            time_split = False
        if k_fold == 0:
            time_ratio = hpo_time_ratio if hpo_enabled else 1
            models = self._train_multi_fold(models=models, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, feature_prune=feature_prune, time_limit=time_limit, time_split=time_split, time_ratio=time_ratio, **fit_args)
        else:
            k_fold_start = 0
            if hpo_enabled or feature_prune:
                time_start = time.time()
                time_ratio = (1 / k_fold) * hpo_time_ratio
                models = self._train_multi_fold(models=models, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, feature_prune=feature_prune,
                                                k_fold_start=0, k_fold_end=1, n_repeats=n_repeats, n_repeat_start=0, time_limit=time_limit, time_split=time_split, time_ratio=time_ratio, **fit_args)
                k_fold_start = 1
                if time_limit is not None:
                    time_limit = time_limit - (time.time() - time_start)

            models = self._train_multi_fold(models=models, hyperparameter_tune_kwargs=None, feature_prune=False, k_fold_start=k_fold_start, k_fold_end=k_fold, n_repeats=n_repeats, n_repeat_start=0, time_limit=time_limit, **fit_args)

        return models

    # TODO: Ban KNN from being a Stacker model outside of aux. Will need to ensemble select on all stack layers ensemble selector to make it work
    # TODO: Robert dataset, LightGBM is super good but RF and KNN take all the time away from it on 1h despite being much worse
    # TODO: Add time_limit_per_model
    # TODO: Rename for v0.1
    def _train_multi_fold(self, X, y, models: List[AbstractModel], time_limit=None, time_split=False,
                          time_ratio=1, hyperparameter_tune_kwargs=None, **kwargs) -> List[str]:
        """
        Trains and saves a list of models sequentially.
        This method should only be called in self._train_multi_initial
        Returns a list of trained model names.
        """
        models_valid = []
        time_start = time.time()
        if time_limit is not None:
            time_limit = time_limit * time_ratio
        if time_limit is not None and len(models) > 0:
            time_limit_model_split = time_limit / len(models)
        else:
            time_limit_model_split = time_limit
        for i, model in enumerate(models):
            if isinstance(model, str):
                model = self.load_model(model)
            elif self.low_memory:
                model = copy.deepcopy(model)
            if hyperparameter_tune_kwargs is not None and isinstance(hyperparameter_tune_kwargs, dict):
                hyperparameter_tune_kwargs_model = hyperparameter_tune_kwargs.get(model.name, None)
            else:
                hyperparameter_tune_kwargs_model = None
            # TODO: Only update scores when finished, only update model as part of final models if finished!
            if time_split:
                time_left = time_limit_model_split
            else:
                if time_limit is None:
                    time_left = None
                else:
                    time_start_model = time.time()
                    time_left = time_limit - (time_start_model - time_start)
            model_name_trained_lst = self._train_single_full(X, y, model, time_limit=time_left,
                                                             hyperparameter_tune_kwargs=hyperparameter_tune_kwargs_model, **kwargs)

            if self.low_memory:
                del model
            models_valid += model_name_trained_lst

        return models_valid

    def _train_multi(self, X, y, models: List[AbstractModel], hyperparameter_tune_kwargs=None, feature_prune=False, k_fold=None, n_repeats=None, n_repeat_start=0, time_limit=None, **kwargs) -> List[str]:
        """
        Train a list of models using the same data.
        Assumes that input data has already been processed in the form the models will receive as input (including stack feature generation).
        Trained models are available in the trainer object.
        Note: Consider using public APIs instead of this.
        Returns a list of trained model names.
        """
        time_limit_total_level = time_limit
        if k_fold is None:
            k_fold = self.k_fold
        if n_repeats is None:
            n_repeats = self.n_repeats
        if (k_fold == 0) and (n_repeats != 1):
            raise ValueError(f'n_repeats must be 1 when k_fold is 0, values: ({n_repeats}, {k_fold})')
        if time_limit is None:
            n_repeats_initial = n_repeats
        else:
            n_repeats_initial = 1
        if n_repeat_start == 0:
            time_start = time.time()
            model_names_trained = self._train_multi_initial(X=X, y=y, models=models, k_fold=k_fold, n_repeats=n_repeats_initial, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, feature_prune=feature_prune,
                                                            time_limit=time_limit, **kwargs)
            n_repeat_start = n_repeats_initial
            if time_limit is not None:
                time_limit = time_limit - (time.time() - time_start)
        else:
            model_names_trained = models
        if (n_repeats > 1) and (n_repeat_start < n_repeats):
            model_names_trained = self._train_multi_repeats(X=X, y=y, models=model_names_trained,
                                                            k_fold=k_fold, n_repeats=n_repeats, n_repeat_start=n_repeat_start, time_limit=time_limit, time_limit_total_level=time_limit_total_level, **kwargs)
        return model_names_trained

    def _train_multi_and_ensemble(self, X, y, X_val, y_val, hyperparameters: dict = None, X_unlabeled=None, num_stack_levels=0, time_limit=None, groups=None, **kwargs) -> List[str]:
        """Identical to self.train_multi_levels, but also saves the data to disk. This should only ever be called once."""
        if self.save_data and not self.is_data_saved:
            self.save_X(X)
            self.save_y(y)
            if X_val is not None:
                self.save_X_val(X_val)
                if y_val is not None:
                    self.save_y_val(y_val)
            self.is_data_saved = True
        if self._groups is None:
            self._groups = groups
        self._num_rows_train = len(X)
        if X_val is not None:
            self._num_rows_train += len(X_val)
        self._num_cols_train = len(list(X.columns))
        model_names_fit = self.train_multi_levels(X, y, hyperparameters=hyperparameters, X_val=X_val, y_val=y_val,
                                                  X_unlabeled=X_unlabeled, level_start=1, level_end=num_stack_levels+1, time_limit=time_limit, **kwargs)
        if len(self.get_model_names()) == 0:
            raise ValueError('AutoGluon did not successfully train any models')
        return model_names_fit

    def _predict_model(self, X, model, model_pred_proba_dict=None):
        if isinstance(model, str):
            model = self.load_model(model)
        X = self.get_inputs_to_model(model=model, X=X, model_pred_proba_dict=model_pred_proba_dict, fit=False)
        y_pred = model.predict(X=X)
        if self._regress_preds_asprobas and model.problem_type == REGRESSION:  # Convert regression preds to classes (during distillation)
            if (len(y_pred.shape) > 1) and (y_pred.shape[1] > 1):
                problem_type = MULTICLASS
            else:
                problem_type = BINARY
            y_pred = get_pred_from_proba(y_pred_proba=y_pred, problem_type=problem_type)
        return y_pred

    def _predict_proba_model(self, X, model, model_pred_proba_dict=None):
        if isinstance(model, str):
            model = self.load_model(model)
        X = self.get_inputs_to_model(model=model, X=X, model_pred_proba_dict=model_pred_proba_dict, fit=False)
        return model.predict_proba(X=X)

    def _get_dummy_stacker(self, level: int, model_levels: dict, use_orig_features=True) -> StackerEnsembleModel:
        model_names = model_levels[level - 1]
        base_models_dict = {}
        for model_name in model_names:
            if model_name in self.models.keys():
                base_models_dict[model_name] = self.models[model_name]
        hyperparameters = dict(
            use_orig_features=use_orig_features,
            max_base_models_per_type=0,
            max_base_models=0,
        )
        dummy_stacker = StackerEnsembleModel(
            path='',
            name='',
            model_base=AbstractModel(
                path='',
                name='',
                problem_type=self.problem_type,
                eval_metric=self.eval_metric,
                hyperparameters={'ag_args_fit': {'quantile_levels': self.quantile_levels}}
            ),
            base_model_names=model_names,
            base_models_dict=base_models_dict,
            base_model_paths_dict=self.get_models_attribute_dict(attribute='path', models=model_names),
            base_model_types_dict=self.get_models_attribute_dict(attribute='type', models=model_names),
            hyperparameters=hyperparameters,
            random_state=level+self.random_state
        )
        dummy_stacker.initialize(num_classes=self.num_classes)
        return dummy_stacker

    # TODO: Enable raw=True for bagged models when X=None
    #  This is non-trivial to implement for multi-layer stacking ensembles on the OOF data.
    # TODO: Consider limiting X to 10k rows here instead of inside the model call
    def get_feature_importance(self, model=None, X=None, y=None, raw=True, **kwargs) -> pd.DataFrame:
        if model is None:
            model = self.model_best
        model: AbstractModel = self.load_model(model)
        if X is None and model.val_score is None:
            raise AssertionError(f'Model {model.name} is not valid for generating feature importances on original training data because no validation data was used during training, please specify new test data to compute feature importances.')

        if X is None:
            if isinstance(model, WeightedEnsembleModel):
                if self.bagged_mode:
                    if raw:
                        raise AssertionError('`feature_stage=\'transformed\'` feature importance on the original training data is not yet supported when bagging is enabled, please specify new test data to compute feature importances.')
                    X = None
                    is_oof = True
                else:
                    if raw:
                        X = self.load_X_val()
                    else:
                        X = None
                    is_oof = False
            elif isinstance(model, BaggedEnsembleModel):
                if raw:
                    raise AssertionError('`feature_stage=\'transformed\'` feature importance on the original training data is not yet supported when bagging is enabled, please specify new test data to compute feature importances.')
                X = self.load_X()
                X = self.get_inputs_to_model(model=model, X=X, fit=True)
                is_oof = True
            else:
                X = self.load_X_val()
                if not raw:
                    X = self.get_inputs_to_model(model=model, X=X, fit=False)
                is_oof = False
        else:
            is_oof = False
            if not raw:
                X = self.get_inputs_to_model(model=model, X=X, fit=False)

        if y is None and X is not None:
            if is_oof:
                y = self.load_y()
            else:
                y = self.load_y_val()

        if raw:
            return self._get_feature_importance_raw(X=X, y=y, model=model, **kwargs)
        else:
            if is_oof:
                kwargs['is_oof'] = is_oof
            return model.compute_feature_importance(X=X, y=y, **kwargs)

    # TODO: Can get feature importances of all children of model at no extra cost, requires scoring the values after predict_proba on each model
    #  Could solve by adding a self.score_all() function which takes model as input and also returns scores of all children models.
    #  This would be best solved after adding graph representation, it lives most naturally in AbstractModel
    # TODO: Can skip features which were pruned on all models that model depends on (Complex to implement, requires graph representation)
    # TODO: Note that raw importance will not equal non-raw importance for bagged models, even if raw features are identical to the model features.
    #  This is because for non-raw, we do an optimization where each fold model calls .compute_feature_importance(), and then the feature importances are averaged across the folds.
    #  This is different from raw, where the predictions of the folds are averaged and then feature importance is computed.
    #  Consider aligning these methods so they produce the same result.
    # The output of this function is identical to non-raw when model is level 1 and non-bagged
    def _get_feature_importance_raw(self, X, y, model, eval_metric=None, **kwargs) -> pd.DataFrame:
        if eval_metric is None:
            eval_metric = self.eval_metric
        if model is None:
            model = self.model_best
        if eval_metric.needs_pred:
            predict_func = self.predict
        else:
            predict_func = self.predict_proba
        model: AbstractModel = self.load_model(model)
        predict_func_kwargs = dict(model=model)
        return compute_permutation_feature_importance(
            X=X, y=y, predict_func=predict_func, predict_func_kwargs=predict_func_kwargs, eval_metric=eval_metric, **kwargs
        )

    def _get_models_load_info(self, model_names):
        model_names = copy.deepcopy(model_names)
        model_paths = self.get_models_attribute_dict(attribute='path', models=model_names)
        model_types = self.get_models_attribute_dict(attribute='type', models=model_names)
        return model_names, model_paths, model_types

    # Sums the attribute value across all models that the provided model depends on, including itself.
    # For instance, this function can return the expected total predict_time of a model.
    # attribute is the name of the desired attribute to be summed, or a dictionary of model name -> attribute value if the attribute is not present in the graph.
    def get_model_attribute_full(self, model, attribute, func=sum):
        base_model_set = self.get_minimum_model_set(model)
        if isinstance(attribute, dict):
            is_dict = True
        else:
            is_dict = False
        if len(base_model_set) == 1:
            if is_dict:
                return attribute[model]
            else:
                return self.model_graph.nodes[base_model_set[0]][attribute]
        # attribute_full = 0
        attribute_lst = []
        for base_model in base_model_set:
            if is_dict:
                attribute_base_model = attribute[base_model]
            else:
                attribute_base_model = self.model_graph.nodes[base_model][attribute]
            if attribute_base_model is None:
                return None
            attribute_lst.append(attribute_base_model)
            # attribute_full += attribute_base_model
        if attribute_lst:
            attribute_full = func(attribute_lst)
        else:
            attribute_full = 0
        return attribute_full

    # Returns dictionary of model name -> attribute value for the provided attribute
    def get_models_attribute_dict(self, attribute, models: list = None) -> dict:
        models_attribute_dict = nx.get_node_attributes(self.model_graph, attribute)
        if models is not None:
            model_names = []
            for model in models:
                if not isinstance(model, str):
                    model = model.name
                model_names.append(model)
            models_attribute_dict = {key: val for key, val in models_attribute_dict.items() if key in model_names}
        return models_attribute_dict

    # TODO: v0.1 Proper error catching
    # Returns attribute value for the given model
    def get_model_attribute(self, model, attribute: str):
        if not isinstance(model, str):
            model = model.name
        return self.model_graph.nodes[model][attribute]

    def set_model_attribute(self, model, attribute: str, val):
        if not isinstance(model, str):
            model = model.name
        self.model_graph.nodes[model][attribute] = val

    # Gets the minimum set of models that the provided model depends on, including itself
    # Returns a list of model names
    def get_minimum_model_set(self, model, include_self=True) -> list:
        if not isinstance(model, str):
            model = model.name
        minimum_model_set = list(nx.bfs_tree(self.model_graph, model, reverse=True))
        if not include_self:
            minimum_model_set = [m for m in minimum_model_set if m != model]
        return minimum_model_set

    # Gets the minimum set of models that the provided models depend on, including themselves
    # Returns a list of model names
    def get_minimum_models_set(self, models: list) -> list:
        models_set = set()
        for model in models:
            models_set = models_set.union(self.get_minimum_model_set(model))
        return list(models_set)

    # Gets the set of base models used directly by the provided model
    # Returns a list of model names
    def get_base_model_names(self, model) -> list:
        if not isinstance(model, str):
            model = model.name
        base_model_set = list(self.model_graph.predecessors(model))
        return base_model_set

    def _get_banned_model_names(self) -> list:
        """Gets all model names which would cause model files to be overwritten if a new model was trained with the name"""
        return self.get_model_names() + list(self._extra_banned_names)

    def leaderboard(self, extra_info=False):
        model_names = self.get_model_names()
        score_val = []
        fit_time_marginal = []
        pred_time_val_marginal = []
        stack_level = []
        fit_time = []
        pred_time_val = []
        can_infer = []
        fit_order = list(range(1, len(model_names)+1))
        score_val_dict = self.get_models_attribute_dict('val_score')
        fit_time_marginal_dict = self.get_models_attribute_dict('fit_time')
        predict_time_marginal_dict = self.get_models_attribute_dict('predict_time')
        for model_name in model_names:
            score_val.append(score_val_dict[model_name])
            fit_time_marginal.append(fit_time_marginal_dict[model_name])
            fit_time.append(self.get_model_attribute_full(model=model_name, attribute='fit_time'))
            pred_time_val_marginal.append(predict_time_marginal_dict[model_name])
            pred_time_val.append(self.get_model_attribute_full(model=model_name, attribute='predict_time'))
            stack_level.append(self.get_model_level(model_name))
            can_infer.append(self.model_graph.nodes[model_name]['can_infer'])

        model_info_dict = defaultdict(list)
        if extra_info:
            # TODO: feature_metadata
            # TODO: disk size
            # TODO: load time
            # TODO: Add persist_if_mem_safe() function to persist in memory all models if reasonable memory size (or a specific model+ancestors)
            # TODO: Add is_persisted() function to check which models are persisted in memory
            # TODO: package_dependencies, package_dependencies_full

            info = self.get_info(include_model_info=True)
            model_info = info['model_info']
            custom_model_info = {}
            for model_name in model_info:
                custom_info = {}
                bagged_info = model_info[model_name].get('bagged_info', {})
                custom_info['num_models'] = bagged_info.get('num_child_models', 1)
                custom_info['memory_size'] = bagged_info.get('max_memory_size', model_info[model_name]['memory_size'])
                custom_info['memory_size_min'] = bagged_info.get('min_memory_size', model_info[model_name]['memory_size'])
                custom_info['child_model_type'] = bagged_info.get('child_model_type', None)
                custom_info['child_hyperparameters'] = bagged_info.get('child_hyperparameters', None)
                custom_info['child_hyperparameters_fit'] = bagged_info.get('child_hyperparameters_fit', None)
                custom_info['child_ag_args_fit'] = bagged_info.get('child_ag_args_fit', None)
                custom_model_info[model_name] = custom_info

            model_info_keys = ['num_features', 'model_type', 'hyperparameters', 'hyperparameters_fit', 'ag_args_fit', 'features']
            model_info_sum_keys = []
            for key in model_info_keys:
                model_info_dict[key] = [model_info[model_name][key] for model_name in model_names]
                if key in model_info_sum_keys:
                    key_dict = {model_name: model_info[model_name][key] for model_name in model_names}
                    model_info_dict[key + '_full'] = [self.get_model_attribute_full(model=model_name, attribute=key_dict) for model_name in model_names]

            model_info_keys = ['num_models', 'memory_size', 'memory_size_min', 'child_model_type', 'child_hyperparameters', 'child_hyperparameters_fit', 'child_ag_args_fit']
            model_info_full_keys = {'memory_size': [('memory_size_w_ancestors', sum)], 'memory_size_min': [('memory_size_min_w_ancestors', max)], 'num_models': [('num_models_w_ancestors', sum)]}
            for key in model_info_keys:
                model_info_dict[key] = [custom_model_info[model_name][key] for model_name in model_names]
                if key in model_info_full_keys:
                    key_dict = {model_name: custom_model_info[model_name][key] for model_name in model_names}
                    for column_name, func in model_info_full_keys[key]:
                        model_info_dict[column_name] = [self.get_model_attribute_full(model=model_name, attribute=key_dict, func=func) for model_name in model_names]

            ancestors = [list(nx.dag.ancestors(self.model_graph, model_name)) for model_name in model_names]
            descendants = [list(nx.dag.descendants(self.model_graph, model_name)) for model_name in model_names]

            model_info_dict['num_ancestors'] = [len(ancestor_lst) for ancestor_lst in ancestors]
            model_info_dict['num_descendants'] = [len(descendant_lst) for descendant_lst in descendants]
            model_info_dict['ancestors'] = ancestors
            model_info_dict['descendants'] = descendants

        df = pd.DataFrame(data={
            'model': model_names,
            'score_val': score_val,
            'pred_time_val': pred_time_val,
            'fit_time': fit_time,
            'pred_time_val_marginal': pred_time_val_marginal,
            'fit_time_marginal': fit_time_marginal,
            'stack_level': stack_level,
            'can_infer': can_infer,
            'fit_order': fit_order,
            **model_info_dict,
        })
        df_sorted = df.sort_values(by=['score_val', 'pred_time_val', 'model'], ascending=[False, True, False]).reset_index(drop=True)

        df_columns_lst = df_sorted.columns.tolist()
        explicit_order = [
            'model',
            'score_val',
            'pred_time_val',
            'fit_time',
            'pred_time_val_marginal',
            'fit_time_marginal',
            'stack_level',
            'can_infer',
            'fit_order',
            'num_features',
            'num_models',
            'num_models_w_ancestors',
            'memory_size',
            'memory_size_w_ancestors',
            'memory_size_min',
            'memory_size_min_w_ancestors',
            'num_ancestors',
            'num_descendants',
            'model_type',
            'child_model_type'
        ]
        explicit_order = [column for column in explicit_order if column in df_columns_lst]
        df_columns_other = [column for column in df_columns_lst if column not in explicit_order]
        df_columns_new = explicit_order + df_columns_other
        df_sorted = df_sorted[df_columns_new]

        return df_sorted

    def get_info(self, include_model_info=False) -> dict:
        num_models_trained = len(self.get_model_names())
        if self.model_best is not None:
            best_model = self.model_best
        else:
            try:
                best_model = self.get_model_best()
            except AssertionError:
                best_model = None
        if best_model is not None:
            best_model_score_val = self.get_model_attribute(model=best_model, attribute='val_score')
            best_model_stack_level = self.get_model_level(best_model)
        else:
            best_model_score_val = None
            best_model_stack_level = None
        # fit_time = None
        num_bag_folds = self.k_fold
        max_core_stack_level = self.get_max_level('core')
        max_stack_level = self.get_max_level()

        problem_type = self.problem_type
        eval_metric = self.eval_metric.name
        time_train_start = self._time_train_start
        num_rows_train = self._num_rows_train
        num_cols_train = self._num_cols_train
        num_classes = self.num_classes
        # TODO:
        #  Disk size of models
        #  Raw feature count
        #  HPO time
        #  Bag time
        #  Feature prune time
        #  Exception count / models failed count
        #  True model count (models * kfold)
        #  AutoGluon version fit on
        #  Max memory usage
        #  CPU count used / GPU count used

        info = {
            'time_train_start': time_train_start,
            'num_rows_train': num_rows_train,
            'num_cols_train': num_cols_train,
            'num_classes': num_classes,
            'problem_type': problem_type,
            'eval_metric': eval_metric,
            'best_model': best_model,
            'best_model_score_val': best_model_score_val,
            'best_model_stack_level': best_model_stack_level,
            'num_models_trained': num_models_trained,
            'num_bag_folds': num_bag_folds,
            'max_stack_level': max_stack_level,
            'max_core_stack_level': max_core_stack_level,
        }

        if include_model_info:
            info['model_info'] = self.get_models_info()

        return info

    def get_models_info(self, models: List[str] = None) -> dict:
        if models is None:
            models = self.get_model_names()
        model_info_dict = dict()
        for model in models:
            if isinstance(model, str):
                if model in self.models.keys():
                    model = self.models[model]
            if isinstance(model, str):
                model_type = self.get_model_attribute(model=model, attribute='type')
                model_path = self.get_model_attribute(model=model, attribute='path')
                model_info_dict[model] = model_type.load_info(path=model_path)
            else:
                model_info_dict[model.name] = model.get_info()
        return model_info_dict

    def reduce_memory_size(self, remove_data=True, remove_fit_stack=False, remove_fit=True, remove_info=False, requires_save=True, reduce_children=False, **kwargs):
        if remove_data and self.is_data_saved:
            data_files = [
                self.path_data + 'X.pkl',
                self.path_data + 'X_val.pkl',
                self.path_data + 'y.pkl',
                self.path_data + 'y_val.pkl',
            ]
            for data_file in data_files:
                try:
                    os.remove(data_file)
                except FileNotFoundError:
                    pass
            if requires_save:
                self.is_data_saved = False
            try:
                os.rmdir(self.path_data)
            except OSError:
                pass
            try:
                os.rmdir(self.path_utils)
            except OSError:
                pass
        models = self.get_model_names()
        for model in models:
            model = self.load_model(model)
            model.reduce_memory_size(remove_fit_stack=remove_fit_stack, remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, reduce_children=reduce_children, **kwargs)
            if requires_save:
                self.save_model(model, reduce_memory=False)
        if requires_save:
            self.save()

    # TODO: Also enable deletion of models which didn't succeed in training (files may still be persisted)
    #  This includes the original HPO fold for stacking
    # Deletes specified models from trainer and from disk (if delete_from_disk=True).
    def delete_models(self, models_to_keep=None, models_to_delete=None, allow_delete_cascade=False, delete_from_disk=True, dry_run=True):
        if models_to_keep is not None and models_to_delete is not None:
            raise ValueError('Exactly one of [models_to_keep, models_to_delete] must be set.')
        if models_to_keep is not None:
            if not isinstance(models_to_keep, list):
                models_to_keep = [models_to_keep]
            minimum_model_set = set()
            for model in models_to_keep:
                minimum_model_set.update(self.get_minimum_model_set(model))
            minimum_model_set = list(minimum_model_set)
            models_to_remove = [model for model in self.get_model_names() if model not in minimum_model_set]
        elif models_to_delete is not None:
            if not isinstance(models_to_delete, list):
                models_to_delete = [models_to_delete]
            minimum_model_set = set(models_to_delete)
            minimum_model_set_orig = copy.deepcopy(minimum_model_set)
            for model in models_to_delete:
                minimum_model_set.update(nx.algorithms.dag.descendants(self.model_graph, model))
            if not allow_delete_cascade:
                if minimum_model_set != minimum_model_set_orig:
                    raise AssertionError('models_to_delete contains models which cause a delete cascade due to other models being dependent on them. Set allow_delete_cascade=True to enable the deletion.')
            minimum_model_set = list(minimum_model_set)
            models_to_remove = [model for model in self.get_model_names() if model in minimum_model_set]
        else:
            raise ValueError('Exactly one of [models_to_keep, models_to_delete] must be set.')

        if dry_run:
            logger.log(30, f'Dry run enabled, AutoGluon would have deleted the following models: {models_to_remove}')
            if delete_from_disk:
                for model in models_to_remove:
                    model = self.load_model(model)
                    logger.log(30, f'\tDirectory {model.path} would have been deleted.')
            logger.log(30, f'To perform the deletion, set dry_run=False')
            return

        if delete_from_disk:
            for model in models_to_remove:
                model = self.load_model(model)
                model.delete_from_disk()

        self.model_graph.remove_nodes_from(models_to_remove)
        for model in models_to_remove:
            if model in self.models:
                self.models.pop(model)

        models_kept = self.get_model_names()

        if self.model_best is not None and self.model_best not in models_kept:
            try:
                self.model_best = self.get_model_best()
            except AssertionError:
                self.model_best = None

        # TODO: Delete from all the other model dicts
        self.save()

    @classmethod
    def load(cls, path, reset_paths=False):
        load_path = path + cls.trainer_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
            obj.reset_paths = reset_paths
            return obj

    @classmethod
    def load_info(cls, path, reset_paths=False, load_model_if_required=True):
        load_path = path + cls.trainer_info_name
        try:
            return load_pkl.load(path=load_path)
        except:
            if load_model_if_required:
                trainer = cls.load(path=path, reset_paths=reset_paths)
                return trainer.get_info()
            else:
                raise

    def save_info(self, include_model_info=False):
        info = self.get_info(include_model_info=include_model_info)

        save_pkl.save(path=self.path + self.trainer_info_name, object=info)
        save_json.save(path=self.path + self.trainer_info_json_name, obj=info)
        return info

    def _process_hyperparameters(self, hyperparameters: dict) -> dict:
        return process_hyperparameters(hyperparameters=hyperparameters)

    def _get_full_model_val_score(self, model: str) -> float:
        model_full_dict_inverse = {full: orig for orig, full in self.model_full_dict.items()}
        model_performances = self.get_models_attribute_dict(attribute='val_score')

        normal_model = model_full_dict_inverse[model]
        if normal_model not in model_performances:
            # normal model is deleted
            if model not in self._model_full_dict_val_score:
                raise ValueError(f'_FULL model {model} had the model it was based on ({normal_model}) deleted, and the validation score was not stored.')
            val_score = self._model_full_dict_val_score[model]
        else:
            # normal model exists
            val_score = model_performances[normal_model]
        return val_score

    def distill(self, X=None, y=None, X_val=None, y_val=None, X_unlabeled=None,
                time_limit=None, hyperparameters=None, holdout_frac=None, verbosity=None,
                models_name_suffix=None, teacher=None, teacher_preds='soft',
                augmentation_data=None, augment_method='spunge', augment_args={'size_factor':5,'max_size':int(1e5)},
                augmented_sample_weight=1.0):
        """ Various distillation algorithms.
            Args:
                X, y: pd.DataFrame and pd.Series of training data.
                    If None, original training data used during predictor.fit() will be loaded.
                    This data is split into train/validation if X_val, y_val are None.
                X_val, y_val: pd.DataFrame and pd.Series of validation data.
                time_limit, hyperparameters, holdout_frac: defined as in predictor.fit()
                teacher (None or str):
                    If None, uses the model with the highest validation score as the teacher model, otherwise use the specified model name as the teacher.
                teacher_preds (None or str): If None, we only train with original labels (no data augmentation, overrides augment_method)
                    If 'hard', labels are hard teacher predictions given by: teacher.predict()
                    If 'soft', labels are soft teacher predictions given by: teacher.predict_proba()
                    Note: 'hard' and 'soft' are equivalent for regression problems.
                    If augment_method specified, teacher predictions are only used to label augmented data (training data keeps original labels).
                    To apply label-smoothing: teacher_preds='onehot' will use original training data labels converted to one-hots for multiclass (no data augmentation).  # TODO: expose smoothing-hyperparameter.
                models_name_suffix (str): Suffix to append to each student model's name, new names will look like: 'MODELNAME_dstl_SUFFIX'
                augmentation_data: pd.DataFrame of additional data to use as "augmented data" (does not contain labels).
                    When specified, augment_method, augment_args are ignored, and this is the only augmented data that is used (teacher_preds cannot be None).
                augment_method (None or str): specifies which augmentation strategy to utilize. Options: [None, 'spunge','munge']
                    If None, no augmentation gets applied.
                }
                augment_args (dict): args passed into the augmentation function corresponding to augment_method.
                augmented_sample_weight (float): Nonnegative value indicating how much to weight augmented samples. This is only considered if sample_weight was initially specified in Predictor.
        """
        if verbosity is None:
            verbosity = self.verbosity

        if teacher is None:
            teacher = self._get_best()

        hyperparameter_tune = False  # TODO: add as argument with scheduler options.
        if augmentation_data is not None and teacher_preds is None:
            raise ValueError("augmentation_data must be None if teacher_preds is None")

        logger.log(20, f"Distilling with teacher='{teacher}', teacher_preds={str(teacher_preds)}, augment_method={str(augment_method)} ...")
        if teacher not in self.get_model_names(can_infer=True):
            raise AssertionError(f"Teacher model '{teacher}' is not a valid teacher model! Either it does not exist or it cannot infer on new data.\n"
                                 f"Valid teacher models: {self.get_model_names(can_infer=True)}")
        if X is None:
            if y is not None:
                raise ValueError("X cannot be None when y specified.")
            X = self.load_X()
            X_val = self.load_X_val()

        if y is None:
            y = self.load_y()
            y_val = self.load_y_val()

        if X_val is None:
            if y_val is not None:
                raise ValueError("X_val cannot be None when y_val specified.")
            if holdout_frac is None:
                holdout_frac = default_holdout_frac(len(X), hyperparameter_tune)
            X, X_val, y, y_val = generate_train_test_split(X, y, problem_type=self.problem_type, test_size=holdout_frac)

        y_val_og = y_val.copy()
        og_bagged_mode = self.bagged_mode
        og_verbosity = self.verbosity
        self.bagged_mode = False  # turn off bagging
        self.verbosity = verbosity  # change verbosity for distillation

        if self.sample_weight is not None:
            X, w = extract_column(X, self.sample_weight)

        if teacher_preds is None or teacher_preds == 'onehot':
            augment_method = None
            logger.log(20, "Training students without a teacher model. Set teacher_preds = 'soft' or 'hard' to distill using the best AutoGluon predictor as teacher.")

        if teacher_preds in ['onehot','soft']:
            y = format_distillation_labels(y, self.problem_type, self.num_classes)
            y_val = format_distillation_labels(y_val, self.problem_type, self.num_classes)

        if augment_method is None and augmentation_data is None:
            if teacher_preds == 'hard':
                y_pred = pd.Series(self.predict(X, model=teacher))
                if (self.problem_type != REGRESSION) and (len(y_pred.unique()) < len(y.unique())):  # add missing labels
                    logger.log(15, "Adding missing labels to distillation dataset by including some real training examples")
                    indices_to_add = []
                    for clss in y.unique():
                        if clss not in y_pred.unique():
                            logger.log(15, f"Fetching a row with label={clss} from training data")
                            clss_index = y[y == clss].index[0]
                            indices_to_add.append(clss_index)
                    X_extra = X.loc[indices_to_add].copy()
                    y_extra = y.loc[indices_to_add].copy()  # these are actually real training examples
                    X = pd.concat([X, X_extra])
                    y_pred = pd.concat([y_pred, y_extra])
                    if self.sample_weight is not None:
                        w = pd.concat([w, w[indices_to_add]])
                y = y_pred
            elif teacher_preds == 'soft':
                y = self.predict_proba(X, model=teacher)
                if self.problem_type == MULTICLASS:
                    y = pd.DataFrame(y)
                else:
                    y = pd.Series(y)
        else:
            X_aug = augment_data(X=X, feature_metadata=self.feature_metadata,
                                 augmentation_data=augmentation_data, augment_method=augment_method, augment_args=augment_args)
            if len(X_aug) > 0:
                if teacher_preds == 'hard':
                    y_aug = pd.Series(self.predict(X_aug, model=teacher))
                elif teacher_preds == 'soft':
                    y_aug = self.predict_proba(X_aug, model=teacher)
                    if self.problem_type == MULTICLASS:
                        y_aug = pd.DataFrame(y_aug)
                    else:
                        y_aug = pd.Series(y_aug)
                else:
                    raise ValueError(f"Unknown teacher_preds specified: {teacher_preds}")

                X = pd.concat([X, X_aug])
                y = pd.concat([y, y_aug])
                if self.sample_weight is not None:
                     w = pd.concat([w, pd.Series([augmented_sample_weight]*len(X_aug))])

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        if self.sample_weight is not None:
            w.reset_index(drop=True, inplace=True)
            X[self.sample_weight] = w

        name_suffix = '_DSTL'  # all student model names contain this substring
        if models_name_suffix is not None:
            name_suffix = name_suffix + "_" + models_name_suffix

        if hyperparameters is None:
            hyperparameters = {'GBM': {}, 'CAT': {}, 'NN': {}, 'RF': {}}
        hyperparameters = self._process_hyperparameters(hyperparameters=hyperparameters)  # TODO: consider exposing ag_args_fit, excluded_model_types as distill() arguments.
        if teacher_preds is not None and teacher_preds != 'hard' and self.problem_type != REGRESSION:
            self._regress_preds_asprobas = True

        core_kwargs = {
            'stack_name': self.distill_stackname,
            'get_models_func': self.construct_model_templates_distillation,
        }
        aux_kwargs = {
            'get_models_func': self.construct_model_templates_distillation,
            'check_if_best': False,
        }

        # self.bagged_mode = True  # TODO: Add options for bagging
        models = self.train_multi_levels(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters,
            time_limit=time_limit,  # FIXME: Also limit augmentation time
            name_suffix=name_suffix,
            core_kwargs=core_kwargs,
            aux_kwargs=aux_kwargs,
        )

        distilled_model_names = []
        w_val = None
        if self.weight_evaluation:
            X_val, w_val = extract_column(X_val, self.sample_weight)
        for model_name in models:  # finally measure original metric on validation data and overwrite stored val_scores
            model_score = self.score(X_val, y_val_og, model=model_name, weights=w_val)
            model_obj = self.load_model(model_name)
            model_obj.val_score = model_score
            model_obj.save()  # TODO: consider omitting for sake of efficiency
            self.model_graph.nodes[model_name]['val_score'] = model_score
            distilled_model_names.append(model_name)
        leaderboard = self.leaderboard()
        logger.log(20, 'Distilled model leaderboard:')
        leaderboard_distilled = leaderboard[leaderboard['model'].isin(models)].reset_index(drop=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            logger.log(20, leaderboard_distilled)

        # reset trainer to old state before distill() was called:
        self.bagged_mode = og_bagged_mode  # TODO: Confirm if safe to train future models after training models in both bagged and non-bagged modes
        self.verbosity = og_verbosity
        return distilled_model_names

import copy, time, traceback, logging
import os
from typing import List
import networkx as nx
import numpy as np
import pandas as pd
import psutil
from collections import defaultdict

from ..constants import AG_ARGS, AG_ARGS_FIT, BINARY, MULTICLASS, REGRESSION, REFIT_FULL_NAME, REFIT_FULL_SUFFIX
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_json, save_pkl
from autogluon.core.utils.exceptions import TimeLimitExceeded, NotEnoughMemoryError, NoValidFeatures
from autogluon.core.utils import dd_list, shuffle_df_rows, default_holdout_frac
from ..utils import get_pred_from_proba, generate_train_test_split, infer_eval_metric
from ..models.abstract.abstract_model import AbstractModel
from ..metrics import log_loss, scorer_expects_y_pred
from ..models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from ..trainer.model_presets.presets_custom import get_preset_custom
from ..trainer.model_presets.presets_distill import get_preset_models_distillation
from ..models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from ..models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
from ..augmentation.distill_utils import format_distillation_labels, augment_data

logger = logging.getLogger(__name__)


# FIXME: Below is major defect!
#  Weird interaction for metrics like AUC during bagging.
#  If kfold = 5, scores are 0.9, 0.85, 0.8, 0.75, and 0.7, the score is not 0.8! It is much lower because probs are combined together and AUC is recalculated
#  Do we want this to happen? Should we calculate score by 5 separate scores and then averaging instead?

# TODO: Dynamic model loading for ensemble models during prediction, only load more models if prediction is uncertain. This dynamically reduces inference time.
# TODO: Try midstack Semi-Supervised. Just take final models and re-train them, use bagged preds for SS rows. This would be very cheap and easy to try.
class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'
    trainer_info_name = 'info.pkl'
    trainer_info_json_name = 'info.json'
    distill_stackname = 'distill'  # name of stack-level for distilled student models

    def __init__(self, path: str, problem_type: str, scheduler_options=None, eval_metric=None, stopping_metric=None,
                 num_classes=None, low_memory=False, feature_metadata=None, k_fold=0, n_repeats=1,
                 stack_ensemble_levels=0, save_data=False, save_bagged_folds=True, random_seed=0, verbosity=2):
        self.path = path
        self.problem_type = problem_type
        self.feature_metadata = feature_metadata
        self.save_data = save_data
        self.random_seed = random_seed  # Integer value added to the stack level to get the random_seed for kfold splits or the train/val split if bagging is disabled
        self.verbosity = verbosity
        if eval_metric is not None:
            self.eval_metric = eval_metric
        else:
            self.eval_metric = infer_eval_metric(problem_type=self.problem_type)

        # stopping_metric is used to early stop all models except for aux models.
        if stopping_metric is not None:
            self.stopping_metric = stopping_metric
        elif self.eval_metric.name == 'roc_auc':
            self.stopping_metric = log_loss
        else:
            self.stopping_metric = self.eval_metric

        self.eval_metric_expects_y_pred = scorer_expects_y_pred(scorer=self.eval_metric)
        logger.log(25, f"AutoGluon will gauge predictive performance using evaluation metric: '{self.eval_metric.name}'")
        if not self.eval_metric_expects_y_pred:
            logger.log(25, "\tThis metric expects predicted probabilities rather than predicted class labels, so you'll need to use predict_proba() instead of predict()")

        logger.log(20, "\tTo change this, specify the eval_metric argument of fit()")
        logger.log(25, f"AutoGluon will early stop models using evaluation metric: '{self.stopping_metric.name}'")
        self.num_classes = num_classes
        self.feature_prune = False  # will be set to True if feature-pruning is turned on.
        self.low_memory = low_memory
        self.bagged_mode = True if k_fold >= 2 else False
        if self.bagged_mode:
            self.k_fold = k_fold  # int number of folds to do model bagging, < 2 means disabled
            self.stack_ensemble_levels = stack_ensemble_levels
            self.stack_mode = True if self.stack_ensemble_levels >= 1 else False
            self.n_repeats = n_repeats
        else:
            self.k_fold = 0
            self.stack_ensemble_levels = 0
            self.stack_mode = False
            self.n_repeats = 1
        self.save_bagged_folds = save_bagged_folds

        self.hyperparameters = {}  # TODO: This is currently required for fetching stacking layer models. Consider incorporating more elegantly

        # self.models_level_all['core'][0] # Includes base models
        # self.models_level_all['core'][1] # Stacker level 1
        # self.models_level_all['aux1'][1] # Stacker level 1 aux models, such as weighted_ensemble
        # self.models_level_all['core'][2] # Stacker level 2
        # Keeps track of the intended use of models in the stack ensemble. When building complex stack ensembles over multiple stages, it can be useful to group related models in the same stack namespace.
        # This is not used during inference, only during fitting of new models as a convenient way to get base models for a new stack layer.
        self.models_level = defaultdict(dd_list)

        self.model_best = None

        self.models = {}  # Dict of model name -> model object. A key, value pair only exists if a model is persisted in memory.
        self.model_graph = nx.DiGraph()  # Directed Acyclic Graph (DAG) of model interactions. Describes how certain models depend on the predictions of certain other models. Contains numerous metadata regarding each model.
        self.model_full_dict = {}  # Dict of normal model -> FULL model. FULL models are produced by self.refit_single_full() and self.refit_ensemble_full().
        self.reset_paths = False

        self.hpo_results = {}  # Stores summary of HPO process
        # Scheduler attributes:
        if scheduler_options is not None:
            self._scheduler_func = scheduler_options[0]  # unpack tuple
            self._scheduler_options = scheduler_options[1]
        else:
            self._scheduler_func = None
            self._scheduler_options = None

        self._time_limit = None  # Internal float of the total time limit allowed for a given fit call. Used in logging statements.
        self._time_train_start = None  # Internal timestamp of the time training started for a given fit call. Used in logging statements.

        self._num_rows_train = None
        self._num_cols_train = None

        self.is_data_saved = False

        self._regress_preds_asprobas = False  # whether to treat regression predictions as class-probabilities (during distillation)

    # path_root is the directory containing learner.pkl
    @property
    def path_root(self):
        return self.path.rsplit(os.path.sep, maxsplit=2)[0] + os.path.sep

    @property
    def path_utils(self):
        return self.path_root + 'utils' + os.path.sep

    @property
    def path_data(self):
        return self.path_utils + 'data' + os.path.sep

    def load_X_train(self):
        path = self.path_data + 'X_train.pkl'
        return load_pkl.load(path=path)

    def load_X_val(self):
        path = self.path_data + 'X_val.pkl'
        return load_pkl.load(path=path)

    def load_y_train(self):
        path = self.path_data + 'y_train.pkl'
        return load_pkl.load(path=path)

    def load_y_val(self):
        path = self.path_data + 'y_val.pkl'
        return load_pkl.load(path=path)

    def save_X_train(self, X, verbose=True):
        path = self.path_data + 'X_train.pkl'
        save_pkl.save(path=path, object=X, verbose=verbose)

    def save_X_val(self, X, verbose=True):
        path = self.path_data + 'X_val.pkl'
        save_pkl.save(path=path, object=X, verbose=verbose)

    def save_y_train(self, y, verbose=True):
        path = self.path_data + 'y_train.pkl'
        save_pkl.save(path=path, object=y, verbose=verbose)

    def save_y_val(self, y, verbose=True):
        path = self.path_data + 'y_val.pkl'
        save_pkl.save(path=path, object=y, verbose=verbose)

    def get_model_names_all(self, can_infer=None):
        model_names_all = list(self.model_graph.nodes)
        # TODO: can_infer is technically more complicated, if an ancestor can't infer then the model can't infer.
        if can_infer is not None:
            node_attributes = nx.get_node_attributes(self.model_graph, 'can_infer')
            model_names_all = [model for model in model_names_all if node_attributes[model] == can_infer]
        return model_names_all

    def get_model_names(self, stack_name):
        model_names = []
        levels = np.sort(list(self.models_level[stack_name].keys()))
        for level in levels:
            model_names += self.models_level[stack_name][level]
        return model_names

    def get_max_level(self, stack_name: str):
        try:
            return sorted(list(self.models_level[stack_name].keys()))[-1]
        except IndexError:
            return -1

    def get_max_level_all(self):
        max_level = 0
        for stack_name in self.models_level.keys():
            max_level = max(max_level, self.get_max_level(stack_name))
        return max_level

    # TODO: Rename method for v0.1
    def get_models(self, hyperparameters: dict, **kwargs) -> List[AbstractModel]:
        """Constructs a list of unfit models based on the hyperparameters dict."""
        raise NotImplementedError

    def get_model_level(self, model_name):
        for stack_name in self.models_level.keys():
            for level in self.models_level[stack_name].keys():
                if model_name in self.models_level[stack_name][level]:
                    return level
        raise ValueError(f'Model {model_name} does not exist in trainer.')

    def set_contexts(self, path_context):
        self.path, model_paths = self.create_contexts(path_context)
        for model, path in model_paths.items():
            self.set_model_attribute(model=model, attribute='path', val=path)

    def create_contexts(self, path_context):
        path = path_context
        model_paths = self.get_models_attribute_dict(attribute='path')
        for model, prev_path in model_paths.items():
            model_local_path = prev_path.split(self.path, 1)[1]
            new_path = path + model_local_path
            model_paths[model] = new_path

        return path, model_paths

    # TODO: Rename to .fit in v0.1
    # TODO: Consider having AbstractTrainer inherit from AbstractModel in v0.1
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        raise NotImplementedError

    # TODO: v0.1 add invalid_model_names argument
    # TODO: make models accept dictionary of level -> list of models for more control of future fit calls.
    # TODO: Enable HPO on levels > 0
    # TODO: Enable feature prune on levels > 0
    # TODO: Remove name_suffix, hacked in
    # TODO: Enable easier re-mapping of trained models -> hyperparameters input (They don't share a key since name can change)
    def train_multi_levels(self, X_train, y_train, X_val, y_val, X_unlabeled=None, hyperparameters: dict = None, base_model_names: List[str] = None, hyperparameter_tune=False, feature_prune=False, core_kwargs: dict = None, level_start=0, level_end=0, stack_name_core='core', stack_name_aux='aux1', time_limit=None, name_suffix='') -> List[str]:
        """
        Trains a multi-layer stack ensemble using the input data on the hyperparameters dict input.
            hyperparameters is used to determine the models used in each stack layer.
        If continuing a stack ensemble with level_start>0, ensure that base_model_names is set to the appropriate base models that will be used by the level_start level models.
        Trains both core and aux models.
            core models are standard models which are fit on the data features. Core models will also use model predictions if base_model_names was specified or if level != 0.
            aux models are ensemble models which only use the predictions of core models as features. These models never use the original features.
        Returns a list of the model names that were trained from this method call, in order of fit.
        """
        self._time_limit = time_limit
        self._time_train_start = time.time()
        time_train_start = self._time_train_start
        if base_model_names is None:
            base_model_names = []
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        else:
            hyperparameters = self._process_hyperparameters(hyperparameters=hyperparameters)
        hyperparameters = copy.deepcopy(hyperparameters)

        # TODO: Hack
        if name_suffix:
            for key_outer in hyperparameters:
                for key_model_type in hyperparameters[key_outer]:
                    for model_hyperparams in hyperparameters[key_outer][key_model_type]:
                        if 'AG_args' in model_hyperparams:
                            model_hyperparams['AG_args']['name_suffix'] = model_hyperparams['AG_args'].get('name_suffix', '') + name_suffix
                        else:
                            model_hyperparams['AG_args'] = dict(name_suffix=name_suffix)

        model_names_fit = []
        for level in range(max(0, level_start), level_end + 1):
            if not base_model_names and level > 0:
                logger.log(30, f'Warning: Training models at stack level {level}, but no base models were specified.')
            elif base_model_names and level == 0:
                raise AssertionError(f'Stack level 0 models cannot have base models, but base_model_names={base_model_names}.')
            if time_limit is None:
                time_limit_core = None
                time_limit_aux = None
            else:
                time_train_level_start = time.time()
                time_limit_for_level = (time_limit - (time_train_level_start - time_train_start)) / (level_end + 1 - level)
                time_limit_core = time_limit_for_level
                time_limit_aux = max(time_limit_for_level * 0.1, min(time_limit, 360))  # Allows aux to go over time_limit, but only by a small amount
            if level != 0:
                if hyperparameter_tune:
                    logger.log(15, 'Warning: Hyperparameter tuning is not implemented for stack levels > 0.')
                hyperparameter_tune = False  # TODO: Enable HPO on levels > 0
                feature_prune = False  # TODO: Enable feature prune on levels > 0
            models = self.get_models(hyperparameters, hyperparameter_tune=hyperparameter_tune, level=level)
            base_model_names, aux_models = self.stack_new_level(
                X=X_train, y=y_train, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled,
                models=models, level=level, base_model_names=base_model_names,
                hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                stack_name_core=stack_name_core, stack_name_aux=stack_name_aux,
                core_kwargs=core_kwargs,
                time_limit_core=time_limit_core, time_limit_aux=time_limit_aux,
                name_suffix_aux=name_suffix,  # TODO: Remove name_suffix_aux, hack
            )
            model_names_fit += base_model_names + aux_models
        self._time_limit = None
        self.save()
        return model_names_fit

    # TODO: Remove name_suffix_aux, hacked in
    def stack_new_level(self, X, y, models: List[AbstractModel], X_val=None, y_val=None, X_unlabeled=None, level=0, base_model_names: List[str] = None, hyperparameter_tune=False, feature_prune=False, stack_name_core='core', stack_name_aux='aux1', core_kwargs: dict = None, time_limit_core=None, time_limit_aux=None, name_suffix_aux='') -> (List[str], List[str]):
        """
        Similar to calling self.stack_new_level_core, except auxiliary models will also be trained via a call to self.stack_new_level_aux, with the models trained from self.stack_new_level_core used as base models.
        """
        if core_kwargs is None:
            core_kwargs = {}
        core_models = self.stack_new_level_core(X=X, y=y, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled, models=models, level=level, base_model_names=base_model_names, stack_name=stack_name_core, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, time_limit=time_limit_core, **core_kwargs)
        aux_args = dict(base_model_names=core_models, level=level+1, stack_name=stack_name_aux, time_limit=time_limit_aux, name_suffix=name_suffix_aux)
        if self.bagged_mode:
            aux_models = self.stack_new_level_aux(X=X, y=y, **aux_args)
        else:
            aux_models = self.stack_new_level_aux(X=X_val, y=y_val, fit=False, **aux_args)
        return core_models, aux_models

    def stack_new_level_core(self, X, y, models: List[AbstractModel], X_val=None, y_val=None, X_unlabeled=None, level=0, base_model_names: List[str] = None, stack_name='core', k_fold=None, n_repeats=None, hyperparameter_tune=False, feature_prune=False, time_limit=None, save_bagged_folds=None, use_orig_features=True, stacker_type=StackerEnsembleModel) -> List[str]:
        """
        Trains all models using the data provided.
        If level > 0, then the models will use base model predictions as additional features.
            The base models used can be specified via base_model_names.
        If self.bagged_mode, then models will be trained as StackerEnsembleModels.
        The data provided in this method should not contain stack features, as they will be automatically generated if necessary.
        """
        if base_model_names is None:
            base_model_names = []
        if k_fold is None:
            k_fold = self.k_fold
        if n_repeats is None:
            n_repeats = self.n_repeats
        if save_bagged_folds is None:
            save_bagged_folds = self.save_bagged_folds

        if self.bagged_mode:
            if level == 0:
                (base_model_names, base_model_paths, base_model_types) = ([], {}, {})
            elif level > 0:
                base_model_names, base_model_paths, base_model_types = self._get_models_load_info(model_names=base_model_names)
                if len(base_model_names) == 0:
                    logger.log(20, 'No base models to train on, skipping stack level...')
                    return []
            else:
                raise AssertionError(f'Stack level cannot be negative! level = {level}')
            invalid_model_names = set(self.get_model_names_all())
            models_stack = []
            for model in models:
                # Ensure name is unique
                model_stack_name = f'{model.name}_STACKER_l{level}'
                num_increment = 2
                while model_stack_name in invalid_model_names:  # Ensure name is unique
                    model_stack_name = f'{model.name}_{num_increment}_STACKER_l{level}'
                    num_increment += 1
                invalid_model_names.add(model_stack_name)

                model_stack = stacker_type(
                    path=self.path, name=model_stack_name, model_base=model, base_model_names=base_model_names,
                    base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types, use_orig_features=use_orig_features,
                    num_classes=self.num_classes, save_bagged_folds=save_bagged_folds, random_state=level+self.random_seed
                )
                models_stack.append(model_stack)
            models = models_stack
        X_train_init = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=True)
        if X_val is not None:
            X_val = self.get_inputs_to_stacker(X_val, base_models=base_model_names, fit=False)

        return self._train_multi(X_train=X_train_init, y_train=y, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level=level, stack_name=stack_name, k_fold=k_fold, n_repeats=n_repeats, time_limit=time_limit)

    # TODO: Consider making level be auto-determined based off of max(base_model_levels)+1
    # TODO: Remove name_suffix, hacked in
    def stack_new_level_aux(self, X, y, base_model_names: List[str], level, fit=True, stack_name='aux1', time_limit=None, name_suffix='') -> List[str]:
        """
        Trains auxiliary models (currently a single weighted ensemble) using the provided base models.
        Level must be greater than the level of any of the base models.
        Auxiliary models never use the original features and only train with the predictions of other models as features.
        """
        X_train_stack_preds = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=fit)
        return self.generate_weighted_ensemble(X=X_train_stack_preds, y=y, level=level, base_model_names=base_model_names, k_fold=0, n_repeats=1, stack_name=stack_name, time_limit=time_limit, name_suffix=name_suffix)

    def predict(self, X, model=None):
        if model is not None:
            return self._predict_model(X, model)
        elif self.model_best is not None:
            return self._predict_model(X, self.model_best)
        else:
            model = self.get_model_best()
            return self._predict_model(X, model)

    def predict_proba(self, X, model=None):
        if model is not None:
            return self._predict_proba_model(X, model)
        elif self.model_best is not None:
            return self._predict_proba_model(X, self.model_best)
        else:
            model = self.get_model_best()
            return self._predict_proba_model(X, model)

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
        if model_level >= 1 and isinstance(model, StackerEnsembleModel):
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

    def score(self, X, y, model=None):
        if self.eval_metric_expects_y_pred:
            y_pred_ensemble = self.predict(X=X, model=model)
            return self.eval_metric(y, y_pred_ensemble)
        else:
            y_pred_proba_ensemble = self.predict_proba(X=X, model=model)
            return self.eval_metric(y, y_pred_proba_ensemble)

    def score_with_y_pred_proba(self, y, y_pred_proba):
        if self.eval_metric_expects_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return self.eval_metric(y, y_pred)
        else:
            return self.eval_metric(y, y_pred_proba)

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
        if not fit:
            model_pred_proba_dict = self.get_model_pred_proba_dict(X=X, models=base_models, model_pred_proba_dict=model_pred_proba_dict)
            model_pred_proba_list = [model_pred_proba_dict[model] for model in base_models]
        else:
            # TODO: After _get_inputs_to_stacker_legacy is removed, this if/else is not necessary, instead pass fit param to get_model_pred_proba_dict()
            model_pred_proba_list = None

        X_stacker_input = self._get_inputs_to_stacker_legacy(X=X, level_start=0, level_end=1, model_levels={0: base_models}, y_pred_probas=model_pred_proba_list, fit=fit)
        if not use_orig_features:
            X_stacker_input = X_stacker_input.drop(columns=X.columns)
        return X_stacker_input

    # TODO: Legacy code, still used during training because it is technically slightly faster and more memory efficient than get_model_pred_proba_dict()
    #  Remove in future as it limits flexibility in stacker inputs during training
    def _get_inputs_to_stacker_legacy(self, X, level_start, level_end, model_levels, y_pred_probas=None, fit=False):
        if level_start > level_end:
            raise AssertionError(f'level_start cannot be greater than level end: ({level_start}, {level_end})')
        if (level_start == 0) and (level_end == 0):
            return X
        if fit:
            if level_start >= 1:
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
            if dummy_stacker.use_orig_features:
                if level_start >= 1:
                    dummy_stacker_start = self._get_dummy_stacker(level=level_start, model_levels=model_levels, use_orig_features=True)
                    cols_to_drop = dummy_stacker_start.stack_columns
                    X = X.drop(cols_to_drop, axis=1)
                X = pd.concat([X_stacker, X], axis=1)
            else:
                X = X_stacker
        else:
            dummy_stackers = {}
            for level in range(level_start, level_end+1):
                if level >= 1:
                    dummy_stackers[level] = self._get_dummy_stacker(level=level, model_levels=model_levels, use_orig_features=True)
            for level in range(level_start, level_end):
                if level >= 1:
                    cols_to_drop = dummy_stackers[level].stack_columns
                else:
                    cols_to_drop = []
                X = dummy_stackers[level+1].preprocess(X=X, preprocess_nonadaptive=False, fit=False, compute_base_preds=True)
                if len(cols_to_drop) > 0:
                    X = X.drop(cols_to_drop, axis=1)
        return X

    # TODO: v0.1 this does not ensure unique model names (although the models they are refitting from should be unique, making this produce unique names as well)
    # You must have previously called fit() with cache_data=True
    # Fits _FULL versions of specified models, but does NOT link them (_FULL stackers will still use normal models as input)
    def refit_single_full(self, X=None, y=None, X_val=None, y_val=None, X_unlabeled=None, models=None):
        if X is None:
            X = self.load_X_train()
            if X_val is None and not self.bagged_mode:
                X_val = self.load_X_val()
        if y is None:
            y = self.load_y_train()
            if y_val is None and not self.bagged_mode:
                y_val = self.load_y_val()

        if X_val is not None and y_val is not None:
            X_full = pd.concat([X, X_val])
            y_full = pd.concat([y, y_val])
        else:
            X_full = X
            y_full = y

        if models is None:
            models = self.get_model_names_all()

        model_levels = defaultdict(dd_list)
        ignore_models = []
        ignore_stack_names = [REFIT_FULL_NAME]
        for stack_name in ignore_stack_names:
            ignore_models += self.get_model_names(stack_name)  # get_model_names returns [] if stack_name does not exist
        for model_name in models:
            if model_name in ignore_models:
                continue
            model_level = self.get_model_level(model_name)
            model_levels[REFIT_FULL_NAME][model_level] += [model_name]

        levels = sorted(model_levels[REFIT_FULL_NAME].keys())
        models_trained_full = []
        model_full_dict = {}
        for level in levels:
            models_level = model_levels[REFIT_FULL_NAME][level]
            for model in models_level:
                model = self.load_model(model)
                model_name = model.name
                model_full = model.convert_to_refitfull_template()
                # Mitigates situation where bagged models barely had enough memory and refit requires more. Worst case results in OOM, but this lowers chance of failure.
                model_full.params_aux['max_memory_usage_ratio'] = model_full.params_aux['max_memory_usage_ratio'] * 1.15
                # TODO: Do it for all models in the level at once to avoid repeated processing of data?
                stacker_type = type(model)
                if issubclass(stacker_type, WeightedEnsembleModel):
                    base_model_names = list(model.stack_column_prefix_to_model_map.values())
                    # TODO: Technically we don't need to re-train the weighted ensemble, we could just copy the original and re-use the weights.
                    if self.bagged_mode:
                        X_train_stack_preds = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=True, use_orig_features=False)
                        y_input = y
                    else:
                        X_train_stack_preds = self.get_inputs_to_stacker(X_val, base_models=base_model_names, fit=False, use_orig_features=False)  # TODO: May want to cache this during original fit, as we do with OOF preds
                        y_input = y_val

                    # TODO: Remove child_hyperparameters, make this cleaner
                    #  This fixes the following: Use the original weighted ensemble's iterations: Currently Dionis spends over 1hr training the refit weighted ensemble because it isn't time limited and goes to 100 iterations.
                    child_hyperparameters = copy.deepcopy(model_full.params)
                    child_hyperparameters[AG_ARGS_FIT] = copy.deepcopy(model_full.params_aux)
                    # TODO: stack_name=REFIT_FULL_NAME_AUX?
                    models_trained = self.generate_weighted_ensemble(X=X_train_stack_preds, y=y_input, level=level, stack_name=REFIT_FULL_NAME, k_fold=0, n_repeats=1, base_model_names=base_model_names, name_suffix=REFIT_FULL_SUFFIX, save_bagged_folds=True, check_if_best=False, child_hyperparameters=child_hyperparameters)
                    # TODO: Do the below more elegantly, ideally as a parameter to the trainer train function to disable recording scores/pred time.
                    for model_weighted_ensemble in models_trained:
                        model_loaded = self.load_model(model_weighted_ensemble)
                        model_loaded.val_score = None
                        model_loaded.predict_time = None
                        self.set_model_attribute(model=model_weighted_ensemble, attribute='val_score', val=None)
                        self.save_model(model_loaded)
                else:
                    base_model_names = self.get_minimum_model_set(model_name, include_self=False)
                    models_trained = self.stack_new_level_core(X=X_full, y=y_full, X_unlabeled=X_unlabeled, models=[model_full], base_model_names=base_model_names, level=level, stack_name=REFIT_FULL_NAME, hyperparameter_tune=False, feature_prune=False, k_fold=0, n_repeats=1, save_bagged_folds=True, stacker_type=stacker_type)
                if len(models_trained) == 1:
                    model_full_dict[model_name] = models_trained[0]
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
    def refit_ensemble_full(self, model='all'):
        if model == 'all':
            ensemble_set = self.get_model_names_all()
        else:
            if model == 'best':
                model = self.get_model_best()
            ensemble_set = self.get_minimum_model_set(model)
        existing_models = self.get_model_names_all()
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
        models = self.get_model_names_all(can_infer=can_infer)
        if not models:
            raise AssertionError('Trainer has no fit models that can infer.')
        model_performances = self.get_models_attribute_dict(attribute='val_score')
        perfs = [(m, model_performances[m]) for m in models if model_performances[m] is not None]
        if not perfs:
            model_full_dict_inverse = {full: orig for orig, full in self.model_full_dict.items()}
            models = [m for m in models if m in model_full_dict_inverse]
            perfs = [(m, model_performances[model_full_dict_inverse[m]]) for m in models if model_performances[model_full_dict_inverse[m]] is not None]
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

    def persist_models(self, model_names='all', with_ancestors=False, max_memory=None) -> list:
        if model_names == 'all':
            model_names = self.get_model_names_all()
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

    def generate_weighted_ensemble(self, X, y, level, base_model_names, k_fold=0, n_repeats=1, stack_name=None, hyperparameters=None, time_limit=None, name_suffix='', save_bagged_folds=None, check_if_best=True, child_hyperparameters=None):
        if save_bagged_folds is None:
            save_bagged_folds = self.save_bagged_folds
        if len(base_model_names) == 0:
            logger.log(20, 'No base models to train on, skipping weighted ensemble...')
            return []

        # TODO: Remove extra_params, currently a hack
        if child_hyperparameters is not None:
            extra_params = {'_tmp_greedy_hyperparameters': child_hyperparameters}
        else:
            extra_params = {}

        invalid_model_names = set(self.get_model_names_all())
        # Ensure name is unique
        model_stack_name = f'weighted_ensemble{name_suffix}_k{k_fold}_l{level}'
        num_increment = 2
        while model_stack_name in invalid_model_names:  # Ensure name is unique
            model_stack_name = f'weighted_ensemble{name_suffix}_{num_increment}_k{k_fold}_l{level}'
            num_increment += 1

        weighted_ensemble_model = WeightedEnsembleModel(
            path=self.path,
            name=model_stack_name,
            base_model_names=base_model_names,
            base_model_paths_dict=self.get_models_attribute_dict(attribute='path', models=base_model_names),
            base_model_types_dict=self.get_models_attribute_dict(attribute='type', models=base_model_names),
            base_model_types_inner_dict=self.get_models_attribute_dict(attribute='type_inner', models=base_model_names),
            base_model_performances_dict=self.get_models_attribute_dict(attribute='val_score', models=base_model_names),
            hyperparameters=hyperparameters, eval_metric=self.eval_metric, stopping_metric=self.eval_metric, num_classes=self.num_classes, save_bagged_folds=save_bagged_folds, random_state=level + self.random_seed,
            **extra_params
        )
        self._train_multi(X_train=X, y_train=y, X_val=None, y_val=None, X_unlabeled=None, models=[weighted_ensemble_model], k_fold=k_fold, n_repeats=n_repeats, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, level=level, time_limit=time_limit)
        if check_if_best and weighted_ensemble_model.name in self.get_model_names_all():
            if self.model_best is None:
                self.model_best = weighted_ensemble_model.name
            else:
                best_score = self.get_model_attribute(self.model_best, 'val_score')
                cur_score = self.get_model_attribute(weighted_ensemble_model.name, 'val_score')
                if cur_score > best_score:
                    # new best model
                    self.model_best = weighted_ensemble_model.name
        return [weighted_ensemble_model.name]

    def _train_single(self, X_train, y_train, model: AbstractModel, X_val=None, y_val=None, X_unlabeled=None, **model_fit_kwargs) -> AbstractModel:
        """
        Trains model but does not add the trained model to this Trainer.
        Returns trained model object.
        """
        if model.feature_metadata is None:
            model.feature_metadata = copy.deepcopy(self.feature_metadata)  # TODO: move this into model creation process?

        if isinstance(model, BaggedEnsembleModel):
            model.fit(X=X_train, y=y_train, X_unlabeled=X_unlabeled, **model_fit_kwargs)
        else:
            model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled, **model_fit_kwargs)
        return model

    def _train_and_save(self, X_train, y_train, model: AbstractModel, X_val=None, y_val=None, X_unlabeled=None, stack_name='core', level=0, **model_fit_kwargs) -> List[str]:
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
            if time_limit is not None:
                if time_limit <= 0:
                    logging.log(15, f'Skipping {model.name} due to lack of time remaining.')
                    return model_names_trained
                if self._time_limit is not None and self._time_train_start is not None:
                    time_left_total = self._time_limit - (fit_start_time - self._time_train_start)
                else:
                    time_left_total = time_limit
                logging.log(20, f'Fitting model: {model.name} ... Training model for up to {round(time_limit, 2)}s of the {round(time_left_total, 2)}s of remaining time.')
            else:
                logging.log(20, f'Fitting model: {model.name} ...')
            model = self._train_single(X_train, y_train, model, X_val, y_val, X_unlabeled=X_unlabeled, **model_fit_kwargs)
            fit_end_time = time.time()
            if isinstance(model, BaggedEnsembleModel):
                if model.bagged_mode or isinstance(model, WeightedEnsembleModel):
                    score = model.score_with_oof(y=y_train)
                else:
                    score = None
            else:
                if X_val is not None and y_val is not None:
                    score = model.score(X=X_val, y=y_val)
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
        except Exception as err:
            logger.exception(f'\tWarning: Exception caused {model.name} to fail during training... Skipping this model.')
            logger.warning(err)
            del model
        else:
            self._add_model(model=model, stack_name=stack_name, level=level)
            model_names_trained.append(model.name)
            if self.low_memory:
                del model
        return model_names_trained

    def _add_model(self, model: AbstractModel, stack_name: str = 'core', level: int = 0):
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
        level : int, default 0
            Stack level of the stack name to assign the model to. This is used for advanced functionality.
            The model's name is appended to self.models_level[stack_name][level]
            The model's base_models (if it has any) must all be a lower level than the model.
        """
        if model.val_score is not None and stack_name != self.distill_stackname:  # TODO: may want to avoid hard-coding logic into specific stack names
            logger.log(20, f'\t{round(model.val_score, 4)}\t = Validation {self.eval_metric.name} score')
        if model.fit_time is not None:
            logger.log(20, f'\t{round(model.fit_time, 2)}s\t = Training runtime')
        if model.predict_time is not None:
            logger.log(20, f'\t{round(model.predict_time, 2)}s\t = Validation runtime')
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
            is_valid=model.is_valid(),
        )
        if isinstance(model, StackerEnsembleModel):
            # TODO: raise exception if any base models are of equal or higher level?
            # TODO: raise exception if no base models and level != 0?
            for stack_column_prefix in model.stack_column_prefix_lst:
                base_model_name = model.stack_column_prefix_to_model_map[stack_column_prefix]
                self.model_graph.add_edge(base_model_name, model.name)
        stack_loc = self.models_level[stack_name]  # TODO: Consider removing, have _train_multi handle this
        if model.name not in stack_loc[level]:
            stack_loc[level].append(model.name)
        if self.low_memory:
            del model

    # TODO: Split this to avoid confusion, HPO should go elsewhere?
    def _train_single_full(self, X_train, y_train, model: AbstractModel, X_unlabeled=None, X_val=None, y_val=None, feature_prune=False,
                           hyperparameter_tune=False, stack_name='core', k_fold=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, time_limit=None) -> List[str]:
        """
        Trains a model, with the potential to train multiple versions of this model with hyperparameter tuning and feature pruning.
        Returns a list of successfully trained and saved model names.
        Models trained from this method will be accessible in this Trainer.
        """
        if k_fold is None:
            k_fold = self.k_fold
        if n_repeats is None:
            n_repeats = self.n_repeats
        if (n_repeat_start == 0) and (k_fold_start == 0):
            model.feature_metadata = copy.deepcopy(self.feature_metadata)  # TODO: Don't set feature_metadata here
        if feature_prune:
            if n_repeat_start != 0:
                raise ValueError(f'n_repeat_start must be 0 to feature_prune, value = {n_repeat_start}')
            elif k_fold_start != 0:
                raise ValueError(f'k_fold_start must be 0 to feature_prune, value = {k_fold_start}')
            self._autotune(X_train=X_train, X_holdout=X_val, y_train=y_train, y_holdout=y_val, model_base=model)  # TODO: Update to use CV instead of holdout
        if hyperparameter_tune:
            if self._scheduler_func is None or self._scheduler_options is None:
                raise ValueError('scheduler_options cannot be None when hyperparameter_tune = True')
            if n_repeat_start != 0:
                raise ValueError(f'n_repeat_start must be 0 to hyperparameter_tune, value = {n_repeat_start}')
            elif k_fold_start != 0:
                raise ValueError(f'k_fold_start must be 0 to hyperparameter_tune, value = {k_fold_start}')
            # hpo_models (dict): keys = model_names, values = model_paths
            try:
                if isinstance(model, BaggedEnsembleModel):
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X=X_train, y=y_train, k_fold=k_fold, scheduler_options=(self._scheduler_func, self._scheduler_options), verbosity=self.verbosity)
                else:
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, scheduler_options=(self._scheduler_func, self._scheduler_options), verbosity=self.verbosity)
            except Exception as err:
                if self.verbosity >= 1:
                    traceback.print_tb(err.__traceback__)
                logger.exception(f'Warning: Exception caused {model.name} to fail during hyperparameter tuning... Skipping this model.')
                logger.debug(err)
                del model
                model_names_trained = []
            else:
                self.hpo_results[model.name] = hpo_results
                model_names_trained = []
                for model_hpo_name, model_path in hpo_models.items():
                    model_hpo = self.load_model(model_hpo_name, path=model_path, model_type=type(model))
                    self._add_model(model=model_hpo, stack_name=stack_name, level=level)
                    model_names_trained.append(model_hpo.name)
        else:
            model_fit_kwargs = dict(
                time_limit=time_limit,
                verbosity=self.verbosity,
            )
            if isinstance(model, BaggedEnsembleModel):
                model_fit_kwargs.update(dict(
                    k_fold=k_fold,
                    k_fold_start=k_fold_start,
                    k_fold_end=k_fold_end,
                    n_repeats=n_repeats,
                    n_repeat_start=n_repeat_start,
                    compute_base_preds=False,
                ))
            if self._scheduler_options is not None:
                model_fit_kwargs.update({
                    'num_cpus': self._scheduler_options['resource']['num_cpus'],
                    'num_gpus': self._scheduler_options['resource']['num_gpus'],
                })  # Additional configurations for model.fit
            model_names_trained = self._train_and_save(X_train, y_train, model, X_val, y_val, X_unlabeled=X_unlabeled, stack_name=stack_name, level=level, **model_fit_kwargs)
        self.save()
        return model_names_trained

    # TODO: How to deal with models that fail during this? They have trained valid models before, but should we still use those models or remove the entire model? Currently we still use models.
    # TODO: Time allowance can be made better by only using time taken during final model training and not during HPO and feature pruning.
    # TODO: Time allowance not accurate if running from fit_continue
    # TODO: Remove level and stack_name arguments, can get them automatically
    # TODO: Make sure that pretraining on X_unlabeled only happens 1 time rather than every fold of bagging. (Do during pretrain API work?)
    def _train_multi_repeats(self, X_train, y_train, X_val, y_val, models: list, k_fold, n_repeats, n_repeat_start=1, stack_name='core', level=0, X_unlabeled=None, time_limit=None, time_limit_total_level=None) -> List[str]:
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
                if isinstance(model, str):
                    model = self.load_model(model)
                if not isinstance(model, BaggedEnsembleModel):
                    raise AssertionError(f'{model.name} must inherit from BaggedEnsembleModel to perform repeated k-fold bagging. Model type: {type(model).__name__}')
                if time_limit is None:
                    time_left = None
                else:
                    time_start_model = time.time()
                    time_left = time_limit - (time_start_model - time_start)

                models_valid_next += self._train_single_full(X_train=X_train, y_train=y_train, model=model, X_unlabeled=X_unlabeled, X_val=X_val, y_val=y_val, stack_name=stack_name, k_fold=k_fold, k_fold_start=0, k_fold_end=None, n_repeats=n + 1, n_repeat_start=n, level=level, time_limit=time_left)
            models_valid = copy.deepcopy(models_valid_next)
            models_valid_next = []
            repeats_completed += 1
        logger.log(20, f'Completed {n_repeat_start + repeats_completed}/{n_repeats} k-fold bagging repeats ...')
        return models_valid

    def _train_multi_initial(self, X_train, y_train, X_val, y_val, models: List[AbstractModel], k_fold, n_repeats, X_unlabeled=None, hyperparameter_tune=False, feature_prune=False, stack_name='core', level=0, time_limit=None) -> List[str]:
        """
        Fits models that have not previously been fit.
        This method should only be called in self._train_multi
        Returns a list of successfully trained and saved model names.
        """
        fit_args = dict(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_unlabeled=X_unlabeled,
            stack_name=stack_name,
            k_fold=k_fold,
            level=level,
        )

        if k_fold == 0:
            models = self._train_multi_fold(models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, time_limit=time_limit, **fit_args)
        else:
            k_fold_start = 0
            if hyperparameter_tune or feature_prune:
                time_start = time.time()
                models = self._train_multi_fold(models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                                                k_fold_start=0, k_fold_end=1, n_repeats=n_repeats, n_repeat_start=0, time_limit=time_limit, **fit_args)
                k_fold_start = 1
                if time_limit is not None:
                    time_limit = time_limit - (time.time() - time_start)

            models = self._train_multi_fold(models=models, hyperparameter_tune=False, feature_prune=False, k_fold_start=k_fold_start, k_fold_end=k_fold, n_repeats=n_repeats, n_repeat_start=0, time_limit=time_limit, **fit_args)

        unique_names = []

        stack_loc = self.models_level[stack_name]
        for item in stack_loc[level]:
            if item not in unique_names:
                unique_names.append(item)
        stack_loc[level] = unique_names  # make unique and preserve order
        return models

    # TODO: Ban KNN from being a Stacker model outside of aux. Will need to ensemble select on all stack layers ensemble selector to make it work
    # TODO: Robert dataset, LightGBM is super good but RF and KNN take all the time away from it on 1h despite being much worse
    # TODO: Add time_limit_per_model
    # TODO: Rename for v0.1
    def _train_multi_fold(self, X_train, y_train, models: List[AbstractModel], X_unlabeled=None, time_limit=None, **kwargs) -> List[str]:
        """
        Trains and saves a list of models sequentially.
        This method should only be called in self._train_multi_initial
        Returns a list of trained model names.
        """
        models_valid = []
        time_start = time.time()
        for i, model in enumerate(models):
            if isinstance(model, str):
                model = self.load_model(model)
            elif self.low_memory:
                model = copy.deepcopy(model)
            # TODO: Only update scores when finished, only update model as part of final models if finished!
            if time_limit is None:
                time_left = None
            else:
                time_start_model = time.time()
                time_left = time_limit - (time_start_model - time_start)
            model_name_trained_lst = self._train_single_full(X_train, y_train, model, X_unlabeled=X_unlabeled, time_limit=time_left, **kwargs)

            if self.low_memory:
                del model
            models_valid += model_name_trained_lst

        return models_valid

    def _train_multi(self, X_train, y_train, X_val, y_val, models: List[AbstractModel], X_unlabeled=None, hyperparameter_tune=False, feature_prune=False, stack_name='core', k_fold=None, n_repeats=None, n_repeat_start=0, level=0, time_limit=None) -> List[str]:
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
            model_names_trained = self._train_multi_initial(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled, models=models, k_fold=k_fold, n_repeats=n_repeats_initial, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                                                            stack_name=stack_name, level=level, time_limit=time_limit)
            n_repeat_start = n_repeats_initial
            if time_limit is not None:
                time_limit = time_limit - (time.time() - time_start)
        else:
            model_names_trained = models
        if (n_repeats > 1) and self.bagged_mode and (n_repeat_start < n_repeats):
            model_names_trained = self._train_multi_repeats(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled, models=model_names_trained,
                                                            k_fold=k_fold, n_repeats=n_repeats, n_repeat_start=n_repeat_start, stack_name=stack_name, level=level, time_limit=time_limit, time_limit_total_level=time_limit_total_level)
        return model_names_trained

    def _train_multi_and_ensemble(self, X_train, y_train, X_val, y_val, hyperparameters: dict = None, X_unlabeled=None, hyperparameter_tune=False, feature_prune=False, time_limit=None) -> List[str]:
        """Identical to self.train_multi_levels, but also saves the data to disk. This should only ever be called once."""
        if self.save_data and not self.is_data_saved:
            self.save_X_train(X_train)
            self.save_y_train(y_train)
            if X_val is not None:
                self.save_X_val(X_val)
                if y_val is not None:
                    self.save_y_val(y_val)
            self.is_data_saved = True

        self._num_rows_train = len(X_train)
        if X_val is not None:
            self._num_rows_train += len(X_val)
        self._num_cols_train = len(list(X_train.columns))
        model_names_fit = self.train_multi_levels(X_train, y_train, X_val, y_val, X_unlabeled=X_unlabeled, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level_start=0, level_end=self.stack_ensemble_levels, time_limit=time_limit)
        if len(self.get_model_names_all()) == 0:
            raise ValueError('AutoGluon did not successfully train any models')
        return model_names_fit

    # TODO: Refactor
    # FIXME: This will likely not work properly
    def _autotune(self, X_train, X_holdout, y_train, y_holdout, model_base: AbstractModel):
        model_base.feature_prune(X_train, X_holdout, y_train, y_holdout)

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

    def _get_dummy_stacker(self, level, model_levels, use_orig_features=True):
        model_names = model_levels[level - 1]
        base_models_dict = {}
        for model_name in model_names:
            if model_name in self.models.keys():
                base_models_dict[model_name] = self.models[model_name]
        dummy_stacker = StackerEnsembleModel(
            path='',
            name='',
            model_base=AbstractModel(path='', name='', problem_type=self.problem_type, eval_metric=self.eval_metric),
            base_model_names=model_names,
            base_models_dict=base_models_dict,
            base_model_paths_dict=self.get_models_attribute_dict(attribute='path', models=model_names),
            base_model_types_dict=self.get_models_attribute_dict(attribute='type', models=model_names),
            use_orig_features=use_orig_features, num_classes=self.num_classes, random_state=level+self.random_seed
        )
        return dummy_stacker

    # TODO: Enable raw=True for bagged models when X=None
    #  This is non-trivial to implement for multi-layer stacking ensembles on the OOF data.
    # TODO: Consider limiting X to 10k rows here instead of inside the model call
    def get_feature_importance(self, model=None, X=None, y=None, features=None, raw=True, subsample_size=1000, silent=False):
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
                X = self.load_X_train()
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
                y = self.load_y_train()
            else:
                y = self.load_y_val()

        if raw:
            feature_importance = self._get_feature_importance_raw(model=model, X=X, y=y, features_to_use=features, subsample_size=subsample_size, silent=silent)
        else:
            feature_importance = model.compute_feature_importance(X=X, y=y, features_to_use=features, subsample_size=subsample_size, is_oof=is_oof, silent=silent)
        return feature_importance

    # TODO: Can get feature importances of all children of model at no extra cost, requires scoring the values after predict_proba on each model
    #  Could solve by adding a self.score_all() function which takes model as input and also returns scores of all children models.
    #  This would be best solved after adding graph representation, it lives most naturally in AbstractModel
    # TODO: Can skip features which were pruned on all models that model depends on (Complex to implement, requires graph representation)
    # TODO: Note that raw importance will not equal non-raw importance for bagged models, even if raw features are identical to the model features.
    #  This is because for non-raw, we do an optimization where each fold model calls .compute_feature_importance(), and then the feature importances are averaged across the folds.
    #  This is different from raw, where the predictions of the folds are averaged and then feature importance is computed.
    #  Consider aligning these methods so they produce the same result.
    # The output of this function is identical to non-raw when model is level 0 and non-bagged
    def _get_feature_importance_raw(self, model, X, y, features_to_use=None, subsample_size=1000, transform_func=None, silent=False):
        time_start = time.time()
        if model is None:
            model = self.model_best
        model: AbstractModel = self.load_model(model)
        if features_to_use is None:
            features_to_use = list(X.columns)
        feature_count = len(features_to_use)

        if not silent:
            logger.log(20, f'Computing raw permutation importance for {feature_count} features on {model.name} ...')

        if (subsample_size is not None) and (len(X) > subsample_size):
            # Reset index to avoid error if duplicated indices.
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            X = X.sample(subsample_size, random_state=0)
            y = y.loc[X.index]

        time_start_score = time.time()
        if transform_func is None:
            score_baseline = self.score(X=X, y=y, model=model)
        else:
            X_transformed = transform_func(X)
            score_baseline = self.score(X=X_transformed, y=y, model=model)
        time_score = time.time() - time_start_score

        if not silent:
            time_estimated = (feature_count + 1) * time_score + time_start_score - time_start
            logger.log(20, f'\t{round(time_estimated, 2)}s\t= Expected runtime')

        X_shuffled = shuffle_df_rows(X=X, seed=0)

        # Assuming X_test or X_val
        # TODO: Can check multiple features at a time only if non-OOF
        permutation_importance_dict = dict()
        X_to_check = X.copy()
        last_processed = None
        for feature in features_to_use:
            if last_processed is not None:  # resetting original values
                X_to_check[last_processed] = X[last_processed].values
            X_to_check[feature] = X_shuffled[feature].values
            if transform_func is None:
                score_feature = self.score(X=X_to_check, y=y, model=model)
            else:
                X_to_check_transformed = transform_func(X_to_check)
                score_feature = self.score(X=X_to_check_transformed, y=y, model=model)
            score_diff = score_baseline - score_feature
            permutation_importance_dict[feature] = score_diff
            last_processed = feature
        feature_importances = pd.Series(permutation_importance_dict).sort_values(ascending=False)

        if not silent:
            logger.log(20, f'\t{round(time.time() - time_start, 2)}s\t= Actual runtime')

        return feature_importances

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

    def leaderboard(self, extra_info=False):
        model_names = self.get_model_names_all()
        score_val = []
        fit_time_marginal = []
        pred_time_val_marginal = []
        stack_level = []
        fit_time = []
        pred_time_val = []
        can_infer = []
        fit_order = list(range(1,len(model_names)+1))
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
                custom_info['child_AG_args_fit'] = bagged_info.get('child_AG_args_fit', None)
                custom_model_info[model_name] = custom_info

            model_info_keys = ['num_features', 'model_type', 'hyperparameters', 'hyperparameters_fit', 'AG_args_fit', 'features']
            model_info_sum_keys = []
            for key in model_info_keys:
                model_info_dict[key] = [model_info[model_name][key] for model_name in model_names]
                if key in model_info_sum_keys:
                    key_dict = {model_name: model_info[model_name][key] for model_name in model_names}
                    model_info_dict[key + '_full'] = [self.get_model_attribute_full(model=model_name, attribute=key_dict) for model_name in model_names]

            model_info_keys = ['num_models', 'memory_size', 'memory_size_min', 'child_model_type', 'child_hyperparameters', 'child_hyperparameters_fit', 'child_AG_args_fit']
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

    def get_info(self, include_model_info=False):
        num_models_trained = len(self.get_model_names_all())
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
        num_bagging_folds = self.k_fold
        max_core_stack_level = self.get_max_level('core')
        max_stack_level = self.get_max_level_all()

        problem_type = self.problem_type
        eval_metric = self.eval_metric.name
        stopping_metric = self.stopping_metric.name
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
            'stopping_metric': stopping_metric,
            'best_model': best_model,
            'best_model_score_val': best_model_score_val,
            'best_model_stack_level': best_model_stack_level,
            'num_models_trained': num_models_trained,
            'num_bagging_folds': num_bagging_folds,
            'max_stack_level': max_stack_level,
            'max_core_stack_level': max_core_stack_level,
            'model_stack_info': self.models_level.copy(),
        }

        if include_model_info:
            info['model_info'] = self.get_models_info()

        return info

    def get_models_info(self, models=None):
        if models is None:
            models = self.get_model_names_all()
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
                self.path_data + 'X_train.pkl',
                self.path_data + 'X_val.pkl',
                self.path_data + 'y_train.pkl',
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
        models = self.get_model_names_all()
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
            models_to_remove = [model for model in self.get_model_names_all() if model not in minimum_model_set]
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
            models_to_remove = [model for model in self.get_model_names_all() if model in minimum_model_set]
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

        models_kept = self.get_model_names_all()
        # TODO: Refactor this part, link models_level to model_graph
        for key in self.models_level:
            for level in self.models_level[key]:
                self.models_level[key][level] = [model for model in self.models_level[key][level] if model in models_kept]

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

    def _process_hyperparameters(self, hyperparameters: dict, ag_args_fit=None, excluded_model_types=None) -> dict:
        if ag_args_fit is None:
            ag_args_fit = {}
        if excluded_model_types is None:
            excluded_model_types = []
        if excluded_model_types:
            logger.log(20, f'Excluded Model Types: {excluded_model_types}')
        hyperparameters = copy.deepcopy(hyperparameters)
        hyperparameters_valid = dict()

        has_levels = False
        top_level_keys = hyperparameters.keys()
        for key in top_level_keys:
            if isinstance(key, int) or key == 'default':
                has_levels = True
        if not has_levels:
            hyperparameters = {'default': hyperparameters}
        top_level_keys = hyperparameters.keys()
        for key in top_level_keys:
            hyperparameters_valid[key] = {}
            for subkey in hyperparameters[key].keys():
                if subkey in excluded_model_types:
                    logger.log(20, f"\tFound '{subkey}' model in hyperparameters, but '{subkey}' is present in `excluded_model_types` and will be removed.")
                    continue  # Don't include excluded models
                if not isinstance(hyperparameters[key][subkey], list):
                    hyperparameters[key][subkey] = [hyperparameters[key][subkey]]
                models_expanded = []
                for i, model in enumerate(hyperparameters[key][subkey]):
                    if isinstance(model, str):
                        candidate_models = get_preset_custom(name=model, problem_type=self.problem_type, num_classes=self.num_classes)
                    else:
                        candidate_models = [model]
                    valid_models = []
                    for candidate in candidate_models:
                        is_valid = True
                        if AG_ARGS in candidate:
                            model_valid_problem_types = candidate[AG_ARGS].get('problem_types', None)
                            if model_valid_problem_types is not None:
                                if self.problem_type not in model_valid_problem_types:
                                    is_valid = False
                        if ag_args_fit:
                            model_ag_fit_args = candidate.get(AG_ARGS_FIT, {})
                            for ag_fit_key in ag_args_fit:
                                if ag_fit_key not in model_ag_fit_args:
                                    model_ag_fit_args[ag_fit_key] = ag_args_fit[ag_fit_key]
                            candidate[AG_ARGS_FIT] = model_ag_fit_args
                        if is_valid:
                            valid_models.append(candidate)
                    models_expanded += valid_models

                hyperparameters_valid[key][subkey] = models_expanded
        if 'default' not in hyperparameters_valid.keys():
            level_keys = [key for key in hyperparameters_valid.keys() if isinstance(key, int)]
            max_level_key = max(level_keys)
            hyperparameters_valid['default'] = copy.deepcopy(hyperparameters_valid[max_level_key])
        return hyperparameters_valid

    def distill(self, X_train=None, y_train=None, X_val=None, y_val=None, X_unlabeled=None,
                time_limits=None, hyperparameters=None, holdout_frac=None, verbosity=None,
                models_name_suffix=None, teacher_preds='soft',
                augmentation_data=None, augment_method='spunge', augment_args={'size_factor':5,'max_size':int(1e5)}):
        """ Various distillation algorithms.
            Args:
                X_train, y_train: pd.DataFrame and pd.Series of training data.
                    If None, original training data used during TabularPrediction.fit() will be loaded.
                    This data is split into train/validation if X_val, y_val are None.
                X_val, y_val: pd.DataFrame and pd.Series of validation data.
                time_limits, hyperparameters, holdout_frac: defined as in TabularPrediction.fit()
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
        """
        if verbosity is None:
            verbosity = self.verbosity

        hyperparameter_tune = False  # TODO: add as argument with scheduler options.
        if augmentation_data is not None and teacher_preds is None:
            raise ValueError("augmentation_data must be None if teacher_preds is None")

        logger.log(20, f"Distilling with teacher_preds={str(teacher_preds)}, augment_method={str(augment_method)} ...")
        if X_train is None:
            if y_train is not None:
                raise ValueError("X cannot be None when y specified.")
            X_train = self.load_X_train()
            if not self.bagged_mode:
                try:
                    X_val = self.load_X_val()
                except FileNotFoundError:
                    pass

        if y_train is None:
            y_train = self.load_y_train()
            if not self.bagged_mode:
                try:
                    y_val = self.load_y_val()
                except FileNotFoundError:
                    pass

        if X_val is None:
            if y_val is not None:
                raise ValueError("X_val cannot be None when y_val specified.")
            if holdout_frac is None:
                holdout_frac = default_holdout_frac(len(X_train), hyperparameter_tune)
            X_train, X_val, y_train, y_val = generate_train_test_split(X_train, y_train, problem_type=self.problem_type, test_size=holdout_frac)

        y_val_og = y_val.copy()
        og_bagged_mode = self.bagged_mode
        og_verbosity = self.verbosity
        self.bagged_mode = False  # turn off bagging
        self.verbosity = verbosity  # change verbosity for distillation

        if teacher_preds is None or teacher_preds == 'onehot':
            augment_method = None
            logger.log(20, "Training students without a teacher model. Set teacher_preds = 'soft' or 'hard' to distill using the best AutoGluon predictor as teacher.")

        if teacher_preds in ['onehot','soft']:
            y_train = format_distillation_labels(y_train, self.problem_type, self.num_classes)
            y_val = format_distillation_labels(y_val, self.problem_type, self.num_classes)

        if augment_method is None and augmentation_data is None:
            if teacher_preds == 'hard':
                y_pred = pd.Series(self.predict(X_train))
                if (self.problem_type != REGRESSION) and (len(y_pred.unique()) < len(y_train.unique())):  # add missing labels
                    logger.log(15, "Adding missing labels to distillation dataset by including some real training examples")
                    indices_to_add = []
                    for clss in y_train.unique():
                        if clss not in y_pred.unique():
                            logger.log(15, f"Fetching a row with label={clss} from training data")
                            clss_index = y_train[y_train == clss].index[0]
                            indices_to_add.append(clss_index)
                    X_extra = X_train.loc[indices_to_add].copy()
                    y_extra = y_train.loc[indices_to_add].copy()  # these are actually real training examples
                    X_train = pd.concat([X_train, X_extra])
                    y_pred = pd.concat([y_pred, y_extra])
                y_train = y_pred
            elif teacher_preds == 'soft':
                y_train = self.predict_proba(X_train)
                if self.problem_type == MULTICLASS:
                    y_train = pd.DataFrame(y_train)
                else:
                    y_train = pd.Series(y_train)
        else:
            X_aug = augment_data(X_train=X_train, feature_metadata=self.feature_metadata,
                                 augmentation_data=augmentation_data, augment_method=augment_method, augment_args=augment_args)
            if len(X_aug) > 0:
                if teacher_preds == 'hard':
                    y_aug = pd.Series(self.predict(X_aug))
                elif teacher_preds == 'soft':
                    y_aug = self.predict_proba(X_aug)
                    if self.problem_type == MULTICLASS:
                        y_aug = pd.DataFrame(y_aug)
                    else:
                        y_aug = pd.Series(y_aug)
                else:
                    raise ValueError(f"Unknown teacher_preds specified: {teacher_preds}")

                X_train = pd.concat([X_train, X_aug])
                y_train = pd.concat([y_train, y_aug])

        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        student_suffix = '_DSTL'  # all student model names contain this substring
        if models_name_suffix is not None:
            student_suffix = student_suffix + "_" + models_name_suffix

        if hyperparameters is None:
            hyperparameters = copy.deepcopy(self.hyperparameters)
            student_model_types = ['GBM','CAT','NN','RF']  # only model types considered for distillation
            default_level_key = 'default'
            if default_level_key in hyperparameters:
                hyperparameters[default_level_key] = {key: hyperparameters[default_level_key][key] for key in hyperparameters[default_level_key] if key in student_model_types}
            else:
                hyperparameters ={key: hyperparameters[key] for key in hyperparameters if key in student_model_types}
                if len(hyperparameters) == 0:
                    raise ValueError("Distillation not yet supported for fit() with per-stack level hyperparameters. "
                                     "Please either manually specify `hyperparameters` in `distill()` or call `fit()` again without per-level hyperparameters before distillation."
                                     "Also at least one of the following model-types must be present in hyperparameters: ['GBM','CAT','NN','RF']")
        else:
            hyperparameters = self._process_hyperparameters(hyperparameters=hyperparameters, ag_args_fit=None, excluded_model_types=None)  # TODO: consider exposing ag_args_fit, excluded_model_types as distill() arguments.
        if teacher_preds is None or teacher_preds == 'hard':
            models_distill = self.get_models(hyperparameters=hyperparameters, name_suffix=student_suffix)
        else:
            models_distill = get_preset_models_distillation(path=self.path, problem_type=self.problem_type,
                                eval_metric=self.eval_metric, stopping_metric=self.stopping_metric,
                                num_classes=self.num_classes, hyperparameters=hyperparameters, name_suffix=student_suffix, invalid_model_names=self.get_model_names_all())
            if self.problem_type != REGRESSION:
                self._regress_preds_asprobas = True

        time_train_start = time.time()
        distilled_model_names = []
        for model in models_distill:
            time_left = None
            if time_limits is not None:
                time_start_model = time.time()
                time_left = time_limits - (time_start_model - time_train_start)

            logger.log(15, f"Distilling student {str(model.name)} with teacher_preds={str(teacher_preds)}, augment_method={str(augment_method)}...")
            models = self._train_single_full(X_train=X_train, y_train=y_train, model=model, X_val=X_val, y_val=y_val, X_unlabeled=X_unlabeled,
                                             hyperparameter_tune=False, stack_name=self.distill_stackname, time_limit=time_left)
            for model_name in models:  # finally measure original metric on validation data and overwrite stored val_scores
                model_score = self.score(X_val, y_val_og, model=model_name)
                model_obj = self.load_model(model_name)
                model_obj.val_score = model_score
                model_obj.save()  # TODO: consider omitting for sake of efficiency
                self.model_graph.nodes[model_name]['val_score'] = model_score
                distilled_model_names.append(model_name)
                logger.log(20, '\t' + str(round(model_obj.val_score, 4)) + '\t = Validation ' + self.eval_metric.name + ' score')
        # reset trainer to old state before distill() was called:
        self.bagged_mode = og_bagged_mode  # TODO: Confirm if safe to train future models after training models in both bagged and non-bagged modes
        self.verbosity = og_verbosity
        return distilled_model_names

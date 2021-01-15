import copy
import logging
import math
import pprint
import time

import numpy as np
import pandas as pd

from autogluon.core.task.base import compile_scheduler_options_v2
from autogluon.core.task.base.base_task import schedulers
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.utils import setup_outputdir, setup_compute, setup_trial_limits, default_holdout_frac

from .dataset import TabularDataset
from .hyperparameter_configs import get_hyperparameter_config
from .predictor_legacy import TabularPredictorV1
from .presets_configs import set_presets, unpack
from .feature_generator_presets import get_default_feature_generator
from ...learner import AbstractLearner, DefaultLearner
from ...trainer import AbstractTrainer

logger = logging.getLogger()  # return root logger


# TODO: v0.1 Add logging comments that models are serialized on disk after fit
class TabularPredictor(TabularPredictorV1):
    """
    AutoGluon Task for predicting values in column of tabular dataset (classification or regression)
    """
    Dataset = TabularDataset
    predictor_file_name = 'predictor.pkl'

    # TODO: v0.1 add pip freeze + python version output after fit + log file, validate that same pip freeze on load as cached
    # TODO: v0.1 predictor.clone()
    # TODO: v0.1 add documentation to init
    def __init__(
            self,
            label,
            problem_type=None,
            eval_metric=None,
            path=None,
            verbosity=2,
            **kwargs
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self._validate_init_kwargs(kwargs)
        path = setup_outputdir(path)

        learner_type = kwargs.pop('learner_type', DefaultLearner)
        learner_kwargs = kwargs.pop('learner_kwargs', dict())  # TODO: id_columns -> ignored_columns +1

        self._learner: AbstractLearner = learner_type(path_context=path, label=label, feature_generator=None,
                                                      eval_metric=eval_metric, problem_type=problem_type, **learner_kwargs)
        self._learner_type = type(self._learner)
        self._trainer = None

    @property
    def path(self):
        return self._learner.path

    @unpack(set_presets)
    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            presets=None,
            hyperparameters=None,
            feature_metadata=None,
            **kwargs):
        """
        Fit models to predict a column of data table based on the other columns.

        # TODO: Move documentation from TabularPrediction.fit to here
        # TODO: Move num_cpus/num_gpus to ag_args_fit
        # TODO: consider adding kwarg option for data which has already been preprocessed by feature generator to skip feature generation.
        # TODO: Remove all `time_limits` in project, replace with `time_limit`
        # TODO: Add logging for which presets were used
        # TODO: TabularDataset 'file_path' make so it does not have to be named. Same with 'df'.
        # TODO: Resolve raw text feature usage in default feature generator

        """
        if self._learner.is_fit:
            raise AssertionError('Predictor is already fit! To fit additional models, refer to `predictor.fit_extra`, or create a new `Predictor`.')
        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_kwargs(kwargs)

        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)

        if verbosity >= 3:
            logger.log(20, '============ fit kwarg info ============')
            logger.log(20, 'User Specified kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs_orig)}')
            logger.log(20, 'Full kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs)}')
            logger.log(20, '========================================')

        holdout_frac = kwargs['holdout_frac']
        num_bag_folds = kwargs['num_bag_folds']
        num_bag_sets = kwargs['num_bag_sets']
        num_stack_levels = kwargs['num_stack_levels']
        auto_stack = kwargs['auto_stack']
        hyperparameter_tune_kwargs = kwargs['hyperparameter_tune_kwargs']
        num_cpus = kwargs['num_cpus']
        num_gpus = kwargs['num_gpus']
        feature_generator = kwargs['feature_generator']
        unlabeled_data = kwargs['unlabeled_data']
        save_bagged_folds = kwargs['save_bagged_folds']

        ag_args = kwargs['ag_args']
        ag_args_fit = kwargs['ag_args_fit']
        ag_args_ensemble = kwargs['ag_args_ensemble']
        excluded_model_types = kwargs['excluded_model_types']

        train_data, tuning_data, unlabeled_data = self._validate_fit_data(train_data=train_data, tuning_data=tuning_data, unlabeled_data=unlabeled_data)

        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        ###################################
        # FIXME: v0.1 This section is a hack
        feature_generator_init_kwargs = dict()
        if 'TEXT_NN_V1' in hyperparameters:
            feature_generator_init_kwargs['enable_raw_text_features'] = True
        else:
            for key in hyperparameters:
                if isinstance(key, int) or key == 'default':
                    if 'TEXT_NN_V1' in hyperparameters[key]:
                        feature_generator_init_kwargs['enable_raw_text_features'] = True
                        break
        ###################################

        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata, init_kwargs=feature_generator_init_kwargs)

        # Process kwargs to create trainer, schedulers, searchers:
        num_bag_folds, num_bag_sets, num_stack_levels = self._sanitize_stack_args(
            num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets, num_stack_levels=num_stack_levels,
            time_limit=time_limit, auto_stack=auto_stack, num_train_rows=len(train_data),
        )

        if hyperparameter_tune_kwargs is not None:
            scheduler_options = self._init_scheduler(hyperparameter_tune_kwargs, time_limit, hyperparameters, num_cpus, num_gpus, num_bag_folds, num_stack_levels)
        else:
            scheduler_options = None
        hyperparameter_tune = scheduler_options is not None
        if hyperparameter_tune:
            logger.log(30, 'Warning: hyperparameter tuning is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.')

        if holdout_frac is None:
            holdout_frac = default_holdout_frac(len(train_data), hyperparameter_tune)

        if ag_args_fit is None:
            ag_args_fit = dict()
        # TODO: v0.1: Update to be 'auto' or None by default to give full control to individual models.
        if 'num_cpus' not in ag_args_fit and num_cpus != 'auto':
            ag_args_fit['num_cpus'] = num_cpus
        if 'num_gpus' not in ag_args_fit and num_gpus != 'auto':
            ag_args_fit['num_gpus'] = num_gpus

        # TODO: v0.1: make core_kwargs a kwargs argument to predictor.fit, add aux_kwargs to predictor.fit
        core_kwargs = {'ag_args': ag_args, 'ag_args_ensemble': ag_args_ensemble, 'ag_args_fit': ag_args_fit, 'excluded_model_types': excluded_model_types}
        self._learner.fit(X=train_data, X_val=tuning_data, X_unlabeled=unlabeled_data,
                          hyperparameter_tune_kwargs=scheduler_options,
                          holdout_frac=holdout_frac, num_bagging_folds=num_bag_folds, num_bagging_sets=num_bag_sets, stack_ensemble_levels=num_stack_levels,
                          hyperparameters=hyperparameters, core_kwargs=core_kwargs,
                          time_limit=time_limit, save_bagged_folds=save_bagged_folds, verbosity=verbosity)
        self._set_post_fit_vars()

        self._post_fit(
            keep_only_best=kwargs['keep_only_best'],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs['set_best_to_refit_full'],
            save_space=kwargs['save_space'],
        )
        self.save()
        return self

    def _post_fit(self, keep_only_best=False, refit_full=False, set_best_to_refit_full=False, save_space=False):
        if refit_full is True:
            if keep_only_best is True:
                if set_best_to_refit_full is True:
                    refit_full = 'best'
                else:
                    logger.warning(f'refit_full was set to {refit_full}, but keep_only_best=True and set_best_to_refit_full=False. Disabling refit_full to avoid training models which would be automatically deleted.')
                    refit_full = False
            else:
                refit_full = 'all'

        if refit_full is not False:
            trainer_model_best = self._trainer.get_model_best()
            self.refit_full(model=refit_full)
            if set_best_to_refit_full:
                if trainer_model_best in self._trainer.model_full_dict.keys():
                    self._trainer.model_best = self._trainer.model_full_dict[trainer_model_best]
                    # Note: model_best will be overwritten if additional training is done with new models, since model_best will have validation score of None and any new model will have a better validation score.
                    # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                    self._trainer.save()
                else:
                    logger.warning(f'Best model ({trainer_model_best}) is not present in refit_full dictionary. Training may have failed on the refit model. AutoGluon will default to using {trainer_model_best} for predictions.')

        if keep_only_best:
            self.delete_models(models_to_keep='best', dry_run=False)

        if save_space:
            self.save_space()

    # TODO: Documentation
    # Enables extra fit calls after the original fit
    def fit_extra(
            self, hyperparameters, time_limit=None,
            base_model_names=None, fit_new_weighted_ensemble=True, relative_stack=True,  # kwargs
            # core_kwargs=None,
            aux_kwargs=None, **kwargs
    ):
        # TODO: Allow disable aux (default to disabled)
        time_start = time.time()

        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_extra_kwargs(kwargs)

        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)

        if verbosity >= 3:
            logger.log(20, '============ fit kwarg info ============')
            logger.log(20, 'User Specified kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs_orig)}')
            logger.log(20, 'Full kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs)}')
            logger.log(20, '========================================')

        # TODO: num_bag_sets
        num_stack_levels = kwargs['num_stack_levels']
        hyperparameter_tune_kwargs = kwargs['hyperparameter_tune_kwargs']
        num_cpus = kwargs['num_cpus']
        num_gpus = kwargs['num_gpus']
        # save_bagged_folds = kwargs['save_bagged_folds']  # TODO: Enable

        ag_args = kwargs['ag_args']
        ag_args_fit = kwargs['ag_args_fit']
        ag_args_ensemble = kwargs['ag_args_ensemble']
        excluded_model_types = kwargs['excluded_model_types']

        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        if num_stack_levels is None:
            hyperparameter_keys = list(hyperparameters.keys())
            highest_level = 0
            for key in hyperparameter_keys:
                if isinstance(key, int):
                    highest_level = max(key, highest_level)
            num_stack_levels = highest_level

        if hyperparameter_tune_kwargs is not None:
            scheduler_options = self._init_scheduler(hyperparameter_tune_kwargs, time_limit, hyperparameters, num_cpus, num_gpus, self._trainer.k_fold, num_stack_levels)
        else:
            scheduler_options = None
        hyperparameter_tune = scheduler_options is not None
        if hyperparameter_tune:
            raise ValueError('Hyperparameter Tuning is not allowed in `fit_extra`.')  # FIXME: Change this
            # logger.log(30, 'Warning: hyperparameter tuning is currently experimental and may cause the process to hang.')
        if ag_args_fit is None:
            ag_args_fit = dict()
        if 'num_cpus' not in ag_args_fit and num_cpus != 'auto':
            ag_args_fit['num_cpus'] = num_cpus
        if 'num_gpus' not in ag_args_fit and num_gpus != 'auto':
            ag_args_fit['num_gpus'] = num_gpus

        # TODO: v0.1: make core_kwargs a kwargs argument to predictor.fit, add aux_kwargs to predictor.fit
        core_kwargs = {'ag_args': ag_args, 'ag_args_ensemble': ag_args_ensemble, 'ag_args_fit': ag_args_fit, 'excluded_model_types': excluded_model_types}

        # TODO: Add special error message if called and training/val data was not cached.
        X_train, y_train, X_val, y_val = self._trainer.load_data()
        fit_models = self._trainer.train_multi_levels(
            X_train=X_train, y_train=y_train, hyperparameters=hyperparameters, X_val=X_val, y_val=y_val, base_model_names=base_model_names, time_limit=time_limit, relative_stack=relative_stack, level_end=num_stack_levels,
            core_kwargs=core_kwargs, aux_kwargs=aux_kwargs
        )

        if time_limit is not None:
            time_limit = time_limit - (time.time() - time_start)

        if fit_new_weighted_ensemble:
            if time_limit is not None:
                time_limit_weighted = max(time_limit, 60)
            else:
                time_limit_weighted = None
            fit_models += self.fit_weighted_ensemble(time_limit=time_limit_weighted)

        self._post_fit(
            keep_only_best=kwargs['keep_only_best'],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs['set_best_to_refit_full'],
            save_space=kwargs['save_space'],
        )
        self.save()
        return self

    # TODO: Move to generic, migrate all tasks to same kwargs logic
    def _init_scheduler(self, hyperparameter_tune_kwargs, time_limit, hyperparameters, num_cpus, num_gpus, num_bag_folds, num_stack_levels):
        num_cpus, num_gpus = setup_compute(num_cpus, num_gpus)  # TODO: use 'auto' downstream
        time_limit_hpo = time_limit
        if num_bag_folds >= 2 and (time_limit_hpo is not None):
            time_limit_hpo = time_limit_hpo / (1 + num_bag_folds * (1 + num_stack_levels))
        # FIXME: Incorrect if user specifies custom level-based hyperparameter config!
        time_limit_hpo, num_trials = setup_trial_limits(time_limit_hpo, None, hyperparameters)  # TODO: Move HPO time allocation to Trainer
        if time_limit is not None:
            time_limit_hpo = None

        if hyperparameter_tune_kwargs is not None and isinstance(hyperparameter_tune_kwargs, str):
            preset_dict = {
                'auto': {'searcher': 'random'},
                'grid': {'searcher': 'grid'},
                'random': {'searcher': 'random'},
                'bayesopt': {'searcher': 'bayesopt'},
                'skopt': {'searcher': 'skopt'},
                # Don't include hyperband and bayesopt hyperband at present
            }
            if hyperparameter_tune_kwargs not in preset_dict:
                raise ValueError(f'Invalid hyperparameter_tune_kwargs preset value "{hyperparameter_tune_kwargs}". Valid presets: {list(preset_dict.keys())}')
            hyperparameter_tune_kwargs = preset_dict[hyperparameter_tune_kwargs]

        # All models use the same scheduler:
        scheduler_options = compile_scheduler_options_v2(
            scheduler_options=hyperparameter_tune_kwargs,
            nthreads_per_trial=num_cpus,
            ngpus_per_trial=num_gpus,
            num_trials=num_trials,
            time_out=time_limit_hpo,
        )
        if scheduler_options is None:
            return None

        assert scheduler_options['searcher'] != 'bayesopt_hyperband', "searcher == 'bayesopt_hyperband' not yet supported"
        # TODO: Fix or remove in v0.1
        if scheduler_options.get('dist_ip_addrs', None):
            logger.log(30, 'Warning: dist_ip_addrs does not currently work for Tabular. Distributed instances will not be utilized.')

        if scheduler_options['num_trials'] == 1:
            logger.log(30, 'Warning: Specified num_trials == 1 or time_limit is too small for hyperparameter_tune, disabling HPO.')
            return None  # FIXME

        scheduler_cls = schedulers[scheduler_options['searcher'].lower()]
        if scheduler_options['time_out'] is None:
            scheduler_options.pop('time_out', None)
        scheduler_options = (scheduler_cls, scheduler_options)  # wrap into tuple
        return scheduler_options

    def _set_post_fit_vars(self, learner: AbstractLearner = None):
        if learner is not None:
            self._learner: AbstractLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._learner.persist_trainer(low_memory=True)
            self._trainer: AbstractTrainer = self._learner.load_trainer()  # Trainer object

    # TODO: Update and correct the logging message on loading directions
    def save(self):
        tmp_learner = self._learner
        tmp_trainer = self._trainer
        super().save()
        self._learner = None
        self._trainer = None
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)
        self._learner = tmp_learner
        self._trainer = tmp_trainer

    @classmethod
    def load(cls, path, verbosity=2):
        set_logger_verbosity(verbosity, logger=logger)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("output_directory cannot be None in load()")

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        predictor: TabularPredictor = load_pkl.load(path=path + cls.predictor_file_name)
        learner = predictor._learner_type.load(path)
        predictor._set_post_fit_vars(learner=learner)
        try:
            from ...version import __version__
            version_inference = __version__
        except:
            version_inference = None
        # TODO: v0.1 Move version var to predictor object in the case where learner does not exist
        try:
            version_fit = predictor._learner.version
        except:
            version_fit = None
        if version_fit is None:
            version_fit = 'Unknown (Likely <=0.0.11)'
        if version_inference != version_fit:
            logger.warning('')
            logger.warning('############################## WARNING ##############################')
            logger.warning('WARNING: AutoGluon version differs from the version used during the original model fit! This may lead to instability and it is highly recommended the model be loaded with the exact AutoGluon version it was fit with.')
            logger.warning(f'\tFit Version:     {version_fit}')
            logger.warning(f'\tCurrent Version: {version_inference}')
            logger.warning('############################## WARNING ##############################')
            logger.warning('')

        return predictor

    @classmethod
    def from_learner(cls, learner: AbstractLearner):
        predictor = cls(label=learner.label, output_directory=learner.path)
        predictor._set_post_fit_vars(learner=learner)
        return predictor

    @staticmethod
    def _validate_init_kwargs(kwargs):
        valid_kwargs = {
            'learner_type',
            'learner_kwargs',
        }
        invalid_keys = []
        for key in kwargs:
            if key not in valid_kwargs:
                invalid_keys.append(key)
        if invalid_keys:
            raise ValueError(f'Invalid kwargs passed: {invalid_keys}\nValid kwargs: {list(valid_kwargs)}')

    def _validate_fit_kwargs(self, kwargs):

        # TODO:
        #  Valid core_kwargs values:
        #  ag_args, ag_args_fit, ag_args_ensemble, save_bagged_folds, stack_name, ensemble_type, name_suffix, time_limit
        #  Valid aux_kwargs values:
        #  name_suffix, time_limit, stack_name, aux_hyperparameters, ag_args, ag_args_ensemble

        # TODO: Remove features from models option for fit_extra
        # TODO: Constructor?
        fit_kwargs_default = dict(
            # data split / ensemble architecture kwargs -> Don't nest but have nested documentation -> Actually do nesting
            holdout_frac=None,  # TODO: Potentially error if num_bag_folds is also specified
            num_bag_folds=None,  # TODO: Potentially move to fit_extra, raise exception if value too large / invalid in fit_extra.
            auto_stack=False,

            # other
            feature_generator="auto",
            unlabeled_data=None,
        )

        kwargs = self._validate_fit_extra_kwargs(kwargs, extra_valid_keys=list(fit_kwargs_default.keys()))

        kwargs_sanitized = fit_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        return kwargs_sanitized

    def _validate_fit_extra_kwargs(self, kwargs, extra_valid_keys=None):
        fit_extra_kwargs_default = dict(
            # data split / ensemble architecture kwargs -> Don't nest but have nested documentation -> Actually do nesting
            num_bag_sets=None,
            num_stack_levels=None,

            hyperparameter_tune_kwargs=None,

            # core_kwargs -> +1 nest
            ag_args=None,
            ag_args_fit=None,
            ag_args_ensemble=None,
            excluded_model_types=None,
            save_bagged_folds=True,  # TODO: Move to ag_args_ensemble

            # aux_kwargs -> +1 nest

            # post_fit_kwargs -> +1 nest
            set_best_to_refit_full=False,
            keep_only_best=False,
            save_space=False,
            refit_full=False,

            # move into ag_args_fit? +1
            num_cpus='auto',
            num_gpus='auto',

            # other
            verbosity=self.verbosity,
        )

        allowed_kwarg_names = list(fit_extra_kwargs_default.keys())
        if extra_valid_keys is not None:
            allowed_kwarg_names += extra_valid_keys
        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                raise ValueError("Unknown keyword argument specified: %s" % kwarg_name)

        kwargs_sanitized = fit_extra_kwargs_default.copy()
        kwargs_sanitized.update(kwargs)

        # Deepcopy args to avoid altering outer context
        deepcopy_args = ['ag_args', 'ag_args_fit', 'ag_args_ensemble', 'excluded_model_types']
        for deepcopy_arg in deepcopy_args:
            kwargs_sanitized[deepcopy_arg] = copy.deepcopy(kwargs_sanitized[deepcopy_arg])

        refit_full = kwargs_sanitized['refit_full']
        set_best_to_refit_full = kwargs_sanitized['set_best_to_refit_full']
        if refit_full and not self._learner.cache_data:
            raise ValueError('`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.')
        if set_best_to_refit_full and not refit_full:
            raise ValueError('`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`.')

        return kwargs_sanitized

    def _validate_fit_data(self, train_data, tuning_data=None, unlabeled_data=None):
        if isinstance(train_data, str):
            train_data = TabularDataset(file_path=train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(file_path=tuning_data)
        if unlabeled_data is not None and isinstance(unlabeled_data, str):
            unlabeled_data = TabularDataset(file_path=unlabeled_data)

        if len(set(train_data.columns)) < len(train_data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        if tuning_data is not None:
            train_features = np.array([column for column in train_data.columns if column != self.label_column])
            tuning_features = np.array([column for column in tuning_data.columns if column != self.label_column])
            if np.any(train_features != tuning_features):
                raise ValueError("Column names must match between training and tuning data")
        if unlabeled_data is not None:
            train_features = sorted(np.array([column for column in train_data.columns if column != self.label_column]))
            unlabeled_features = sorted(np.array([column for column in unlabeled_data.columns]))
            if np.any(train_features != unlabeled_features):
                raise ValueError("Column names must match between training and unlabeled data.\n"
                                 "Unlabeled data must have not the label column specified in it.\n")
        return train_data, tuning_data, unlabeled_data

    def _set_feature_generator(self, feature_generator='auto', feature_metadata=None, init_kwargs=None):
        if self._learner.feature_generator is not None:
            if isinstance(feature_generator, str) and feature_generator == 'auto':
                feature_generator = self._learner.feature_generator
            else:
                raise AssertionError('FeatureGenerator already exists!')
        self._learner.feature_generator = get_default_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata, init_kwargs=init_kwargs)

    def _sanitize_stack_args(self, num_bag_folds, num_bag_sets, num_stack_levels, time_limit, auto_stack, num_train_rows):
        if auto_stack:
            # TODO: What about datasets that are 100k+? At a certain point should we not bag?
            # TODO: What about time_limit? Metalearning can tell us expected runtime of each model, then we can select optimal folds + stack levels to fit time constraint
            if num_bag_folds is None:
                num_bag_folds = min(10, max(5, math.floor(num_train_rows / 100)))
            if num_stack_levels is None:
                num_stack_levels = min(1, max(0, math.floor(num_train_rows / 750)))
        if num_bag_folds is None:
            num_bag_folds = 0
        if num_stack_levels is None:
            num_stack_levels = 0
        if not isinstance(num_bag_folds, int):
            raise ValueError(f'num_bag_folds must be an integer. (num_bag_folds={num_bag_folds})')
        if not isinstance(num_stack_levels, int):
            raise ValueError(f'num_stack_levels must be an integer. (num_stack_levels={num_stack_levels})')
        if num_bag_folds < 2 and num_bag_folds != 0:
            raise ValueError(f'num_bag_folds must be equal to 0 or >=2. (num_bag_folds={num_bag_folds})')
        if num_stack_levels != 0 and num_bag_folds == 0:
            raise ValueError(f'num_stack_levels must be 0 if num_bag_folds is 0. (num_stack_levels={num_stack_levels}, num_bag_folds={num_bag_folds})')
        if num_bag_sets is None:
            if num_bag_folds >= 2:
                if time_limit is not None:
                    num_bag_sets = 20  # TODO: v0.1 Reduce to 5 or 3 as 20 is unnecessarily extreme as a default.
                else:
                    num_bag_sets = 1
            else:
                num_bag_sets = 1
        if not isinstance(num_bag_sets, int):
            raise ValueError(f'num_bag_sets must be an integer. (num_bag_sets={num_bag_sets})')
        return num_bag_folds, num_bag_sets, num_stack_levels


# Location to store WIP functionality that will be later added to TabularPredictor
class _TabularPredictorExperimental(TabularPredictor):
    # TODO: Documentation, flesh out capabilities
    # TODO: Rename feature_generator -> feature_pipeline for users?
    # TODO: Return transformed data?
    # TODO: feature_generator_kwargs?
    def fit_feature_generator(self, data: pd.DataFrame, feature_generator='auto', feature_metadata=None):
        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata)
        self._learner.fit_transform_features(data)

    # TODO: rename to `advice`
    # TODO: Add documentation
    def _advice(self):
        is_feature_generator_fit = self._learner.feature_generator.is_fit()
        is_learner_fit = self._learner.trainer_path is not None
        exists_trainer = self._trainer is not None

        advice_dict = dict(
            is_feature_generator_fit=is_feature_generator_fit,
            is_learner_fit=is_learner_fit,
            exists_trainer=exists_trainer,
            # TODO
        )

        advice_list = []

        if not advice_dict['is_feature_generator_fit']:
            advice_list.append('FeatureGenerator has not been fit, consider calling `predictor.fit_feature_generator(data)`.')
        if not advice_dict['is_learner_fit']:
            advice_list.append('Learner is not fit, consider calling `predictor.fit(...)`')
        if not advice_dict['exists_trainer']:
            advice_list.append('Trainer is not initialized, consider calling `predictor.fit(...)`')
        # TODO: Advice on unused features (if no model uses a feature)
        # TODO: Advice on fit_extra
        # TODO: Advice on distill
        # TODO: Advice on leaderboard
        # TODO: Advice on persist
        # TODO: Advice on refit_full
        # TODO: Advice on feature_importance
        # TODO: Advice on dropping poor models

        logger.log(20, '======================= AutoGluon Advice =======================')
        if advice_list:
            for advice in advice_list:
                logger.log(20, advice)
        else:
            logger.log(20, 'No further advice found.')
        logger.log(20, '================================================================')

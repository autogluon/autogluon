import copy
import logging
import math

import numpy as np
import pandas as pd

from autogluon.core.task.base import compile_scheduler_options_v2
from autogluon.core.task.base.base_task import schedulers
from autogluon.core.utils import verbosity2loglevel
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.utils import setup_outputdir, setup_compute, setup_trial_limits, default_holdout_frac

from .dataset import TabularDataset
from .hyperparameter_configs import get_hyperparameter_config
from .predictor import TabularPredictor
from .presets_configs import set_presets, unpack
from ...features import AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator
from ...learner import AbstractLearner, DefaultLearner
from ...trainer import AbstractTrainer

logger = logging.getLogger()  # return root logger


# TODO: v0.1 Add comments that models are serialized on disk after fit
class TabularPredictorV2(TabularPredictor):
    """
    AutoGluon Task for predicting values in column of tabular dataset (classification or regression)
    """
    Dataset = TabularDataset
    predictor_file_name = 'predictor.pkl'

    # TODO: v0.1 add documentation to init
    def __init__(
            self,
            label,
            problem_type=None,
            eval_metric=None,
            output_directory=None,
            verbosity=2,
            **kwargs
    ):
        self._validate_init_kwargs(kwargs)
        output_directory = setup_outputdir(output_directory)  # TODO: Rename to directory/path?
        self.verbosity = verbosity

        learner_type = kwargs.pop('learner_type', DefaultLearner)
        learner_kwargs = kwargs.pop('learner_kwargs', dict())

        self._learner: AbstractLearner = learner_type(path_context=output_directory, label=label, feature_generator=None,
                                                      eval_metric=eval_metric, problem_type=problem_type, **learner_kwargs)
        self._learner_type = type(self._learner)
        self._trainer = None

    # TODO: Documentation, flesh out capabilities
    # TODO: Rename feature_generator -> feature_pipeline for users?
    # TODO: Return transformed data?
    # TODO: kwargs?
    def fit_feature_generator(self, data: pd.DataFrame, feature_generator='auto', feature_metadata=None):
        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata)
        self._learner.fit_transform_features(data)

    @unpack(set_presets)
    def fit(self,
            train_data,
            tuning_data=None,
            time_limits=None,
            presets=None,
            hyperparameters=None,
            feature_generator="auto",
            feature_metadata=None,
            hyperparameter_tune_kwargs=None,
            **kwargs):
        """
        Fit models to predict a column of data table based on the other columns.

        # TODO: Move documentation from TabularPrediction.fit to here
        # TODO: Move all scheduler/searcher specific arguments into hyperparameter_tune_kwargs

        """
        self._validate_fit_kwargs(kwargs)

        verbosity = kwargs.get('verbosity', self.verbosity)
        if verbosity < 0:
            verbosity = 0
        elif verbosity > 4:
            verbosity = 4

        logger.setLevel(verbosity2loglevel(verbosity))

        # TODO: Stopping metric -> Default to None for models, let model choose if None as if hyperparameter
        # AG_args_fit = {'num_gpus': -1} -> 'auto' by default
        # TODO: v0.1 - time_limits -> time_limit? -> +1
        # TODO: v0.1 - stack_ensemble_levels -> num_stack_levels / num_stack_layers? -> num_stack_levels
        # TODO: v0.1 - id_columns -> ignored_columns? -> +1
        # TODO: v0.1 - num_cpus/num_gpus -> rename/rework -> num_threads, num_gpus in AG_args_fit -> HPO overrides
        # TODO: v0.1 - visualizer -> consider reworking/removing -> AG_args_fit argument
        # TODO: v0.1 - HPO arguments to a generic hyperparameter_tune_kwargs parameter?
        # TODO: v0.1 - stack_ensemble_levels is silently ignored if num_bagging_folds < 2, ensure there is a warning printed

        holdout_frac = kwargs.get('holdout_frac', None)
        num_bagging_folds = kwargs.get('num_bagging_folds', None)
        num_bagging_sets = kwargs.get('num_bagging_sets', None)
        stack_ensemble_levels = kwargs.get('stack_ensemble_levels', None)
        auto_stack = kwargs.get('auto_stack', False)
        num_cpus = kwargs.get('num_cpus', None)
        num_gpus = kwargs.get('num_gpus', None)
        unlabeled_data = kwargs.get('unlabeled_data', None)
        save_bagged_folds = kwargs.get('save_bagged_folds', True)
        refit_full = kwargs.get('refit_full', False)
        set_best_to_refit_full = kwargs.get('set_best_to_refit_full', False)

        ag_args = kwargs.get('AG_args', None)
        ag_args_fit = kwargs.get('AG_args_fit', None)
        ag_args_ensemble = kwargs.get('AG_args_ensemble', None)
        excluded_model_types = kwargs.get('excluded_model_types', [])

        self._set_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata)
        train_data, tuning_data, unlabeled_data = self._validate_fit_data(train_data=train_data, tuning_data=tuning_data, unlabeled_data=unlabeled_data)

        if refit_full and not self._learner.cache_data:
            raise ValueError('`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.')
        if set_best_to_refit_full and not refit_full:
            raise ValueError('`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`.')

        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        # Process kwargs to create trainer, schedulers, searchers:
        num_bagging_folds, num_bagging_sets, stack_ensemble_levels = self._sanitize_stack_args(
            num_bagging_folds=num_bagging_folds, num_bagging_sets=num_bagging_sets, stack_ensemble_levels=stack_ensemble_levels,
            time_limits=time_limits, auto_stack=auto_stack, num_train_rows=len(train_data),
        )

        if hyperparameter_tune_kwargs is not None:
            scheduler_options = self._init_scheduler(hyperparameter_tune_kwargs, time_limits, hyperparameters, num_cpus, num_gpus, num_bagging_folds, stack_ensemble_levels)
        else:
            scheduler_options = None

        if scheduler_options is not None:
            hyperparameter_tune = True
            logger.log(30, 'Warning: hyperparameter tuning is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.')
        else:
            hyperparameter_tune = False

        if holdout_frac is None:
            holdout_frac = default_holdout_frac(len(train_data), hyperparameter_tune)

        # TODO: visualizer to args_fit?
        # TODO: Does not work with advanced model hyperparameters option
        if scheduler_options is not None:
            visualizer = scheduler_options[1]['visualizer']
            # Add visualizer to NN hyperparameters:
            if (visualizer is not None) and (visualizer != 'none') and ('NN' in hyperparameters):
                hyperparameters['NN']['visualizer'] = visualizer

        self._learner.fit(X=train_data, X_val=tuning_data, X_unlabeled=unlabeled_data, scheduler_options=scheduler_options,
                          hyperparameter_tune=hyperparameter_tune,
                          holdout_frac=holdout_frac, num_bagging_folds=num_bagging_folds, num_bagging_sets=num_bagging_sets, stack_ensemble_levels=stack_ensemble_levels,
                          hyperparameters=hyperparameters, ag_args=ag_args, ag_args_fit=ag_args_fit, ag_args_ensemble=ag_args_ensemble, excluded_model_types=excluded_model_types,
                          time_limit=time_limits, save_bagged_folds=save_bagged_folds, verbosity=verbosity)
        self._set_post_fit_vars()

        keep_only_best = kwargs.get('keep_only_best', False)
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

        save_space = kwargs.get('save_space', False)
        if save_space:
            self.save_space()
        self.save()

    def _init_scheduler(self, hyperparameter_tune_kwargs, time_limit, hyperparameters, num_cpus, num_gpus, num_bagging_folds, stack_ensemble_levels):
        num_cpus, num_gpus = setup_compute(num_cpus, num_gpus)
        time_limits_hpo = time_limit
        if num_bagging_folds >= 2 and (time_limits_hpo is not None):
            time_limits_hpo = time_limits_hpo / (1 + num_bagging_folds * (1 + stack_ensemble_levels))
        # FIXME: Incorrect if user specifies custom level-based hyperparameter config!
        time_limits_hpo, num_trials = setup_trial_limits(time_limits_hpo, None, hyperparameters)  # TODO: Move HPO time allocation to Trainer

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
            time_out=time_limits_hpo,
        )
        if scheduler_options is None:
            return None

        assert scheduler_options['searcher'] != 'bayesopt_hyperband', "searcher == 'bayesopt_hyperband' not yet supported"
        # TODO: Fix or remove in v0.1
        if scheduler_options.get('dist_ip_addrs', None):
            logger.log(30, 'Warning: dist_ip_addrs does not currently work. Distributed instances will not be utilized.')

        if scheduler_options['num_trials'] == 1:
            logger.log(30, 'Warning: Specified num_trials == 1 or time_limits is too small for hyperparameter_tune, disabling HPO.')
            return None  # FIXME

        scheduler_cls = schedulers[scheduler_options['searcher'].lower()]
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
    def load(cls, directory, verbosity=2):
        logger.setLevel(verbosity2loglevel(verbosity))  # Reset logging after load (may be in new Python session)
        if directory is None:
            raise ValueError("output_directory cannot be None in load()")

        directory = setup_outputdir(directory)  # replace ~ with absolute path if it exists
        predictor: TabularPredictorV2 = load_pkl.load(path=directory + cls.predictor_file_name)
        learner = predictor._learner_type.load(directory)
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

    @staticmethod
    def _validate_fit_kwargs(kwargs):
        allowed_kwarg_names = {
            'holdout_frac',
            'num_bagging_folds',
            'num_bagging_sets',
            'stack_ensemble_levels',
            'auto_stack',
            'AG_args',
            'AG_args_fit',
            'AG_args_ensemble',
            'excluded_model_types',
            'set_best_to_refit_full',
            'save_bagged_folds',
            'keep_only_best',
            'save_space',
            'refit_full',
            'num_cpus',
            'num_gpus',
            'unlabeled_data',
            'verbosity',
        }

        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                raise ValueError("Unknown keyword argument specified: %s" % kwarg_name)

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

    def _set_feature_generator(self, feature_generator='auto', feature_metadata=None):
        if self._learner.feature_generator is not None:
            if isinstance(feature_generator, str) and feature_generator == 'auto':
                feature_generator = self._learner.feature_generator
            else:
                raise AssertionError('FeatureGenerator already exists!')
        self._learner.feature_generator = self._get_default_feature_generator(feature_generator=feature_generator, feature_metadata=feature_metadata)

    # TODO: Move out of predictor?
    def _get_default_feature_generator(self, feature_generator, feature_metadata=None):
        if feature_generator is None:
            feature_generator = IdentityFeatureGenerator()
        elif isinstance(feature_generator, str):
            if feature_generator == 'auto':
                feature_generator = AutoMLPipelineFeatureGenerator()
            else:
                raise ValueError(f"Unknown feature_generator preset: '{feature_generator}', valid presets: {['auto']}")
        if feature_metadata is not None:
            if feature_generator.feature_metadata_in is None and not feature_generator.is_fit():
                feature_generator.feature_metadata_in = copy.deepcopy(feature_metadata)
            else:
                raise AssertionError('`feature_metadata_in` already exists in `feature_generator`.')
        return feature_generator

    def _sanitize_stack_args(self, num_bagging_folds, num_bagging_sets, stack_ensemble_levels, time_limits, auto_stack, num_train_rows):
        if auto_stack:
            # TODO: What about datasets that are 100k+? At a certain point should we not bag?
            # TODO: What about time_limits? Metalearning can tell us expected runtime of each model, then we can select optimal folds + stack levels to fit time constraint
            if num_bagging_folds is None:
                num_bagging_folds = min(10, max(5, math.floor(num_train_rows / 100)))
            if stack_ensemble_levels is None:
                stack_ensemble_levels = min(1, max(0, math.floor(num_train_rows / 750)))
        if num_bagging_folds is None:
            num_bagging_folds = 0
        if stack_ensemble_levels is None:
            stack_ensemble_levels = 0
        if not isinstance(num_bagging_folds, int):
            raise ValueError(f'num_bagging_folds must be an integer. (num_bagging_folds={num_bagging_folds})')
        if not isinstance(stack_ensemble_levels, int):
            raise ValueError(f'stack_ensemble_levels must be an integer. (stack_ensemble_levels={stack_ensemble_levels})')
        if num_bagging_folds < 2 and num_bagging_folds != 0:
            raise ValueError(f'num_bagging_folds must be equal to 0 or >=2. (num_bagging_folds={num_bagging_folds})')
        if stack_ensemble_levels != 0 and num_bagging_folds == 0:
            raise ValueError(f'stack_ensemble_levels must be 0 if num_bagging_folds is 0. (stack_ensemble_levels={stack_ensemble_levels}, num_bagging_folds={num_bagging_folds})')
        if num_bagging_sets is None:
            if num_bagging_folds >= 2:
                if time_limits is not None:
                    num_bagging_sets = 20  # TODO: v0.1 Reduce to 5 or 3 as 20 is unnecessarily extreme as a default.
                else:
                    num_bagging_sets = 1
            else:
                num_bagging_sets = 1
        if not isinstance(num_bagging_sets, int):
            raise ValueError(f'num_bagging_sets must be an integer. (num_bagging_sets={num_bagging_sets})')
        return num_bagging_folds, num_bagging_sets, stack_ensemble_levels

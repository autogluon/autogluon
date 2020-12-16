import copy
import logging
import math

import numpy as np
import pandas as pd

from autogluon.core.task.base import compile_scheduler_options
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
            feature_metadata_in=None,
            feature_generator="automl",
            output_directory=None,
            verbosity=2,
            **kwargs
    ):
        self._validate_init_kwargs(kwargs)
        output_directory = setup_outputdir(output_directory)  # TODO: Rename to directory/path?
        self.feature_metadata_in = feature_metadata_in  # TODO: Unused, FIXME: currently overwritten after .fit, split into two variables: one for pre and one for post processing.
        self.verbosity = verbosity  # TODO: Unused

        learner_type = kwargs.pop('learner_type', DefaultLearner)
        learner_kwargs = kwargs.pop('learner_kwargs', dict())

        # TODO: v0.1 Add presets for feature_generator: `feature_generator='special'`, ignore_text, etc.
        if feature_generator is None:
            feature_generator = IdentityFeatureGenerator()
        elif feature_generator == "automl":
            feature_generator = AutoMLPipelineFeatureGenerator()
        if feature_metadata_in is not None:
            if feature_generator.feature_metadata_in is None and not feature_generator.is_fit():
                feature_generator.feature_metadata_in = copy.deepcopy(feature_metadata_in)
            else:
                raise AssertionError('`feature_metadata_in` already exists in `feature_generator`.')

        self._learner: AbstractLearner = learner_type(path_context=output_directory, label=label, feature_generator=feature_generator,
                                                      eval_metric=eval_metric, problem_type=problem_type, **learner_kwargs)
        self._learner_type = type(self._learner)
        self._trainer = None

    # TODO: Documentation, flesh out capabilities
    def fit_feature_generator(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._learner.fit_transform_features(data)

    @unpack(set_presets)
    def fit(self,
            train_data,
            tuning_data=None,
            time_limits=None,
            presets=None,
            eval_metric=None,
            stopping_metric=None,
            auto_stack=False,
            hyperparameter_tune=False,
            hyperparameters=None,
            holdout_frac=None,
            num_bagging_folds=0,
            num_bagging_sets=None,
            stack_ensemble_levels=0,
            num_trials=None,
            search_strategy='random',
            **kwargs):
        """
        Fit models to predict a column of data table based on the other columns.

        # TODO: Move documentation from TabularPrediction.fit to here
        # TODO: Move all scheduler/searcher specific arguments into search_options and scheduler_options to simplify documentation.

        """
        self._validate_fit_kwargs(kwargs)

        assert search_strategy != 'bayesopt_hyperband', "search_strategy == 'bayesopt_hyperband' not yet supported"

        verbosity = kwargs.get('verbosity', self.verbosity)
        if verbosity < 0:
            verbosity = 0
        elif verbosity > 4:
            verbosity = 4

        logger.setLevel(verbosity2loglevel(verbosity))

        # TODO: v0.1 - time_limits -> time_limit?
        # TODO: v0.1 - stack_ensemble_levels -> num_stack_levels / num_stack_layers?
        # TODO: v0.1 - id_columns -> ignored_columns?
        # TODO: v0.1 - nthreads_per_trial/ngpus_per_trial -> rename/rework
        # TODO: v0.1 - visualizer -> consider reworking/removing
        # TODO: v0.1 - HPO arguments to a generic hyperparameter_tune_kwargs parameter?
        # TODO: v0.1 - stack_ensemble_levels is silently ignored if num_bagging_folds < 2, ensure there is a warning printed

        feature_prune = kwargs.get('feature_prune', False)
        scheduler_options = kwargs.get('scheduler_options', None)
        search_options = kwargs.get('search_options', None)
        nthreads_per_trial = kwargs.get('nthreads_per_trial', None)
        ngpus_per_trial = kwargs.get('ngpus_per_trial', None)
        dist_ip_addrs = kwargs.get('dist_ip_addrs', None)
        visualizer = kwargs.get('visualizer', None)
        unlabeled_data = kwargs.get('unlabeled_data', None)

        train_data, tuning_data, unlabeled_data = self._validate_fit_data(train_data=train_data, tuning_data=tuning_data, unlabeled_data=unlabeled_data)

        if feature_prune:
            feature_prune = False  # TODO: Fix feature pruning to add back as an option
            # Currently disabled, needs to be updated to align with new model class functionality
            logger.log(30, 'Warning: feature_prune does not currently work, setting to False.')
        # TODO: Fix or remove in v0.1
        if dist_ip_addrs is not None:
            logger.log(30, 'Warning: dist_ip_addrs does not currently work. Distributed instances will not be utilized.')

        refit_full = kwargs.get('refit_full', False)
        if refit_full and not self._learner.cache_data:
            raise ValueError('`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.')

        set_best_to_refit_full = kwargs.get('set_best_to_refit_full', False)
        if set_best_to_refit_full and not refit_full:
            raise ValueError('`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`.')

        save_bagged_folds = kwargs.get('save_bagged_folds', True)

        if hyperparameter_tune:
            logger.log(30, 'Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.')

        if dist_ip_addrs is None:
            dist_ip_addrs = []

        if search_options is None:
            search_options = dict()

        if hyperparameters is None:
            hyperparameters = 'default'
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)

        # Process kwargs to create trainer, schedulers, searchers:
        ag_args = kwargs.get('AG_args', None)
        ag_args_fit = kwargs.get('AG_args_fit', None)
        ag_args_ensemble = kwargs.get('AG_args_ensemble', None)
        excluded_model_types = kwargs.get('excluded_model_types', [])
        nthreads_per_trial, ngpus_per_trial = setup_compute(nthreads_per_trial, ngpus_per_trial)
        num_train_rows = len(train_data)
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

        time_limits_orig = copy.deepcopy(time_limits)
        time_limits_hpo = copy.deepcopy(time_limits)

        if num_bagging_folds >= 2 and (time_limits_hpo is not None):
            time_limits_hpo = time_limits_hpo / (1 + num_bagging_folds * (1 + stack_ensemble_levels))
        # FIXME: Incorrect if user specifies custom level-based hyperparameter config!
        time_limits_hpo, num_trials = setup_trial_limits(time_limits_hpo, num_trials, hyperparameters)  # TODO: Move HPO time allocation to Trainer

        if (num_trials is not None) and hyperparameter_tune and (num_trials == 1):
            hyperparameter_tune = False
            logger.log(30, 'Warning: Specified num_trials == 1 or time_limits is too small for hyperparameter_tune, setting to False.')

        if holdout_frac is None:
            holdout_frac = default_holdout_frac(num_train_rows, hyperparameter_tune)

        # Add visualizer to NN hyperparameters:
        if (visualizer is not None) and (visualizer != 'none') and ('NN' in hyperparameters):
            hyperparameters['NN']['visualizer'] = visualizer

        # All models use the same scheduler:
        scheduler_options = compile_scheduler_options(
            scheduler_options=scheduler_options,
            search_strategy=search_strategy,
            search_options=search_options,
            nthreads_per_trial=nthreads_per_trial,
            ngpus_per_trial=ngpus_per_trial,
            checkpoint=None,
            num_trials=num_trials,
            time_out=time_limits_hpo,
            resume=False,
            visualizer=visualizer,
            time_attr='epoch',
            reward_attr='validation_performance',
            dist_ip_addrs=dist_ip_addrs)
        scheduler_cls = schedulers[search_strategy.lower()]
        scheduler_options = (scheduler_cls, scheduler_options)  # wrap into tuple
        self._learner.fit(X=train_data, X_val=tuning_data, X_unlabeled=unlabeled_data, scheduler_options=scheduler_options,
                          hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                          holdout_frac=holdout_frac, num_bagging_folds=num_bagging_folds, num_bagging_sets=num_bagging_sets, stack_ensemble_levels=stack_ensemble_levels,
                          hyperparameters=hyperparameters, ag_args=ag_args, ag_args_fit=ag_args_fit, ag_args_ensemble=ag_args_ensemble, excluded_model_types=excluded_model_types,
                          time_limit=time_limits_orig, save_bagged_folds=save_bagged_folds, verbosity=verbosity)
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

        # TODO: v0.1: return newly trained model names?

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
            'AG_args',
            'AG_args_fit',
            'AG_args_ensemble',
            'excluded_model_types',
            'set_best_to_refit_full',
            'save_bagged_folds',
            'keep_only_best',
            'save_space',
            'refit_full',
            'feature_prune',
            'scheduler_options',
            'search_options',
            'nthreads_per_trial',
            'ngpus_per_trial',
            'dist_ip_addrs',
            'visualizer',
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

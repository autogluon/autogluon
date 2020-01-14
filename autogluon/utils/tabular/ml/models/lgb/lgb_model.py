import gc, copy, random, time, os, logging, warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from ......core import Int, Space
from .....try_import import try_import_lightgbm
from ......task.base import BasePredictor
from ..abstract.abstract_model import AbstractModel, fixedvals_from_searchspaces
from ...utils import construct_dataset
from .callbacks import early_stopping_custom
from ...constants import BINARY, MULTICLASS, REGRESSION
from . import lgb_utils
from .hyperparameters.searchspaces import get_default_searchspace
from .hyperparameters.lgb_trial import lgb_trial
from .hyperparameters.parameters import get_param_baseline

warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version") # lightGBM brew libomp warning
logger = logging.getLogger(__name__)  # TODO: Currently


# TODO: Save dataset to binary and reload for HPO. This will avoid the memory spike overhead when training each model and instead it will only occur once upon saving the dataset.
class LGBModel(AbstractModel):
    def __init__(self, path: str, name: str, problem_type: str, objective_func,
                 num_classes=None, hyperparameters=None, features=None, debug=0):
        self.num_classes = num_classes
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features, debug=debug)

        self.eval_metric_name = self.objective_func.name
        self.is_higher_better = True
        self.best_iteration = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type, num_classes=self.num_classes)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def get_eval_metric(self):
        return lgb_utils.func_generator(metric=self.objective_func, is_higher_better=True, needs_pred_proba=not self.metric_needs_y_pred, problem_type=self.problem_type)

    # TODO: Avoid deleting X_train and X_test to not corrupt future runs
    def fit(self, X_train=None, Y_train=None, X_test=None, Y_test=None, dataset_train=None, dataset_val=None, time_limit=None, **kwargs):
        start_time = time.time()
        params = self.params.copy()
        # TODO: kwargs can have num_cpu, num_gpu. Currently these are ignored.
        verbosity = kwargs.get('verbosity', 2)
        params = fixedvals_from_searchspaces(params)
        if 'min_data_in_leaf' in params:
            if params['min_data_in_leaf'] > X_train.shape[0]: # TODO: may not be necessary
                params['min_data_in_leaf'] = max(1, int(X_train.shape[0]/5.0))

        num_boost_round = params.pop('num_boost_round', 1000)
        logger.log(15, 'Training Gradient Boosting Model for %s rounds...' % num_boost_round)
        logger.log(15, "with the following hyperparameter settings:")
        logger.log(15, params)
        seed_val = params.pop('seed_value', None)

        eval_metric = self.get_eval_metric()
        dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, dataset_train=dataset_train, dataset_val=dataset_val)
        gc.collect()
        
        if verbosity <= 1:
            verbose_eval = False
        elif verbosity == 2:
            verbose_eval = 1000
        elif verbosity == 3:
            verbose_eval = 50
        else:
            verbose_eval = True
        
        self.eval_results = {}
        callbacks = []
        valid_names = ['train_set']
        valid_sets = [dataset_train]
        if dataset_val is not None:
            callbacks += [
                early_stopping_custom(150, metrics_to_use=[('valid_set', self.eval_metric_name)], max_diff=None, start_time=start_time, time_limit=time_limit,
                                      ignore_dart_warning=True, verbose=verbose_eval, manual_stop_file=False),
                ]
            valid_names = ['valid_set'] + valid_names
            valid_sets = [dataset_val] + valid_sets

        try_import_lightgbm()
        import lightgbm as lgb
        callbacks += [
            # lgb.reset_parameter(learning_rate=lambda iter: alpha * (0.999 ** iter)),
        ]
        # lr_over_time = lambda iter: 0.05 * (0.99 ** iter)
        # alpha = 0.1
        
        train_params = {
            'params': params.copy(),
            'train_set': dataset_train,
            'num_boost_round': num_boost_round,
            'valid_sets': valid_sets,
            'valid_names': valid_names,
            'callbacks': callbacks,
            'verbose_eval': verbose_eval,
        }
        if type(eval_metric) != str:
            train_params['feval'] = eval_metric
        if seed_val is not None:
            train_params['params']['seed'] = seed_val
            random.seed(seed_val)
            np.random.seed(seed_val)
        # Train lgbm model:
        self.model = lgb.train(**train_params)
        # del dataset_train
        # del dataset_val
        # print('running gc...')
        # gc.collect()
        # print('ran garbage collection...')
        self.best_iteration = self.model.best_iteration
        params['num_boost_round'] = num_boost_round
        if seed_val is not None:
            params['seed_value'] = seed_val
        # self.model.save_model(self.path + 'model.txt')
        # model_json = self.model.dump_model()
        #
        # with open(self.path + 'model.json', 'w+') as f:
        #     json.dump(model_json, f, indent=4)
        # save_pkl.save(path=self.path + self.model_name_trained, object=self)  # TODO: saving self instead of model, not consistent with save callbacks
        # save_pointer.save(path=self.path + self.latest_model_checkpoint, content_path=self.path + self.model_name_trained)

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)
        if self.problem_type == REGRESSION:
            return self.model.predict(X)

        y_pred_proba = self.model.predict(X)
        if (self.problem_type == BINARY):
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == MULTICLASS:
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def cv(self, X=None, y=None, k_fold=5, dataset_train=None):
        logger.warning("Warning: Running GBM cross-validation. This is currently unstable.")
        try_import_lightgbm()
        import lightgbm as lgb
        if dataset_train is None:
            dataset_train, _ = self.generate_datasets(X_train=X, Y_train=y)
        gc.collect()
        params = copy.deepcopy(self.params)
        eval_metric = self.get_eval_metric()
        # TODO: Either edit lgb.cv to return models / oof preds or make custom implementation!
        cv_params = {
            'params': params,
            'train_set': dataset_train,
            'num_boost_round': self.num_boost_round,
            'nfold': k_fold,
            'early_stopping_rounds': 150,
            'verbose_eval': 1000,
            'seed': 0,
        }
        if type(eval_metric) != str:
            cv_params['feval'] = eval_metric
            cv_params['params']['metric'] = 'None'
        else:
            cv_params['params']['metric'] = eval_metric
        if self.problem_type == REGRESSION:
            cv_params['stratified'] = False

        logger.log(15, 'Current parameters:')
        logger.log(15, params)
        eval_hist = lgb.cv(**cv_params)  # TODO: Try to use customer early stopper to enable dart
        best_score = eval_hist[self.eval_metric_name + '-mean'][-1]
        logger.log(15, 'Best num_boost_round: %s ', len(eval_hist[self.eval_metric_name+'-mean']))
        logger.log(15, 'Best CV score: %s' % best_score)
        return best_score

    def convert_to_weight(self, X: DataFrame):
        logger.debug(X)
        w = X['count']
        X = X.drop(['count'], axis=1)
        return X, w

    def generate_datasets(self, X_train: DataFrame, Y_train: Series, X_test=None, Y_test=None, dataset_train=None, dataset_val=None, save=False):
        lgb_dataset_params_keys = ['objective', 'two_round','num_threads', 'num_classes', 'verbose'] # Keys that are specific to lightGBM Dataset object construction.
        data_params = {}
        for key in lgb_dataset_params_keys:
            if key in self.params:
                data_params[key] = self.params[key]
        data_params = data_params.copy()

        W_train = None  # TODO: Add weight support
        W_test = None  # TODO: Add weight support
        if X_train is not None:
            X_train = self.preprocess(X_train)
        if X_test is not None:
            X_test = self.preprocess(X_test)
        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike
        if not dataset_train:
            # X_train, W_train = self.convert_to_weight(X=X_train)
            dataset_train = construct_dataset(x=X_train, y=Y_train, location=self.path + 'datasets/train', params=data_params, save=save, weight=W_train)
            # dataset_train = construct_dataset_lowest_memory(X=X_train, y=Y_train, location=self.path + 'datasets/train', params=data_params)
        if (not dataset_val) and (X_test is not None) and (Y_test is not None):
            # X_test, W_test = self.convert_to_weight(X=X_test)
            dataset_val = construct_dataset(x=X_test, y=Y_test, location=self.path + 'datasets/val', reference=dataset_train, params=data_params, save=save, weight=W_test)
            # dataset_val = construct_dataset_lowest_memory(X=X_test, y=Y_test, location=self.path + 'datasets/val', reference=dataset_train, params=data_params)
        return dataset_train, dataset_val

    def debug_features_to_use(self, X_test_in):
        feature_splits = self.model.feature_importance()
        total_splits = feature_splits.sum()
        feature_names = list(X_test_in.columns.values)
        feature_count = len(feature_names)
        feature_importances = pd.DataFrame(data=feature_names, columns=['feature'])
        feature_importances['splits'] = feature_splits
        feature_importances_unused = feature_importances[feature_importances['splits'] == 0]
        feature_importances_used = feature_importances[feature_importances['splits'] >= (total_splits/feature_count)]
        logger.debug(feature_importances_unused)
        logger.debug(feature_importances_used)
        logger.debug('feature_importances_unused: %s' % len(feature_importances_unused))
        logger.debug('feature_importances_used: %s' % len(feature_importances_used))
        features_to_use = list(feature_importances_used['feature'].values)
        logger.debug(str(features_to_use))
        return features_to_use

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        # verbosity = kwargs.get('verbosity', 2)
        start_time = time.time()
        logger.log(15, "Beginning hyperparameter tuning for Gradient Boosting Model...")
        self._set_default_searchspace()
        if isinstance(self.params['min_data_in_leaf'], Int):
            upper_minleaf = self.params['min_data_in_leaf'].upper
            if upper_minleaf > X_train.shape[0]: # TODO: this min_data_in_leaf adjustment based on sample size may not be necessary
                upper_minleaf = max(1, int(X_train.shape[0]/5.0))
                lower_minleaf = self.params['min_data_in_leaf'].lower
                if lower_minleaf > upper_minleaf:
                    lower_minleaf = max(1, int(upper_minleaf/3.0))
                self.params['min_data_in_leaf'] = Int(lower=lower_minleaf, upper=upper_minleaf)
        params_copy = self.params.copy()
        seed_val = self.params.pop('seed_value', None)
        num_boost_round = self.params.pop('num_boost_round', 1000)

        directory = self.path # also create model directory if it doesn't exist
        # TODO: This will break on S3! Use tabular/utils/savers for datasets, add new function
        if not os.path.exists(directory):
            os.makedirs(directory)
        scheduler_func = scheduler_options[0] # Unpack tuple
        scheduler_options = scheduler_options[1]
        if scheduler_func is None or scheduler_options is None:
            raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")
        num_threads = scheduler_options['resource'].get('num_cpus', -1)
        self.params['num_threads'] = num_threads
        # num_gpus = scheduler_options['resource']['num_gpus'] # TODO: unused

        dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        dataset_train_filename = "dataset_train.bin"
        train_file = self.path + dataset_train_filename
        if os.path.exists(train_file): # clean up old files first
            os.remove(train_file)
        dataset_train.save_binary(train_file)
        if dataset_val is not None:
            dataset_val_filename = "dataset_val.bin" # names without directory info
            val_file = self.path + dataset_val_filename
            if os.path.exists(val_file): # clean up old files first
                os.remove(val_file)
            dataset_val.save_binary(val_file)
        else:
            dataset_val_filename = None

        if not np.any([isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy]):
            logger.warning("Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for Gradient Boosting Model: ")
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, str(hyperparam)+":   " + str(params_copy[hyperparam]))

        lgb_trial.register_args(dataset_train_filename=dataset_train_filename,
            dataset_val_filename=dataset_val_filename,
            directory=directory, lgb_model=self, **params_copy)
        scheduler = scheduler_func(lgb_trial, **scheduler_options)
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading data to remote workers...")
            scheduler.upload_files([train_file, val_file]) # TODO: currently does not work.
            directory = self.path # TODO: need to change to path to working directory used on every remote machine
            lgb_trial.update(directory=directory)
            logger.log(15, "uploaded")

        scheduler.run()
        scheduler.join_jobs()

        # Store results / models from this HPO run:
        best_hp = scheduler.get_best_config() # best_hp only contains searchable stuff
        hpo_results = {'best_reward': scheduler.get_best_reward(),
                       'best_config': best_hp,
                       'total_time': time.time() - start_time,
                       'metadata': scheduler.metadata,
                       'training_history': scheduler.training_history,
                       'config_history': scheduler.config_history,
                       'reward_attr': scheduler._reward_attr,
                       'args': lgb_trial.args
                      }
        hpo_results = BasePredictor._format_results(hpo_results) # results summarizing HPO for this model
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            raise NotImplementedError("need to fetch model files from remote Workers")
            # TODO: need to handle locations carefully: fetch these files and put them into self.path directory:
            # 1) hpo_results['trial_info'][trial]['metadata']['trial_model_file']
        hpo_models = {} # stores all the model names and file paths to model objects created during this HPO run.
        hpo_model_performances = {}
        for trial in sorted(hpo_results['trial_info'].keys()):
            # TODO: ignore models which were killed early by scheduler (eg. in Hyperband). Ask Hang how to ID these?
            file_id = "trial_"+str(trial) # unique identifier to files from this trial
            file_prefix = file_id + "_"
            trial_model_name = self.name+"_"+file_id
            trial_model_path = self.path + file_prefix
            hpo_models[trial_model_name] = trial_model_path
            hpo_model_performances[trial_model_name] = hpo_results['trial_info'][trial][scheduler._reward_attr]

        logger.log(15, "Time for Gradient Boosting hyperparameter optimization: %s" % str(hpo_results['total_time']))
        self.params.update(best_hp)
        # TODO: reload model params from best trial? Do we want to save this under cls.model_file as the "optimal model"
        logger.log(15, "Best hyperparameter configuration for Gradient Boosting Model: ")
        logger.log(15, best_hp)
        return (hpo_models, hpo_model_performances, hpo_results)
        # TODO: do final fit here?
        # args.final_fit = True
        # final_model = scheduler.run_with_config(best_config)
        # save(final_model)

    def _set_default_searchspace(self):
        """ Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from
            default fixed value to default spearch space.
        """
        def_search_space = get_default_searchspace(problem_type=self.problem_type, num_classes=self.num_classes).copy()
        for key in self.nondefault_params: # delete all user-specified hyperparams from the default search space
            _ = def_search_space.pop(key, None)
        self.params.update(def_search_space)


""" OLD code: TODO: remove once confirming no longer needed.

# my (minorly) revised HPO function 

    def hyperparameter_tune(self, X_train, X_test, y_train, y_test, spaces=None, scheduler_options=None): # scheduler_options unused for now
        print("Beginning hyperparameter tuning for Gradient Boosting Model...")
        X = pd.concat([X_train, X_test], ignore_index=True)
        y = pd.concat([y_train, y_test], ignore_index=True)
        if spaces is None:
            spaces = LGBMSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()

        X = self.preprocess(X)
        dataset_train, _ = self.generate_datasets(X_train=X, Y_train=y)
        space = spaces[0]
        param_baseline = self.params

        @use_named_args(space)
        def objective(**params):
            print(params)
            new_params = copy.deepcopy(param_baseline)
            new_params['verbose'] = -1
            for param in params:
                new_params[param] = params[param]

            new_model = copy.deepcopy(self)
            new_model.params = new_params
            score = new_model.cv(dataset_train=dataset_train)

            print(score)
            if self.is_higher_better:
                score = -score

            return score

        reg_gp = gp_minimize(objective, space, verbose=True, n_jobs=1, n_calls=15)

        print('best score: {}'.format(reg_gp.fun))

        optimal_params = copy.deepcopy(param_baseline)
        for i, param in enumerate(space):
            optimal_params[param.name] = reg_gp.x[i]

        self.params = optimal_params
        print(self.params)

        # TODO: final fit should not be here eventually
        self.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
        self.save()
        return ({self.name: self.path}, {}) # dummy hpo_info

# Original HPO function: 

    def hyperparameter_tune(self, X, y, spaces=None):
        if spaces is None:
            spaces = LGBMSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()

        X = self.preprocess(X)
        dataset_train, _ = self.generate_datasets(X_train=X, Y_train=y)

        print('starting skopt')
        space = spaces[0]

        param_baseline = self.params

        @use_named_args(space)
        def objective(**params):
            print(params)
            new_params = copy.deepcopy(param_baseline)
            new_params['verbose'] = -1
            for param in params:
                new_params[param] = params[param]

            new_model = copy.deepcopy(self)
            new_model.params = new_params
            score = new_model.cv(dataset_train=dataset_train)

            print(score)
            if self.is_higher_better:
                score = -score

            return score

        reg_gp = gp_minimize(objective, space, verbose=True, n_jobs=1, n_calls=15)

        print('best score: {}'.format(reg_gp.fun))

        optimal_params = copy.deepcopy(param_baseline)
        for i, param in enumerate(space):
            optimal_params[param.name] = reg_gp.x[i]

        self.params = optimal_params
        print(self.params)
        return optimal_params

"""

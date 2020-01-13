import logging, time, pickle, os
import numpy as np
import pandas as pd

from ...utils import get_pred_from_proba
from ...constants import BINARY, REGRESSION
from ......core import Space, Categorical, List, NestedSpace
from ......task.base import BasePredictor
from .... import metrics
from ....utils.decorators import calculate_time
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl
from .model_trial import model_trial

logger = logging.getLogger(__name__)


# Methods useful for all models:
def fixedvals_from_searchspaces(params):
    """ Converts any search space hyperparams in params dict into fixed default values. """
    if np.any([isinstance(params[hyperparam], Space) for hyperparam in params]):
        logger.warning("Attempting to fit model without HPO, but search space is provided. fit() will only consider default hyperparameter values from search space.")
        bad_keys = [hyperparam for hyperparam in params if isinstance(params[hyperparam], Space)][:] # delete all keys which are of type autogluon Space
        params = params.copy()
        for hyperparam in bad_keys:
            params[hyperparam] = hp_default_value(params[hyperparam])
        return params
    else:
        return params


def hp_default_value(hp_value):
    """ Extracts default fixed value from hyperparameter search space hp_value to use a fixed value instead of a search space.
    """
    if not isinstance(hp_value, Space):
        return hp_value
    if isinstance(hp_value, Categorical):
        return hp_value[0]
    elif isinstance(hp_value, List):
        return [z[0] for z in hp_value]
    elif isinstance(hp_value, NestedSpace):
        raise ValueError("Cannot extract default value from NestedSpace. Please specify fixed value instead of: %s" % str(hp_value))
    else:
        return hp_value.get_hp('dummy_name').default_value


class AbstractModel:
    model_file_name = 'model.pkl'

    def __init__(self, path: str, name: str, problem_type: str, objective_func, model=None, hyperparameters=None, features=None, feature_types_metadata=None, debug=0):
        """ Creates a new model. 
            Args:
                path (str): directory where to store all outputs
                name (str): name of subdirectory inside path where model will be saved
                hyperparameters (dict): various hyperparameters that will be used by model (can be search spaces instead of fixed values)
        """
        self.name = name
        self.path = self.create_contexts(path + name + '/')
        self.model = model
        self.problem_type = problem_type
        self.objective_func = objective_func # Note: we require higher values = better performance
        self.feature_types_metadata = feature_types_metadata  # TODO: Should this be passed to a model on creation? Should it live in a Dataset object and passed during fit? Currently it is being updated prior to fit by trainer

        if type(objective_func) == metrics._ProbaScorer:
            self.metric_needs_y_pred = False
        elif type(objective_func) == metrics._ThresholdScorer:
            self.metric_needs_y_pred = False
        else:
            self.metric_needs_y_pred = True

        self.features = features
        self.debug = debug
        if type(model) == str:
            self.model = self.load_model(model)

        self.params = {}
        self._set_default_params()
        self.nondefault_params = []
        if hyperparameters is not None:
            self.params.update(hyperparameters.copy())
            self.nondefault_params = list(hyperparameters.keys())[:] # These are hyperparameters that user has specified.

    # Checks if model is ready to make predictions for final result
    def is_valid(self):
        return self.is_fit()

    # Checks if a model has been fit
    def is_fit(self):
        return self.model is not None

    def _set_default_params(self):
        pass

    def _set_default_param_value(self, param_name, param_value):
        if param_name not in self.params:
            self.params[param_name] = param_value

    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        path = path_context
        return path

    # Extensions of preprocess must act identical in bagged situations, otherwise test-time predictions will be incorrect
    # This means preprocess cannot be used for normalization
    # TODO: Add preprocess_stateful() to enable stateful preprocessing for models such as KNN
    def preprocess(self, X):
        if self.features is not None:
            return X[self.features]
        return X

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, **kwargs):
        # kwargs may contain: num_cpus, num_gpus
        X_train = self.preprocess(X_train)
        self.model = self.model.fit(X_train, Y_train)

    def predict(self, X, preprocess=True):
        y_pred_proba = self.predict_proba(X, preprocess=preprocess)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)
        if self.problem_type == REGRESSION:
            return self.model.predict(X)

        y_pred_proba = self.model.predict_proba(X)

        if self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif y_pred_proba.shape[1] > 2:
            return y_pred_proba
        else:
            return y_pred_proba[:, 1]

    def score(self, X, y):
        if self.metric_needs_y_pred:
            y_pred = self.predict(X=X)
            return self.objective_func(y, y_pred)
        else:
            y_pred_proba = self.predict_proba(X=X)
            return self.objective_func(y, y_pred_proba)

    def score_with_y_pred_proba(self, y, y_pred_proba):
        if self.metric_needs_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return self.objective_func(y, y_pred)
        else:
            return self.objective_func(y, y_pred_proba)

    # TODO: Add simple generic CV logic
    def cv(self, X, y, k_fold=5):
        raise NotImplementedError

    def save(self, file_prefix ="", directory = None, return_filename=False, verbose=True):
        if directory is None:
            directory = self.path
        file_name = directory + file_prefix + self.model_file_name
        save_pkl.save(path=file_name, object=self, verbose=verbose)
        if return_filename:
            return file_name

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        load_path = path + file_prefix + cls.model_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path, verbose=verbose)
        else:
            obj = load_pkl.load(path=load_path, verbose=verbose)
            obj.set_contexts(path)
            return obj

    @calculate_time
    def debug_feature_gain(self, X_test, Y_test, model, features_to_use=None):
        sample_size = 10000
        if len(X_test) > sample_size:
            X_test = X_test.sample(sample_size, random_state=0)
            Y_test = Y_test.loc[X_test.index]
        else:
            X_test = X_test.copy()
            Y_test = Y_test.copy()

        X_test.reset_index(drop=True, inplace=True)
        Y_test.reset_index(drop=True, inplace=True)

        X_test = model.preprocess(X_test)

        if not features_to_use:
            features = X_test.columns.values
        else:
            features = features_to_use
        feature_count = len(features)

        model_score_base = model.score(X=X_test, y=Y_test)

        model_score_diff = []

        row_count = X_test.shape[0]
        rand_shuffle = np.random.randint(0, row_count, size=row_count)

        X_test_shuffled = X_test.iloc[rand_shuffle].reset_index(drop=True)
        compute_count = 200
        indices = [x for x in range(0, feature_count, compute_count)]

        # TODO: Make this faster by multi-threading?
        for i, indice in enumerate(indices):
            if indice + compute_count > feature_count:
                compute_count = feature_count - indice

            logger.debug(indice)
            x = [X_test.copy() for _ in range(compute_count)]  # TODO Make this much faster, only make this and concat it once. Then just update values and reset the values edited each iteration
            for j, val in enumerate(x):
                feature = features[indice+j]
                val[feature] = X_test_shuffled[feature]
            X_test_raw = pd.concat(x, ignore_index=True)
            if model.metric_needs_y_pred:
                Y_pred = model.predict(X_test_raw, preprocess=False)
            else:
                Y_pred = model.predict_proba(X_test_raw, preprocess=False)
            row_index = 0
            for j in range(compute_count):
                row_index_end = row_index + row_count
                Y_pred_cur = Y_pred[row_index:row_index_end]
                row_index = row_index_end
                score = model.objective_func(Y_test, Y_pred_cur)
                model_score_diff.append(model_score_base - score)

        results = pd.Series(data=model_score_diff, index=features)
        results = results.sort_values(ascending=False)

        return results
        # self.save_debug()

    def _get_default_searchspace(self, problem_type):
        return NotImplementedError

    def _set_default_searchspace(self):
        """ Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from
            default fixed value to default spearch space.
        """
        def_search_space = self._get_default_searchspace(problem_type=self.problem_type).copy()
        # Note: when subclassing AbstractModel, you must define or import get_default_searchspace() from the appropriate location.
        for key in self.nondefault_params: # delete all user-specified hyperparams from the default search space
            _ = def_search_space.pop(key, None)
        if self.params is not None:
            self.params.update(def_search_space)

    # After calling this function, model should be able to be fit as if it was new, as well as deep-copied.
    def convert_to_template(self):
        return self

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        # verbosity = kwargs.get('verbosity', 2)
        start_time = time.time()
        logger.log(15, "Starting generic AbstractModel hyperparameter tuning for %s model..." % self.name)
        self._set_default_searchspace()
        params_copy = self.params.copy()
        directory = self.path # also create model directory if it doesn't exist
        # TODO: This will break on S3. Use tabular/utils/savers for datasets, add new function
        if not os.path.exists(directory):
            os.makedirs(directory)
        scheduler_func = scheduler_options[0] # Unpack tuple
        scheduler_options = scheduler_options[1]
        if scheduler_func is None or scheduler_options is None:
            raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")
        self.params['num_threads'] = scheduler_options['resource'].get('num_cpus', None)
        self.params['num_gpus'] = scheduler_options['resource'].get('num_gpus', None)
        dataset_train_filename = 'dataset_train.p'
        train_path = directory+dataset_train_filename
        pickle.dump((X_train,Y_train), open(train_path, 'wb'))
        if (X_test is not None) and (Y_test is not None):
            dataset_val_filename = 'dataset_val.p'
            val_path = directory+dataset_val_filename
            pickle.dump((X_test,Y_test), open(val_path, 'wb'))
        else:
            dataset_val_filename = None
        if not np.any([isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy]):
            logger.warning("Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for %s model: " % self.name)
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, str(hyperparam)+ ":   " +str(params_copy[hyperparam]))

        model_trial.register_args(dataset_train_filename=dataset_train_filename,
            dataset_val_filename=dataset_val_filename, directory=directory, model=self, **params_copy)
        scheduler = scheduler_func(model_trial, **scheduler_options)
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading data to remote workers...")
            scheduler.upload_files([train_path, val_path]) # TODO: currently does not work.
            directory = self.path # TODO: need to change to path to working directory used on every remote machine
            model_trial.update(directory=directory)
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
                       'args': model_trial.args
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

        logger.log(15, "Time for %s model HPO: %s" % (self.name, str(hpo_results['total_time'])))
        self.params.update(best_hp)
        # TODO: reload model params from best trial? Do we want to save this under cls.model_file as the "optimal model"
        logger.log(15, "Best hyperparameter configuration for %s model: " % self.name)
        logger.log(15, str(best_hp))
        return (hpo_models, hpo_model_performances, hpo_results)
        # TODO: do final fit here?
        # args.final_fit = True
        # final_model = scheduler.run_with_config(best_config)
        # save(final_model)

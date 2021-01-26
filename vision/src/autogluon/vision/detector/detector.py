"""Object Detection task"""
import copy
import pickle
import logging
import warnings

from autogluon.core.utils import verbosity2loglevel, get_gpu_free_memory, get_gpu_count
from gluoncv.auto.tasks import ObjectDetection as _ObjectDetection

__all__ = ['ObjectDetector']

class ObjectDetector(object):
    """AutoGluon Predictor for detecting objects in images

    Parameters
    ----------
    path : str, default = None
        The directory for saving logs or intermediate data. If unspecified, will create a sub-directory under
        current working directory.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed. 
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings). 
        If using logging, you can alternatively control amount of information printed via logger.setLevel(L), 
        where L ranges from 0 to 50 (Note: higher values of L correspond to fewer print statements, opposite of verbosity levels)
    """
    # Dataset is a subclass of `pd.DataFrame`, with `image` and `bbox` columns.
    Dataset = _ObjectDetection.Dataset

    def __init__(self, path=None, verbosity=2):
        self._log_dir = path
        self._verbosity = verbosity
        self._detector = None
        self._fit_summary = {}

    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            num_trials=1,
            hyperparameters=None,
            **kwargs):
        """Automatic fit process for object detection.

        Parameters
        ----------
        train_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            For more details of how to construct a object detection dataset, please checkout:
            `http://preview.d2l.ai/d8/main/object_detection/getting_started.html`.
            If a string is provided, will search for k8 datasets.
        tuning_data : pd.DataFrame or str, default = None
            Holdout tuning data for validation, reserved for model selection and hyperparameter-tuning,
            can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
            If `None`, the validation dataset will be randomly split from `train_data` according to `holdout_frac`.
        time_limit : int, default = None
            Time limit in seconds, if not specified, will run until all tuning and training finished.
            If `time_limit` is hit during `fit`, the
            HPO process will interrupt and return the current best configuration.
        hyperparameters : dict, default = None
            Extra hyperparameters for specific models.
            Accepted args includes(not limited to):
            epochs : int, default value based on network
                The `epochs` for model training.
            batch_size : int
                Mini batch size
            lr : float
                Trainer learning rate for optimization process.
            You can get the list of accepted hyperparameters in `config.yaml` saved by this predictor.
        **kwargs :
            holdout_frac : float, default = 0.1
                The random split ratio for `tuning_data` if `tuning_data==None`.
            random_state : int, default = None
                The random_state(seed) for shuffling data, only used if `tuning_data==None`.
                Note that the `random_state` only affect the splitting process, not model training.
                If not specified(None), will leave the original random sampling intact.
            nthreads_per_trial : int, default = (# cpu cores)
                Number of CPU threads for each trial, if `None`, will detect the # cores on current instance.
            ngpus_per_trial : int, default = (# gpus)
                Number of GPUs to use for each trial, if `None`, will detect the # gpus on current instance.
            hyperparameter_tune_kwargs: dict, default = None
                num_trials : int, default = 1
                    The number of HPO trials when `time_limit` is None. 
                    If `time_limit` is set, `num_trials` will be overwritten with an infinite large number.
                search_strategy : str, default = 'random'
                    Searcher strategy for HPO, 'random' by default.
                    Options include: ‘random’ (random search), ‘bayesopt’ (Gaussian process Bayesian optimization),
                    ‘grid’ (grid search).
                max_reward : float, default = 0.9
                    The reward threashold for stopping criteria. If `max_reward` is reached during HPO, the scheduler
                    will terminate earlier to reduce time cost.
                scheduler_options : dict, default = None
                    Extra options for HPO scheduler, please refer to `autogluon.core.Searcher` for details.
        """
        log_level = verbosity2loglevel(self._verbosity)
        if self._detector is not None:
            self._detector._logger.setLevel(log_level)
            self._detector._logger.propagate = True
            self._fit_summary = self._detector.fit(train_data, tuning_data, 1 - holdout_frac, random_state, resume=False)
            return

        # init/validate kwargs
        kwargs = self._validate_kwargs(kwargs)
        # unpack
        num_trials = kwargs['hyperparameter_tune_kwargs']['num_trials']
        nthreads_per_trial = kwargs['nthreads_per_trial']
        ngpus_per_trial = kwargs['ngpus_per_trial']
        holdout_frac = kwargs['holdout_frac']
        random_state = kwargs['random_state']
        search_strategy = kwargs['hyperparameter_tune_kwargs']['search_strategy']
        max_reward = kwargs['hyperparameter_tune_kwargs']['max_reward']
        scheduler_options = kwargs['hyperparameter_tune_kwargs']['scheduler_options']

        # new HPO task
        if time_limit is not None:
            num_trials = 99999
        if time_limit is None and num_trials is None:
            raise ValueError("`time_limit` and kwargs['hyperparameter_tune_kwargs']['num_trials'] can not be `None` at the same time, "
                             "otherwise the training will not be terminated gracefully.")
        config={'log_dir': self._log_dir,
                'num_trials': 99999 if num_trials is None else max(1, num_trials),
                'time_limits': 2147483647 if time_limit is None else max(1, time_limit),
                'search_strategy': search_strategy,
                'max_reward': max_reward,
                }
        if nthreads_per_trial is not None:
            config['nthreads_per_trial'] = nthreads_per_trial
        if ngpus_per_trial is not None:
            config['ngpus_per_trial'] = ngpus_per_trial
        if isinstance(hyperparameters, dict):
            if 'batch_size' in hyperparameters:
                bs = hyperparameters['batch_size']
                if ngpus_per_trial is not None and ngpus_per_trial > 1 and bs > 16:
                    # using gpus, check batch size vs. available gpu memory
                    free_gpu_memory = get_gpu_free_memory()
                    if not free_gpu_memory:
                        warnings.warn('Unable to detect free GPU memory, we are unable to verify '
                                      'whether your data mini-batches will fit on the GPU for the specified batch_size.')
                    elif len(free_gpu_memory) < ngpus_per_trial:
                        warnings.warn(f'Detected GPU memory for {len(free_gpu_memory)} gpus but {ngpus_per_trial} is requested.')
                    elif sum(free_gpu_memory[:ngpus_per_trial]) / bs < 1280:
                        warnings.warn(f'batch_size: {bs} is potentially larger than what your gpus can support ' +
                                      f'free memory: {free_gpu_memory[:ngpus_per_trial]} ' +
                                      'Try reducing "batch_size" if you encounter memory issues.')
            # check if hyperparameters overwriting existing config
            for k, v in hyperparameters.items():
                if k in config:
                    raise ValueError(f'Overwriting {k} = {config[k]} to {v} by hyperparameters is ambiguous.')
            config.update(hyperparameters)
        if scheduler_options is not None:
            config.update(scheduler_options)
        # verbosity
        if log_level > logging.INFO:
            logging.getLogger('gluoncv.auto.tasks.object_detection').propagate = False
            for logger_name in ('SSDEstimator', 'CenterNetEstimator', 'YOLOv3Estimator', 'FasterRCNNEstimator'):
                logging.getLogger(logger_name).setLevel(log_level)
                logging.getLogger(logger_name).propagate = False
        task = _ObjectDetection(config=config)
        task._logger.setLevel(log_level)
        task._logger.propagate = True
        self._detector = task.fit(train_data, tuning_data, 1 - holdout_frac, random_state)
        self._detector._logger.setLevel(log_level)
        self._detector._logger.propagate = True
        self._fit_summary = task.fit_summary()

    def _validate_kwargs(self, kwargs):
        """validate and initialize default kwargs"""
        kwargs['holdout_frac'] = kwargs.get('holdout_frac', 0.1)
        if not (0 < kwargs['holdout_frac'] < 1.0):
            raise ValueError(f'Range error for `holdout_frac`, expected to be within range (0, 1), given {kwargs["holdout_frac"]}')
        kwargs['random_state'] = kwargs.get('random_state', None)
        kwargs['nthreads_per_trial'] = kwargs.get('nthreads_per_trial', None)
        kwargs['ngpus_per_trial'] = kwargs.get('ngpus_per_trial', None)
        if kwargs['ngpus_per_trial'] > 0:
            detected_gpu = get_gpu_count()
            if detected_gpu < kwargs['ngpus_per_trial']:
                raise ValueError(f"Insufficient detected # gpus {detected_gpu} vs requested {kwargs['ngpus_per_trial']}")
        # tune kwargs
        hpo_tune_args = kwargs.get('hyperparameter_tune_kwargs', {})
        hpo_tune_args['num_trials'] = hpo_tune_args.get('num_trials', 1)
        hpo_tune_args['search_strategy'] = hpo_tune_args.get('search_strategy', 'random')
        if not hpo_tune_args['search_strategy'] in ('random', 'bayesopt', 'grid'):
            raise ValueError(f"Invalid search strategy: {hpo_tune_args['search_strategy']}, supported: ('random', 'bayesopt', 'grid')")
        hpo_tune_args['max_reward'] = hpo_tune_args.get('max_reward', 0.9)
        if hpo_tune_args['max_reward'] < 0:
            raise ValueError(f"Expected `max_reward` to be a positive float number between 0 and 1.0, given hpo_tune_args['max_reward']")
        hpo_tune_args['scheduler_options'] = hpo_tune_args.get('scheduler_options', None)
        kwargs['hyperparameter_tune_kwargs'] = hpo_tune_args
        return kwargs

    def predict(self, data, as_pandas=True):
        """Predict objects in image, return the confidences, bounding boxes of each predicted object.

        Parameters
        ----------
        data : str, pd.DataFrame or ndarray
            The input data, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.
        as_pandas : bool, default = True
            Whether to return the output as a pandas object (True) or list of numpy array(s) (False).
            Pandas object is a DataFrame.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain (`pred_score`, `pred_bbox`, `pred_id`).
            If more than one image in input, the returned dataframe will contain `images` column,
            and all results are concatenated.
        """
        if self._detector is None:
            raise RuntimeError('Detector is not initialized, try `fit` first.')
        ret = self._detector.predict(data)
        if as_pandas:
            return ret
        else:
            return ret.as_numpy()

    def evaluate(self, data):
        """Evaluate model performance on validation data.

        Parameters
        ----------
        data : pd.DataFrame or iterator
            The validation data.
        """
        if self._detector is None:
            raise RuntimeError('Detector not initialized, try `fit` first.')
        return self._detector.evaluate(data)

    def fit_summary(self):
        """Return summary of last `fit` process.

        Returns
        -------
        dict
            The summary of last `fit` process. Major keys are ('train_map', 'val_map', 'total_time',...)

        """
        return copy.copy(self._fit_summary)

    def save(self, file_name):
        """Dump predictor to disk.

        Parameters
        ----------
        file_name : str
            The file name of saved copy.

        """
        with open(file_name, 'wb') as fid:
            pickle.dump(self, fid)

    @classmethod
    def load(cls, file_name, verbosity=2):
        """Load previously saved predictor.

        Parameters
        ----------
        file_name : str
            The file name for saved pickle file.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed. 
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings). 
            If using logging, you can alternatively control amount of information printed via logger.setLevel(L), 
            where L ranges from 0 to 50 (Note: higher values of L correspond to fewer print statements, opposite of verbosity levels)

        """
        with open(file_name, 'rb') as fid:
            obj = pickle.load(fid)
        obj._verbosity = verbosity
        return obj

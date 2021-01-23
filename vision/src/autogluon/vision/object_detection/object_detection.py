"""Object Detection task"""
import copy
import pickle
import logging

from autogluon.core.utils import verbosity2loglevel
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
            holdout_frac=0.1,
            random_state=None,
            time_limit=None,
            num_trials=1,
            hyperparameters=None,
            search_strategy='random',
            scheduler_options=None,
            num_cpus=None,
            num_gpus=None):
        """Automatic fit process for object detection.

        Parameters
        ----------
        train_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            For more details of how to construct a object detection dataset, please checkout:
            `http://preview.d2l.ai/d8/main/object_detection/getting_started.html`.
            If a string is provided, will search for k8 datasets.
        tuning_data : pd.DataFrame or str, default = None
            Holdout tuning data for validation, can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
            If `None`, the validation dataset will be randomly split from `train_data` according to `holdout_frac`.
        holdout_frac : float, default = 0.1
            The random split ratio for `tuning_data` if `tuning_data==None`.
        random_state : numpy.random.state, default = None
            The random_state for shuffling, only used if `tuning_data==None`.
            Note that the `random_state` only affect the splitting process, not model training.
        time_limit : int, default = None
            Time limit in seconds, if not specified, will run until all tuning and training finished.
            If `time_limit` is hit during `fit`, the
            HPO process will interrupt and return the current best configuration.
        num_trials : int, default = 1
            The number of HPO trials. If `None`, will run infinite trials until `time_limit` is met.
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
        search_strategy : str, default = 'random'
            Searcher strategy for HPO, 'random' by default.
            Options include: ‘random’ (random search), ‘bayesopt’ (Gaussian process Bayesian optimization),
            ‘skopt’ (SKopt Bayesian optimization), ‘grid’ (grid search).
        scheduler_options : dict, default = None
            Extra options for HPO scheduler, please refer to `autogluon.core.Searcher` for details.
        num_cpus : int, default = (# cpu cores)
            Number of CPU threads for each trial, if `None`, will detect the # cores on current instance.
        num_gpus : int, default = (# gpus)
            Number of GPUs to use for each trial, if `None`, will detect the # gpus on current instance.
        """
        log_level = verbosity2loglevel(self._verbosity)
        if self._detector is not None:
            self._detector._logger.setLevel(log_level)
            self._detector._logger.propagate = True
            self._fit_summary = self._detector.fit(train_data, tuning_data, 1 - holdout_frac, random_state, resume=False)
            return

        # new HPO task
        config={'log_dir': self._log_dir,
                'num_trials': 99999 if num_trials is None else max(1, num_trials),
                'time_limits': time_limit,
                'search_strategy': search_strategy,
                }
        if num_cpus is not None:
            config['nthreads_per_trial'] = num_cpus
        if num_gpus is not None:
            config['ngpus_per_trial'] = num_gpus
        if isinstance(hyperparameters, dict):
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

    def predict(self, x):
        """Predict objects in image, return the confidences, bounding boxes of each predicted object.

        Parameters
        ----------
        x : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain (`pred_score`, `pred_bbox`, `pred_id`).
            If more than one image in input, the returned dataframe will contain `images` column,
            and all results are concatenated.
        """
        if self._detector is None:
            raise RuntimeError('Detector is not initialized, try `fit` first.')
        return self._detector.predict(x)

    def evaluate(self, tuning_data):
        """Evaluate model performance on validation data.

        Parameters
        ----------
        tuning_data : pd.DataFrame or iterator
            The validation data.
        """
        if self._detector is None:
            raise RuntimeError('Detector not initialized, try `fit` first.')
        return self._detector.evaluate(tuning_data)

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
    def load(cls, file_name):
        """Load previously saved predictor.

        Parameters
        ----------
        file_name : str
            The file name for saved pickle file.

        """
        with open(file_name, 'rb') as fid:
            obj = pickle.load(fid)
        return obj

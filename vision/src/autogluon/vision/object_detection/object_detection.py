"""Object Detection task"""
import copy
import pickle
import logging

from autogluon.core.utils import verbosity2loglevel
from gluoncv.auto.tasks import ObjectDetection as _ObjectDetection

__all__ = ['ObjectDetector']

class ObjectDetector(object):
    """AutoGluon Predictor for for detecting objects in images

    Parameters
    ----------
    log_dir : str
        The directory for saving logs, by default using `pwd`: the current working directory.

    """
    # Dataset is a subclass of `pd.DataFrame`, with `image` and `bbox` columns.
    Dataset = _ObjectDetection.Dataset

    def __init__(self, log_dir=None):
        self._log_dir = log_dir
        self._detector = None
        self._fit_summary = {}

    def fit(self,
            train_data,
            val_data=None,
            holdout_frac=0.1,
            random_state=None,
            time_limit=12*60*60,
            num_trials=1,
            hyperparameters=None,
            search_strategy='random',
            scheduler_options=None,
            nthreads_per_trial=None,
            ngpus_per_trial=None,
            dist_ip_addrs=None,
            verbosity=3):
        """Automatic fit process for object detection.

        Parameters
        ----------
        train_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            For more details of how to construct a object detection dataset, please checkout:
            `http://preview.d2l.ai/d8/main/object_detection/getting_started.html`.
            If a string is provided, will search for k8 datasets.
        val_data : pd.DataFrame or str, default = None
            Training data, can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
            If `None`, the validation dataset will be randomly split from `train_data`.
        holdout_frac : float, default = 0.1
            The random split ratio for `val_data` if `val_data==None`.
        random_state : numpy.random.state, default = None
            The random_state for shuffling, only used if `val_data==None`.
            Note that the `random_state` only affect the splitting process, not model training.
        time_limit : int, default = 43200
            Time limit in seconds, default is 12 hours. If `time_limit` is hit during `fit`, the
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
        nthreads_per_trial : int, default = (# cpu cores)
            Number of CPU threads for each trial, if `None`, will detect the # cores on current instance.
        ngpus_per_trial : int, default = (# gpus)
            Number of GPUs to use for each trial, if `None`, will detect the # gpus on current instance.
        dist_ip_addrs : list, default = None
            If not `None`, will spawn tasks on distributed nodes.
        verbosity : int, default = 3
            Controls how detailed of a summary to ouput.
            Set <= 0 for no output printing, 1 to print just high-level summary,
            2 to print summary and create plots, >= 3 to print all information produced during fit().
        """
        log_level = verbosity2loglevel(verbosity)
        if self._detector is not None:
            self._detector._logger.setLevel(log_level)
            self._detector._logger.propagate = True
            self._fit_summary = self._detector.fit(train_data, val_data, 1 - holdout_frac, random_state, resume=False)
            return

        # new HPO task
        config={'log_dir': self._log_dir,
                'num_trials': 99999 if num_trials is None else max(1, num_trials),
                'time_limits': time_limit,
                'search_strategy': search_strategy,
                }
        if nthreads_per_trial is not None:
            config['nthreads_per_trial'] = nthreads_per_trial
        if ngpus_per_trial is not None:
            config['ngpus_per_trial'] = ngpus_per_trial
        if dist_ip_addrs is not None:
            config['dist_ip_addrs'] = dist_ip_addrs
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
        self._detector = task.fit(train_data, val_data, 1 - holdout_frac, random_state)
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

    def evaluate(self, val_data):
        """Evaluate model performance on validation data.

        Parameters
        ----------
        val_data : pd.DataFrame or iterator
            The validation data.
        """
        if self._detector is None:
            raise RuntimeError('Detector not initialized, try `fit` first.')
        return self._detector.evaluate(val_data)

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

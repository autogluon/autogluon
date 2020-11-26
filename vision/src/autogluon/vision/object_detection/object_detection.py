"""Object Detection task"""
import copy
import pickle

from gluoncv.auto.tasks import ObjectDetection as _ObjectDetection

__all__ = ['ObjectDetection']

class ObjectDetection(object):
    """AutoGluon Predictor for for detecting objects in images

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.

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
            train_size=0.9,
            random_state=None,
            time_limit=12*60*60,
            epochs=None,
            num_trials=None,
            hyperparameters=None,
            search_strategy='random',
            scheduler_options=None,
            nthreads_per_trial=None,
            ngpus_per_trial=None,
            dist_ip_addrs=None):
        """Automatic fit process.

        Parameters
        ----------
        train_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
        val_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
            If `None`, the validation dataset will be randomly split from `train_data`.
        train_size : float
            The random split ratio for `train_data` if `val_data==None`.
            The new `val_data` size will be `1-train_size`.
        random_state : numpy.random.state
            The random_state for shuffling, only used if `val_data==None`.
            Note that the `random_state` only affect the splitting process, not model training.
        time_limit : int
            Time limit in seconds, default is 12 hours. If `time_limit` is hit during `fit`, the
            HPO process will interupt and return the current best configuration.
        epochs : int
            The `epochs` for model training, if `None` is provided, then default `epochs` for model
            will be used.
        num_trials : int, default is 1
            The number of HPO trials. If `None`, will run only one trial.
        hyperparameters : dict
            Extra hyperparameters for specific models.
        search_strategy : str
            Searcher strategy for HPO, 'random' by default.
        scheduler_options : dict
            Extra options for HPO scheduler, please refer to `autogluon.Searcher` for details.
        nthreads_per_trial : int
            Number of CPU threads for each trial, if `None`, will detect the # cores on current instance.
        ngpus_per_trial : int
            Number of GPUs to use for each trial, if `None`, will detect the # gpus on current instance.
        dist_ip_addrs : list
            If not `None`, will spawn tasks on distributed nodes.

        """
        if self._detector is not None:
            self._fit_summary = self._detector.fit(train_data, val_data, train_size, random_state, resume=False)
            return

        # new HPO task
        config={'log_dir': self._log_dir,
                'num_trials': 1 if num_trials is None else num_trials,
                'time_limits': time_limit,
                'search_strategy': search_strategy,
                }
        if nthreads_per_trial is not None:
            config.update({'nthreads_per_trial': nthreads_per_trial})
        if ngpus_per_trial is not None:
            config.update({'ngpus_per_trial': ngpus_per_trial})
        if dist_ip_addrs is not None:
            config.update({'dist_ip_addrs': dist_ip_addrs})
        if epochs is not None:
            config.update({'epochs': epochs})
        if isinstance(hyperparameters, dict):
            config.update(hyperparameters)
        if scheduler_options is not None:
            config.update(scheduler_options)
        task = _ObjectDetection(config=config)
        self._detector = task.fit(train_data, val_data, train_size, random_state)
        self._fit_summary = task.fit_summary()
        return self

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

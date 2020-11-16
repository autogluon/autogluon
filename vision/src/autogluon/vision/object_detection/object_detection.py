"""Object Detection task"""
import copy
import pickle

from gluoncv.auto.tasks import ObjectDetection as _ObjectDetection

__all__ = ['Predictor']

class ObjectDetectionPredictor(object):
    """AutoGluon Predictor for for detecting objects in images

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.

    """
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
            time_limit=None,
            num_trials=None,
            hyperparameters=None,
            search_strategy='random',
            scheduler_options=None,
            nthreads_per_trial=None,
            ngpus_per_trial=None,
            dist_ip_addrs=None):
        """Fit auto estimator given the input data .


        """
        if self._detector is not None:
            self._fit_summary = self._detector.fit(train_data, val_data, train_size, random_state, resume=False)
            return

        # new HPO task
        config={'log_dir': self._log_dir,
                'num_trials': 1 if num_trials is None else num_trials,
                'time_limits': time_limit,
                'search_strategy': search_strategy,
                'nthreads_per_trial': nthreads_per_trial,
                'ngpus_per_trial': ngpus_per_trial,
                'dist_ip_addrs': dist_ip_addrs
                }
        if isinstance(hyperparameters, dict):
            config.update(hyperparameters)
        if scheduler_options is not None:
            config.update(scheduler_options)
        task = _ObjectDetection(config=config)
        self._detector = task.fit(train_data, val_data, train_size, random_state)
        self._fit_summary = task.fit_summary()

    def predict(self, x):
        if self._detector is None:
            raise RuntimeError('Detector is not initialized, try `fit` first.')
        return self._detector.predict(x)

    def evaluate(self, val_data):
        if self._detector is None:
            raise RuntimeError('Detector not initialized, try `fit` first.')
        return self._detector.evaluate(x)

    def fit_summary(self):
        return copy.copy(self._fit_summary)

    def save(self, file_name):
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

Predictor = ObjectDetectionPredictor

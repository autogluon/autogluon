"""Image classification task"""

from gluoncv.auto.tasks import ImageClassification as _ImageClassification

__all__ = ['ImageClassification']

class ImageClassification(object):
    """AutoGluon Task for classifying images based on their whole contents

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.

    """
    Dataset = _ImageClassification.Dataset
    def __init__(self, log_dir=None):
        self._log_dir = log_dir
        self._predictor = None

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
        config={'log_dir': self._log_dir,
                'num_trials': 1 if num_trials is None else num_trials,
                'time_limits': time_limit,
                'search_strategy': search_strategy,
                'nthreads_per_trial': nthreads_per_trial,
                'ngpus_per_trial': ngpus_per_trial,
                'dist_ip_addrs': dist_ip_addrs
                }
        if hyperparameters is not None:
            config.update(hyperparameters)
        if scheduler_options is not None:
            config.update(scheduler_options)
        task = _ImageClassification(config=config)
        self._predictor = task.fit(train_data, val_data, train_size, random_state)

    def predict(self, x):
        if self._predictor is None:
            raise RuntimeError('Predictor not initialized, try `fit` first.')
        return self._predictor.predict(x)

    @classmethod
    def load(cls, filename):
        """Load previously saved trainer.

        Parameters
        ----------
        filename : str
            The file name for saved pickle file.

        """
        return super().load(filename)

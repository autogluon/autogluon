"""Object Detection task"""

from gluoncv.auto.tasks import ObjectDetection as _ObjectDetection

__all__ = ['ObjectDetection']

class ObjectDetection(_ObjectDetection):
    """AutoGluon Task for detecting objects in images

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.

    """
    def __init__(self, config=None, logger=None):
        super().__init__(config=config, logger=logger)

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None):
        """Fit auto estimator given the input data .

        Returns
        -------
        Estimator
            The estimator obtained by training on the specified dataset.

        """
        super().fit(train_data, val_data, train_size, random_state)

    @classmethod
    def load(cls, filename):
        """Load previously saved trainer.

        Parameters
        ----------
        filename : str
            The file name for saved pickle file.

        """
        return super().load(filename)

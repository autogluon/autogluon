
import logging

from autogluon.core.dataset import TabularDataset
from autogluon.core.task.base import BaseTask

from .predictor_legacy import TabularPredictorV1

__all__ = ['TabularPrediction']

logger = logging.getLogger()  # return root logger


# TODO v0.1: Remove
class TabularPrediction(BaseTask):
    """
    AutoGluon Task for predicting values in column of tabular dataset (classification or regression)
    """

    Dataset = TabularDataset
    Predictor = TabularPredictorV1

    @staticmethod
    def load(output_directory, verbosity=2) -> TabularPredictorV1:
        """
        Load a predictor object previously produced by `fit()` from file and returns this object.
        It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.

        Parameters
        ----------
        output_directory : str
            Path to directory where trained models are stored (i.e. the output_directory specified in previous call to `fit`).
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information will be printed by the loaded `Predictor`.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where L ranges from 0 to 50 (Note: higher values L correspond to fewer print statements, opposite of verbosity levels)

        Returns
        -------
        :class:`autogluon.task.tabular_prediction.TabularPredictor` object that can be used to make predictions.
        """
        return TabularPredictorV1.load(output_directory=output_directory, verbosity=verbosity)

    @staticmethod
    def fit(**kwargs):
        """
        Removed in favor of :class:`autogluon.tabular.TabularPredictor`
        """
        raise AssertionError(
            'CRITICAL: TabularPrediction has been removed and has been replaced by TabularPredictor.\n'
            'Please use `autogluon.tabular.TabularPredictor` instead and refer to documentation for tutorials:\n'
            'https://auto.gluon.ai/dev/tutorials/tabular_prediction/tabular-quickstart.html \n'
            'https://auto.gluon.ai/dev/tutorials/tabular_prediction/tabular-indepth.html \n'
            'https://auto.gluon.ai/dev/api/autogluon.task.html#autogluon.tabular.TabularPredictor \n'
            'https://auto.gluon.ai/dev/api/autogluon.task.html#autogluon.tabular.TabularPredictor.fit \n'
            'https://auto.gluon.ai/dev/api/autogluon.task.html#tabulardataset'
        )

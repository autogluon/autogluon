"""Text Prediction Model based on BERT"""
from typing import Union
from ..abstract.abstract_model import AbstractModel
from ... import metrics
from ...features.feature_metadata import FeatureMetadata

# Import autogluon text specific dependencies
try:
    import autogluon.text
    from autogluon.text import TextPrediction
except ImportError:
    raise ImportError('autogluon.text has not been installed. '
                      'You may try to install "autogluon.text" first by running. '
                      '`python3 -m pip install autogluon.text`')


class BertTextPredictionV1Model(AbstractModel):
    def __init__(self, path: str, name: str, problem_type: str,
                 eval_metric: Union[str, metrics.Scorer] = None,
                 num_classes=None, stopping_metric=None, model=None,
                 hyperparameters=None, features=None,
                 feature_metadata: FeatureMetadata = None, debug=0, **kwargs):
        """The BertTextPredictionV1Model

        Parameters
        ----------
        path
            The path of the model
        name
            Directory where to store all outputs.
        problem_type
            Type of problem this model will handle.
            Valid options: ['binary', 'multiclass', 'regression'].
        eval_metric
            The evaluation metric.
        num_classes
            The number of classes
        stopping_metric
            The stopping metric.
        model
            The model
        hyperparameters
            The hyperparameters of the model
        features
            The features
        feature_metadata
            The feature metadata
        debug
            Whether to turn on debug mode
        kwargs
            Other arguments
        """
        super().__init__(path=path, name=name, problem_type=problem_type, eval_metric=eval_metric,
                         num_classes=num_classes, stopping_metric=stopping_metric, model=model,
                         hyperparameters=hyperparameters, features=features,
                         feature_metadata=feature_metadata, debug=debug, **kwargs)



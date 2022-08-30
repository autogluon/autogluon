"""Wrapper of the MultiModalPredictor."""
import logging
from .automm_model import MultiModalPredictorModel

logger = logging.getLogger(__name__)

# TODO: Add unit tests
class FTTransformerModel(MultiModalPredictorModel):
    def __init__(self, **kwargs):
        """Wrapper of autogluon.multimodal.MultiModalPredictor.

        The features can be a mix of
        - categorical column
        - numerical column

        The labels can be categorical or numerical.

        Parameters
        ----------
        path
            The directory to store the modeling outputs.
        name
            Name of subdirectory inside path where model will be saved.
        problem_type
            Type of problem that this model will handle.
            Valid options: ['binary', 'multiclass', 'regression'].
        eval_metric
            The evaluation metric.
        num_classes
            The number of classes.
        stopping_metric
            The stopping metric.
        model
            The internal model object.
        hyperparameters
            The hyperparameters of the model
        features
            Names of the features.
        feature_metadata
            The feature metadata.
        """
        super().__init__(**kwargs)

    def _set_default_params(self):
        default_params = {
            "data.categorical.convert_to_text": False,
            "model.names": ["categorical_transformer", "numerical_transformer", "fusion_transformer"],
            "model.numerical_transformer.embedding_arch": ["linear"],
            "env.batch_size": 128,
            "optimization.max_epochs": 2000,  # Specify a large value to train until convergence
            "optimization.weight_decay": 1.0e-5,
            "optimization.lr_choice": None,
            "optimization.lr_schedule": "polynomial_decay",
            "optimization.warmup_steps": 0.0,
            "optimization.patience": 20,
            "optimization.top_k": 3,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

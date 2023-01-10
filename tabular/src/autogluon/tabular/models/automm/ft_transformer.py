"""Wrapper of the MultiModalPredictor."""
import logging
from typing import Dict

from autogluon.common.features.types import R_INT, R_FLOAT, R_CATEGORY, \
    S_TEXT_NGRAM, S_TEXT_SPECIAL

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

    def _fit(self, X, num_gpus='auto', **kwargs):
        if not isinstance(num_gpus, str):
            if num_gpus == 0:
                logger.log(30, f'WARNING: Training {self.name} on CPU (no GPU specified). This could take a long time. Use GPU to speed up training.')
        super()._fit(X, num_gpus=num_gpus, **kwargs)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_SPECIAL],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _set_default_params(self):
        default_params = {
            "data.categorical.convert_to_text": False,
            "model.names": ["categorical_transformer", "numerical_transformer", "fusion_transformer"],
            "model.numerical_transformer.embedding_arch": ["linear"],
            "env.batch_size": 128,
            "env.per_gpu_batch_size": 128,
            "env.num_workers": 0,
            "env.num_workers_evaluation": 0,
            "optimization.max_epochs": 2000,  # Specify a large value to train until convergence
            "optimization.weight_decay": 1.0e-5,
            "optimization.lr_choice": None,
            "optimization.lr_schedule": "polynomial_decay",
            "optimization.warmup_steps": 0.0,
            "optimization.patience": 20,
            "optimization.top_k": 3,
            '_max_features': 300,  # FIXME: This is a hack, move to AG_ARGS_FIT for v0.7
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def get_minimum_resources(self, is_gpu_available=False) -> Dict[str, int]:
        return {
            "num_cpus": 1,
            "num_gpus": 0,  # allow FT_Transformer to be trained on CPU only
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            'fold_fitting_strategy': 'auto',
            'fold_fitting_strategy_gpu': 'sequential_local',  # Crashes when using GPU in parallel bagging
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls):
        return {'handles_text': False}

"""Wrapper of the MultiModalPredictor."""

from __future__ import annotations

import logging
from typing import Dict

from autogluon.common.features.types import R_CATEGORY, R_FLOAT, R_INT, S_TEXT_NGRAM, S_TEXT_SPECIAL

from .automm_model import MultiModalPredictorModel

logger = logging.getLogger(__name__)


# TODO: Add unit tests
class FTTransformerModel(MultiModalPredictorModel):
    ag_key = "FT_TRANSFORMER"
    ag_name = "FTTransformer"

    _default_auxiliary_params_extra = dict(
        valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY],
        ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_SPECIAL],
    )
    minimum_num_gpus = 0  # allow FT_Transformer to be trained on CPU only
    gpu_required = False
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "auto",
        "fold_fitting_strategy_gpu": "sequential_local",  # Crashes when using GPU in parallel bagging
    }

    def _fit(self, X, num_gpus="auto", **kwargs):
        if not isinstance(num_gpus, str):
            if num_gpus == 0:
                logger.log(
                    30,
                    f"WARNING: Training {self.name} on CPU (no GPU specified). This could take a long time. Use GPU to speed up training.",
                )
        super()._fit(X, num_gpus=num_gpus, **kwargs)

    def _set_default_params(self):
        default_params = {
            "data.categorical.convert_to_text": False,
            "model.names": ["ft_transformer"],
            "model.ft_transformer.embedding_arch": ["linear"],
            "env.batch_size": 128,
            "env.per_gpu_batch_size": 128,
            "env.num_workers": 0,
            "env.num_workers_inference": 0,
            "optim.max_epochs": 2000,  # Specify a large value to train until convergence
            "optim.weight_decay": 1.0e-5,
            "optim.lr_choice": None,
            "optim.lr_schedule": "polynomial_decay",
            "optim.warmup_steps": 0.0,
            "optim.patience": 20,
            "optim.top_k": 3,
            "_max_features": 300,  # FIXME: This is a hack, move to AG_ARGS_FIT for v0.7
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def _class_tags(cls):
        return {"handles_text": False}

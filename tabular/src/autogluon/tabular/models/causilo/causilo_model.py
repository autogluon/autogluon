from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from autogluon.core.models.abstract.shared_weights import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from ._weight_fetch import weight_fetch_policy

logger = logging.getLogger(__name__)

_HAS_LOGGED_CAUSILO_NONCOMMERCIAL: bool = False


class CausiloModel(AbstractTorchModel):
    """Causilo is a pretrained tabular foundation model by Nums AI that predicts via in-context learning.

    Codebase: https://github.com/nums-ai/causilo (Apache-2.0)
    Weights: https://huggingface.co/nums-ai/causilo (Causilo License v1.0, non-commercial)

    Label sets wider than the ten-class head are fit through the library's own
    error-correcting output codes, so no ``max_classes`` cap is set.

    .. versionadded:: 1.6.4
    """

    ag_key = "CAUSILO"
    ag_name = "Causilo"
    ag_priority = 40
    seed_name = "random_state"
    gpu_strongly_recommended: bool = True  # in-context inference is an order of magnitude slower on CPU
    _supported_problem_types = ["binary", "multiclass", "regression", "quantile"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"]}
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,  # Refit on the full data for faster inference and similar quality as the bag.
    }
    """Set fold_fitting_strategy to sequential_local, as parallel folding crashes if model weights aren't pre-downloaded."""

    #: ``Engine.fit`` loads the network through ``causilo.checkpoints.load_pretrained_model(task)`` and
    #: moves it to the engine's device; one build per task and device per process. The loader has no
    #: device input, so ``_fit`` records the device on ``self.device`` before the fit.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="causilo.checkpoints:load_pretrained_model", key=("task",)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inference_threads: int = 1

    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, verbosity: int = 2, **kwargs):
        import torch
        from causilo import CausiloClassifier, CausiloRegressor

        if verbosity >= 2:
            self._log_license()
        # Context fitting has no iterative training, early stopping or validation split.
        params = self._get_model_params().copy()
        device = self._resolve_fit_device(num_gpus=num_gpus)
        is_regression = self.problem_type in ["regression", "quantile"]
        model_cls = CausiloRegressor if is_regression else CausiloClassifier
        self._inference_threads = num_cpus
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_cpus)
            self.device = device  # keys the shared network; the library's loader names no device
            self.model = model_cls(device=device, **params)
            with weight_fetch_policy(self.aux_params.fetch_pretrained_weights, stage="fit", model_name=self.name):
                self.model.fit(self.preprocess(X), y)
        finally:
            torch.set_num_threads(previous_threads)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        import torch

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._inference_threads)
            if self.problem_type == "quantile":
                return self.model.predict(
                    self.preprocess(X, **kwargs), output_type="quantiles", quantiles=self.quantile_levels
                )
            return super()._predict_proba(X, **kwargs)
        finally:
            torch.set_num_threads(previous_threads)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Load the pickle, applying the load-stage weight-fetch policy.

        The fitted model pickles without the network, which is put back while loading -- a network
        fetch on a host with a cold cache. That happens inside ``super().load``, before the model's
        ``aux_params`` can be read, so the only policy source is the environment
        (``AG_FETCH_PRETRAINED_WEIGHTS``), which is also the right scope for an inference-time policy.
        """
        with weight_fetch_policy(True, stage="load", model_name=cls.__name__):
            return super().load(path=path, reset_paths=reset_paths, verbose=verbose)

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", 8)

    def get_device(self) -> str:
        return str(self.model._engine.device)

    def _set_device(self, device: str):
        """Move the network and the fitted attention caches, then point the estimator at the device."""
        from dataclasses import replace

        from causilo.engine import resolve_device
        from causilo.execution.memory import Stage
        from causilo.execution.precision import stage_dtype
        from causilo.serialization import transfer_state

        engine = self.model._engine
        target = resolve_device(str(device))
        engine.model.to(target)
        state = engine.state
        if state.caches is not None or state.code_caches is not None:
            column_dtype = stage_dtype(engine.task, Stage.COLUMN, target)
            prediction_dtype = stage_dtype(engine.task, Stage.PREDICTION, target)

            def moved(caches):
                return tuple(
                    replace(
                        cache,
                        columns=tuple(transfer_state(c, target, dtype=column_dtype) for c in cache.columns),
                        prediction=transfer_state(cache.prediction, target, dtype=prediction_dtype),
                    )
                    for cache in caches
                )

            if state.caches is not None:
                state = replace(state, caches=moved(state.caches))
            if state.code_caches is not None:  # one cache tuple per output-code row of a many-class fit
                state = replace(state, code_caches=tuple(moved(caches) for caches in state.code_caches))
            engine.state = state
        engine.device = target
        engine.device_request = str(target)
        self.model.device = str(target)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def _log_license(self):
        global _HAS_LOGGED_CAUSILO_NONCOMMERCIAL
        if not _HAS_LOGGED_CAUSILO_NONCOMMERCIAL:
            logger.log(
                30,
                f"\tWarning: {self.ag_name} is a NONCOMMERCIAL model. "
                "Usage of its weights (including through AutoGluon) is not permitted "
                "for commercial tasks unless granted explicit permission by the model authors (Nums AI).",
            )
            _HAS_LOGGED_CAUSILO_NONCOMMERCIAL = True

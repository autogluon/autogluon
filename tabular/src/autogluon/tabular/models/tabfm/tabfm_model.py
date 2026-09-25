from __future__ import annotations

import contextlib
import logging
from typing import ClassVar, Iterator

import numpy as np
import pandas as pd

from autogluon.core.models.abstract.shared_weights import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from ._weight_fetch import weight_fetch_policy

logger = logging.getLogger(__name__)

_HAS_LOGGED_TABFM_NONCOMMERCIAL: bool = False


@contextlib.contextmanager
def _root_logging_left_unconfigured() -> Iterator[None]:
    """Remove the root handler tabfm's ``absl.logging`` calls install on first use.

    Logging through the root logger while it has no handler makes the ``logging`` module configure
    one, which would change the logging setup of the process AutoGluon runs in.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers:
                root.removeHandler(handler)


class TabFMModel(AbstractTorchModel):
    """TabFM is a tabular foundation model by Google Research that predicts via in-context learning.

    It handles mixed numerical and categorical columns and missing values with its own
    preprocessing, so the typed DataFrame is passed through unchanged. The checkpoint's
    classification head is ten classes wide.

    Codebase: https://github.com/google-research/tabfm (Apache-2.0)
    Weights: https://huggingface.co/google/tabfm-1.0.0-pytorch (TabFM Non-Commercial License v1.0)

    .. versionadded:: 1.6.4
    """

    ag_key = "TABFM"
    ag_name = "TabFM"
    ag_priority = 40
    seed_name = "random_state"
    gpu_strongly_recommended: bool = True  # in-context inference is an order of magnitude slower on CPU
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {"max_classes": 10}
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,  # Refit on the full data for faster inference and similar quality as the bag.
    }
    """Set fold_fitting_strategy to sequential_local, as parallel folding crashes if model weights aren't pre-downloaded."""

    #: The estimator takes the network ``tabfm_v1_0_0.load`` returns; one build per model type,
    #: checkpoint path, dtype and device per process.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabfm.src.pytorch.tabfm_v1_0_0:load", key=("model_type", "checkpoint_path", "dtype")
    )

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, verbosity: int = 2, **kwargs):
        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch

        if verbosity >= 2:
            self._log_license()
        device = self._resolve_fit_device(num_gpus=num_gpus)
        is_classification = self.problem_type in ["binary", "multiclass"]
        model_cls = TabFMClassifier if is_classification else TabFMRegressor
        X = self.preprocess(X, y=y)
        with _root_logging_left_unconfigured():
            with weight_fetch_policy(self.aux_params.fetch_pretrained_weights, stage="fit", model_name=self.name):
                network = tabfm_v1_0_0_pytorch.load(
                    model_type="classification" if is_classification else "regression", device=device
                )
            self.model = model_cls(model=network, **self._get_model_params()).fit(X=X, y=y)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        with _root_logging_left_unconfigured():
            return super()._predict_proba(X, **kwargs)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Load the pickle, applying the load-stage weight-fetch policy.

        The fitted model pickles without the network, which is put back while loading -- a network
        fetch on a host with a cold cache. That happens inside ``super().load``, before the model's
        ``aux_params`` can be read, so the only policy source is the environment
        (``AG_FETCH_PRETRAINED_WEIGHTS``), which is also the right scope for an inference-time policy.
        """
        with _root_logging_left_unconfigured(), weight_fetch_policy(True, stage="load", model_name=cls.__name__):
            return super().load(path=path, reset_paths=reset_paths, verbose=verbose)

    def get_device(self) -> str:
        param = next(self.model.model.parameters(), None)
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        # The estimator runs wherever its network lives.
        self.model.model.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def _log_license(self):
        global _HAS_LOGGED_TABFM_NONCOMMERCIAL
        if not _HAS_LOGGED_TABFM_NONCOMMERCIAL:
            logger.log(
                30,
                f"\tWarning: {self.ag_name} is a NONCOMMERCIAL model. "
                "Usage of its weights (including through AutoGluon) is not permitted "
                "for commercial tasks unless granted explicit permission by the model authors (Google).",
            )
            _HAS_LOGGED_TABFM_NONCOMMERCIAL = True

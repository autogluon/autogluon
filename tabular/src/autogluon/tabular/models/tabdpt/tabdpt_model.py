from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# FIXME: Nick:
#  TODO: batch_size is linear to memory usage
#   512 default
#   should be less for very large datasets
#   128 batch_size on Bioresponse -> 12 GB VRAM
#       Train Data Rows:    2500
#       Train Data Columns: 1776
#       Problem Type:       binary
#  FIXME: Just set context_size = infinity, everything is way faster, memory usage is way lower, etc.
#   Train Data Rows:    100000
#   Train Data Columns: 10
#   binary
#   only takes 6.7 GB during inference with batch_size = 512
# FIXME: Make it work when loading on CPU?
# FIXME: Can we run 8 in parallel to speed up?
# TODO: clip_sigma == 1 is terrible, clip_sigma == 16 maybe very good? What about higher values?
#  clip_sigma >= 16 is roughly all equivalent
# FIXME: TabDPT stores self.X_test for no reason
# FIXME: TabDPT creates faiss_knn even if it is never used. Better if `context_size=None` means it is never created.
# TODO: unit test
# TODO: memory estimate
class TabDPTModel(AbstractTorchModel):
    gpu_strongly_recommended: bool = True  # in-context inference is 12-63x slower on CPU
    ag_key = "TABDPT"
    ag_name = "TabDPT"
    seed_name = "seed"
    ag_priority = 50
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_random_seed = 0

    #: Hugging Face repo hosting every TabDPT checkpoint.
    _hf_repo_id: ClassVar[str] = "Layer6/TabDPT"
    #: Checkpoint filename in :attr:`_hf_repo_id` to pin via ``model_weight_path``, or None to
    #: use the installed ``tabdpt`` package's default weights. The tabdpt package can only load
    #: checkpoints of its own version (e.g. tabdpt 1.2 cannot load the v1.1 checkpoint), so pins
    #: must match the package the extra installs.
    _checkpoint_filename: ClassVar[str | None] = None
    #: Estimator constructor kwargs pinned for this version, mapped to the version's default value
    #: (resolved from the fit hyperparameters, falling back to the default). Empty -> the installed
    #: package's defaults.
    _constructor_defaults: ClassVar[dict[str, object]] = {}
    #: Predict-time hyperparameters accepted by this version, split by task (``temperature`` /
    #: ``permute_classes`` are classification-only).
    _predict_param_names: ClassVar[dict[str, tuple[str, ...]]] = {
        "classifier": ("context_size", "n_ensembles", "permute_classes", "temperature"),
        "regressor": ("context_size", "n_ensembles"),
    }

    _default_auxiliary_params_extra = {
        "max_rows": 100000,  # TODO: Test >100k rows
        "max_features": 2500,  # TODO: Test >2500 features
        # TabDPT decomposes a label into base-`max_num_classes` digits above the checkpoint's
        # output-head width and sums the per-digit log-probabilities
        # (`tabdpt.classifier._predict_large_cls`), so the head width bounds a forward pass, not
        # the usable class count. Verified to fit and predict a calibrated distribution at 11, 20,
        # 100 and 160 classes; the cost is ceil(log_head(n_classes)) passes per prediction.
        "max_classes": 160,
    }
    minimum_num_gpus = 0.5
    _default_ag_args_ensemble_extra = {
        "refit_folds": True,
        # Sequential fold fitting is much faster for TabDPT: a child fit is near-instant
        # (in-context model), so parallel fold workers pay far more in per-worker CUDA
        # context + checkpoint loading than they save. Benchmarked on 1 GPU with an
        # 8-fold bag: sequential is 2-4x faster than any parallel fold split
        # (0.125/0.25/0.5 GPU per fold) at both 600 and 30k train rows.
        "fold_fitting_strategy": "sequential_local",
    }
    default_resources_physical_cores_only = True
    default_num_gpus = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._predict_hps = None
        self._use_flash_og = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        device = self._resolve_fit_device(num_gpus=num_gpus)
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        model_cls = TabDPTClassifier if self.problem_type in [BINARY, MULTICLASS] else TabDPTRegressor
        fit_params, self._predict_hps = self._get_tabdpt_params(num_gpus=num_gpus)

        X = self.preprocess(X, y=y)
        y = y.to_numpy()
        if self._checkpoint_filename is not None:
            fit_params["model_weight_path"] = self._download_checkpoint()
        self.model = model_cls(
            device=device,
            **fit_params,
        )
        self.model.fit(X=X, y=y)

    @classmethod
    def _download_checkpoint(cls) -> str:
        """Resolve this version's checkpoint to a local path (from cache, else download).

        Tries the local cache first so prefetched / offline compute nodes skip the etag
        HEAD-request that ``hf_hub_download`` makes by default.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return hf_hub_download(
                repo_id=cls._hf_repo_id,
                filename=cls._checkpoint_filename,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return hf_hub_download(repo_id=cls._hf_repo_id, filename=cls._checkpoint_filename)

    def _get_tabdpt_params(self, num_gpus: float) -> tuple[dict, dict]:
        model_params = self._get_model_params()

        valid_predict_params = (
            self.seed_name,
            "context_size",
            "permute_classes",
            "temperature",
            "n_ensembles",
            "batch_size",
        )

        predict_params = {}
        for hp in valid_predict_params:
            if hp in model_params:
                predict_params[hp] = model_params.pop(hp)
        predict_params.setdefault(self.seed_name, self.default_random_seed)
        predict_params.setdefault("context_size", None)

        task = "classifier" if self.problem_type in [BINARY, MULTICLASS] else "regressor"
        supported_predict_params = (self.seed_name, *self._predict_param_names[task])
        predict_params = {key: val for key, val in predict_params.items() if key in supported_predict_params}

        fit_params = model_params
        for param, default in self._constructor_defaults.items():
            fit_params.setdefault(param, default)

        fit_params.setdefault("verbose", False)
        fit_params.setdefault("compile", False)
        if fit_params.get("use_flash", True):
            fit_params["use_flash"] = self._use_flash(num_gpus=num_gpus)
        return fit_params, predict_params

    @staticmethod
    def _use_flash(num_gpus: float) -> bool:
        """Detect if torch's native flash attention is available on the current machine."""
        if num_gpus == 0:
            return False

        import torch

        if not torch.cuda.is_available():
            return False

        if not torch.backends.cuda.is_flash_attention_available():
            return False

        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)

        return capability != (7, 5)

    def _post_fit(self, **kwargs):
        super()._post_fit(**kwargs)
        self._use_flash_og = self.model.use_flash
        return self

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)
        if device == "cpu":
            self.model.use_flash = False
            self.model.model.use_flash = False
        else:
            self.model.use_flash = self._use_flash_og
            self.model.model.use_flash = self._use_flash_og

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            y_pred = self.model.predict(X, **self._predict_hps)
            return y_pred

        y_pred_proba = self.model.ensemble_predict_proba(X, **self._predict_hps)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """TabDPT requires a numpy array as input, with missing values left as NaN.

        TabDPT handles NaN itself and does more with it than impute: it appends a binary
        missing-indicator column per affected feature before mean-imputing (`tabdpt.estimator`),
        so the encoding's default -1 for missing would both hide the missingness and put an
        out-of-range value on the feature's numeric axis.
        """
        X = super()._preprocess(X, **kwargs)
        X = self._label_encode_categoricals(X, preserve_missing=True)
        return X.to_numpy()

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    # FIXME: This is copied from TabPFN, but TabDPT is not the same
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate based on TabPFN's memory estimate logic in:
        https://github.com/PriorLabs/TabPFN/blob/57a2efd3ebdb3886245e4d097cefa73a5261a969/src/tabpfn/model/memory.py#L147.

        This is based on GPU memory usage, but hopefully with overheads it also approximates CPU memory usage.
        """
        # TODO: update, this is not correct anymore, consider using internal TabPFN functions directly.
        features_per_group = 3  # Based on TabPFNv2 default (unused)
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], min(X.shape[1], 500)
        n_feature_groups = (n_features) / features_per_group + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        memory_estimate = model_mem + 4 * X_mem + 2 * activation_mem + baseline_overhead_mem_est

        # TabDPT memory estimation is very inaccurate because it is using TabPFN memory estimate. Double it to be safe.
        memory_estimate = memory_estimate * 2

        # Note: This memory estimate is way off if `context_size` is not None
        return int(memory_estimate)

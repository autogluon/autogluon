from __future__ import annotations

import logging
from typing import Any, ClassVar

from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract import shared_weights as _shared
from autogluon.core.models.abstract._shared_weights_registry import SharedWeightsClassSettings
from autogluon.core.models.abstract.shared_weights import SharedWeights

logger = logging.getLogger(__name__)


# TODO: Add type hints once torch is a required dependency
class AbstractTorchModel(AbstractModel):
    """
    .. versionadded:: 1.5.0
    """

    shared_weights: ClassVar[SharedWeights | None] = None
    """How this model's pretrained network is shared across fits in one process; ``None`` never shares.

    A declaration names the library call that builds the network and the inputs that decide which
    network it is (see :class:`~autogluon.core.models.abstract.shared_weights.SharedWeights`). With
    it, ``fit`` runs that call once per process and checkpoint, the bagged children and the refit
    model reuse the network, the fitted model pickles without the weights and takes them back on
    load, and ``set_device`` swaps registry entries instead of moving a shared module. A fit opts out
    with ``ag_args_fit={"share_pretrained_weights": False}``; a class with the ``share_weights`` class
    setting; a configuration through the declaration's ``disabled_by``. Nothing in ``_fit`` changes.
    """

    gpu_strongly_recommended: bool = False
    """Whether fitting this model on CPU is slow enough to warn about.

    Set by in-context-learning models, whose prediction cost is dominated by
    attending over the training context: on CPU they measure 12-63x slower end to
    end than on GPU (versus ~1.5-3x for models trained with SGD), which makes a
    silent CPU fallback look like a hang rather than a configuration problem.
    See :meth:`_log_cpu_fallback_warning`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = None
        self.device_train = None
        self._shared_state: _shared.SharedState | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get("shared_weights")
        if spec is not None:
            _shared.validate_declaration(spec, cls)
            if getattr(cls, "class_settings_cls", None) is None:
                # ``TabularPredictor.fit(model_class_settings={"<key>": {"share_weights": False}})`` works for every
                # declaring class; a class with its own settings extends SharedWeightsClassSettings instead.
                cls.class_settings_cls = SharedWeightsClassSettings

    # --- shared pretrained weights ------------------------------------------------------------------

    def fit(self, **kwargs):
        """Fit through ``AbstractModel.fit``; with ``shared_weights`` declared, the library's loader is memoized meanwhile."""
        device = _shared.fit_device(kwargs.get("num_gpus"))
        return _shared.fit(self, lambda: super(AbstractTorchModel, self).fit(**kwargs), device=device)

    def _shares_weights(self) -> bool:
        """Whether this fit takes its network from the registry: a declaration, ``share_pretrained_weights``, the class setting."""
        return _shared.shares(self)

    def __getstate__(self) -> dict:
        return _shared.getstate(self, self.__dict__.copy())

    def __setstate__(self, state: dict) -> None:
        state.setdefault("_shared_state", None)
        self.__dict__.update(state)

    def predict(self, X, **kwargs):
        _shared.ensure_network(self)
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        _shared.ensure_network(self)
        return super().predict_proba(X, **kwargs)

    def prepare_for_inference(self) -> None:
        """Put a shared network back (after a load) and set it to eval mode; nothing for a model that owns its network."""
        if self._shared_state is not None:
            _shared.prepare_for_inference(self)

    def _resolve_fit_device(self, num_gpus: int | float, gpu_device: str = "cuda") -> str:
        """Resolve the torch device for `_fit` from the allocated `num_gpus`.

        Logs the CPU-fallback warning (see `_log_cpu_fallback_warning`) and raises when a
        GPU was allocated but CUDA is unavailable. `gpu_device` is the device string
        returned for GPU fits (pass e.g. "cuda:0" for models that assume a single
        visible GPU).
        """
        self._log_cpu_fallback_warning(num_gpus=num_gpus)
        if num_gpus == 0:
            return "cpu"
        from torch.cuda import is_available

        if not is_available():
            # TODO: Consider warning and falling back to CPU instead of raising.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )
        return gpu_device

    def _log_cpu_fallback_warning(self, num_gpus: int | float) -> None:
        """Warn once per fit when a ``gpu_strongly_recommended`` model fits on CPU
        without the user having asked for it.

        Silent when the user set ``num_gpus`` themselves (e.g. ``num_gpus=0`` in
        ``ag_args_fit``), since then the CPU fit is the requested behavior.
        """
        if not self.gpu_strongly_recommended or num_gpus:
            return
        if self._user_params_aux.get("num_gpus") is not None:
            return  # explicit user choice, not a fallback

        try:
            import torch

            gpu_present = torch.cuda.is_available()
        except Exception:
            gpu_present = False

        reason = (
            "no GPU was allocated to it, though this machine has one - check `num_gpus` "
            "in `ag_args_fit` and the resources available to the fit"
            if gpu_present
            else "no CUDA GPU is available on this machine"
        )
        logger.log(
            30,
            f"\tWARNING: {self.name} is fitting on CPU because {reason}. This model attends over the "
            f"training context for every prediction, so a CPU fit is typically an order of magnitude "
            f"slower than on GPU (measured 12-63x end to end) and may appear to hang. "
            f"Pass `num_gpus=0` explicitly to silence this warning.",
        )

    def suggest_device_infer(self, verbose: bool = False) -> str:
        import torch

        # Put the model on the same device it was trained on (GPU/MPS) if it is available; otherwise use CPU
        if self.device_train is None:
            original_device_type = None  # skip update because no device is recorded
        elif isinstance(self.device_train, str):
            original_device_type = self.device_train
        else:
            original_device_type = self.device_train.type
        if original_device_type is None:
            # fallback to CPU
            device = torch.device("cpu")
        elif "cuda" in original_device_type:
            # cuda: nvidia GPU
            device = torch.device(original_device_type if torch.cuda.is_available() else "cpu")
        elif "mps" in original_device_type:
            # mps: Apple Silicon
            device = torch.device(original_device_type if torch.backends.mps.is_available() else "cpu")
        else:
            device = torch.device(original_device_type)

        if verbose and (original_device_type != device.type):
            logger.log(
                15,
                f"Model is trained on {original_device_type}, but the device is not available - "
                f"loading on {device.type}...",
            )

        return device.type

    @classmethod
    def to_torch_device(cls, device: str):
        import torch

        return torch.device(device)

    def get_device(self) -> str:
        """
        Returns torch.device(...) of the fitted model

        A model whose network is shared reports the device of its registry entry. Otherwise requires
        implementation by the inheriting model class; refer to overriding methods in existing models
        for reference implementations.
        """
        if not _shared.owns_network(self) and self._shared_state.device is not None:
            return self._shared_state.device
        raise NotImplementedError

    def set_device(self, device: str):
        if not isinstance(device, str):
            device = device.type
        self.device = device
        # A shared network is never moved: the registry entry for the new device replaces it, and the
        # estimator's device fields and own tensors follow. ``_set_device`` runs for a model that owns its network.
        if not _shared.set_device(self, device):
            self._set_device(device=device)

    def _set_device(self, device: str):
        """
        Sets the device for the inner model object.

        Requires implementation by the inheriting model class.
        Refer to overriding methods in existing models for reference implementations.

        If your model does not need to edit inner model object details, you can simply make the logic `pass`.
        """
        raise NotImplementedError

    def _post_fit(self, **kwargs):
        super()._post_fit(**kwargs)
        if self._get_class_tags().get("can_set_device", False):
            self.device_train = self.get_device()
            self.device = self.device_train
        return self

    def _pickles_pretrained_weights(self) -> bool:
        """Whether a pickle of this fitted model carries its network's tensors.

        False for a fit that shared its network (the pickle is weightless and the network is reattached
        on load), else the ``pickles_pretrained_weights`` class tag. ``save`` moves the model to
        ``set_device_on_save_to`` only when it does.
        """
        if not _shared.owns_network(self):
            return False
        return bool(self._get_class_tags().get("pickles_pretrained_weights", True))

    def _get_memory_size(self) -> int:
        shared = _shared.memory_size(self)
        return super()._get_memory_size() if shared is None else shared

    def get_info(self, include_feature_metadata: bool = True) -> dict:
        info = super().get_info(include_feature_metadata=include_feature_metadata)
        if self.shared_weights is not None:
            info["shared_weights"] = _shared.info(self)
        return info

    def save(self, path: str = None, verbose=True) -> str:
        """
        Need to set device to CPU to be able to load on a non-GPU environment
        """
        reset_device = False
        og_device = self.device

        # Save on CPU to ensure the model can be loaded without GPU
        if self.is_fit() and self._pickles_pretrained_weights():
            device_save = self._get_class_tags().get("set_device_on_save_to", None)
            if device_save is not None:
                self.set_device(device=device_save)
                reset_device = True
        path = super().save(path=path, verbose=verbose)
        # Put the model back to the device after the save
        if reset_device:
            self.set_device(device=og_device)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        Loads the model from disk to memory.
        The loaded model will be on the same device it was trained on (cuda/mps);
        if the device is not available (trained on GPU, deployed on CPU), then `cpu` will be used.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in os.path.join(path, cls.model_file_name).
        reset_paths : bool, default True
            Whether to reset the self.path value of the loaded model to be equal to path.
            It is highly recommended to keep this value as True unless accessing the original self.path value is important.
            If False, the actual valid path and self.path may differ, leading to strange behaviour and potential exceptions if the model needs to load any other files at a later time.
        verbose : bool, default True
            Whether to log the location of the loaded file.

        Returns
        -------
        model : cls
            Loaded model object.
        """
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        # Put the model on the same device it was trained on (GPU/MPS) if it is available; otherwise use CPU
        if model.is_fit() and model._get_class_tags().get("set_device_on_load", False):
            device = model.suggest_device_infer(verbose=verbose)
            model.set_device(device=device)

        return model

    @classmethod
    def _class_tags(cls):
        return {
            "can_set_device": True,
            "set_device_on_save_to": "cpu",
            "set_device_on_load": True,
            # Whether the pickle of a fitted model holds its network's tensors, so `save` must move
            # them to `set_device_on_save_to` first. False for a model whose pickle is weightless.
            "pickles_pretrained_weights": True,
        }

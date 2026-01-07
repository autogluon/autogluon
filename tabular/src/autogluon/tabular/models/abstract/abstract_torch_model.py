from __future__ import annotations

import logging

from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


# TODO: Add type hints once torch is a required dependency
class AbstractTorchModel(AbstractModel):
    """
    .. versionadded:: 1.5.0
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = None
        self.device_train = None

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

        Requires implementation by the inheriting model class.
        Refer to overriding methods in existing models for reference implementations.
        """
        raise NotImplementedError

    def set_device(self, device: str):
        if not isinstance(device, str):
            device = device.type
        self.device = device
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

    def save(self, path: str = None, verbose=True) -> str:
        """
        Need to set device to CPU to be able to load on a non-GPU environment
        """
        reset_device = False
        og_device = self.device

        # Save on CPU to ensure the model can be loaded without GPU
        if self.is_fit():
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
        }

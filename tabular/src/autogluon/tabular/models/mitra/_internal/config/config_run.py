from __future__ import annotations

from dataclasses import dataclass

import torch

from ..._internal.config.config_pretrain import ConfigSaveLoadMixin
from ..._internal.config.enums import ModelName


@dataclass
class ConfigRun(ConfigSaveLoadMixin):
    device: torch.device
    seed: int
    model_name: ModelName
    hyperparams: dict

    @classmethod
    def create(
        cls,  
        device: torch.device, 
        seed: int, 
        model_name: ModelName, 
        hyperparams: dict
    ) -> "ConfigRun":
       
        return cls(
            device=device,
            seed=seed,
            model_name=model_name,
            hyperparams=hyperparams
        )
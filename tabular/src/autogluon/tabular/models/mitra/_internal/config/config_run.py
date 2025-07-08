from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import torch

from ..._internal.config.config_pretrain import ConfigSaveLoadMixin
from ..._internal.config.enums import DatasetSize, DownstreamTask, ModelName, Task

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
    ) -> Self:
       
        return cls(
            device=device,
            seed=seed,
            model_name=model_name,
            hyperparams=hyperparams
    )
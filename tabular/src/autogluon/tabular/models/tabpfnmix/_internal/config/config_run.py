from __future__ import annotations

from dataclasses import dataclass

import torch

from .config_save_load_mixin import ConfigSaveLoadMixin
from ..core.enums import Task


@dataclass
class ConfigRun(ConfigSaveLoadMixin):
    task: Task
    hyperparams: dict
    seed: int = 0
    device: torch.device = None

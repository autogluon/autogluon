from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.enums import Task


@dataclass
class ConfigRun:
    task: Task
    hyperparams: dict
    seed: int = 0
    device: torch.device = None

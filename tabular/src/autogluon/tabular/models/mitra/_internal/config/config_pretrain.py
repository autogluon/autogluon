from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml
import os

import torch
from omegaconf import DictConfig, OmegaConf

from ..._internal.config.enums import GeneratorName, ModelName, LossName, Task

@dataclass
class ConfigData():
    generator: GeneratorName
    min_samples_support: int
    max_samples_support: int
    n_samples_query: int
    min_features: int
    max_features: int
    max_classes: int
    sample_multinomial_categorical: bool
    sample_multinomial_label: bool
    generator_hyperparams: dict
    task: Task

    def __post_init__(self):

        assert self.min_samples_support <= self.max_samples_support
        assert self.min_features <= self.max_features

@dataclass
class ConfigModel():
    name: ModelName
    hyperparams: dict


@dataclass
class ConfigPreprocessing():
    use_quantile_transformer: bool
    use_feature_count_scaling: bool

@dataclass
class ConfigGradScaler():
    enabled: bool
    scale_init: float
    scale_min: float
    growth_interval: int


    def __post_init__(self):
        assert self.scale_init >= self.scale_min, "Scale init must be greater than scale min"
        assert self.scale_min >= 1, "Scale min lower than 1 makes no sense for mixed precision training"
        assert type(self.scale_init) == float, "Scale init must be a float, otherwise gradscaler will return an error"
        assert type(self.scale_min) == float, "Scale min must be a float, otherwise gradscaler will return an error"

@dataclass
class ConfigOptim():
    steps: int
    log_every_n_steps: int
    eval_every_n_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    cosine_scheduler: bool
    max_grad_norm: float
    label_smoothing: float
    regression_loss: LossName
    use_pretrained_weights: bool
    path_to_weights: str
    resume_states: bool
    path_to_states: str
    precision: str
    grad_scaler: ConfigGradScaler

    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig) -> Self:

        grad_scaler = ConfigGradScaler(**cfg_hydra.grad_scaler)
        cfg_dict: dict = OmegaConf.to_container(cfg_hydra)      # type: ignore
        del cfg_dict["grad_scaler"]

        regression_loss = LossName[cfg_dict["regression_loss"]]
        del cfg_dict["regression_loss"]

        return cls(
            grad_scaler=grad_scaler,
            regression_loss=regression_loss,
            **cfg_dict
        )

    def __post_init__(self):
        assert hasattr(torch, self.precision), f"Precision {self.precision} not supported by torch"

class ConfigSaveLoadMixin(yaml.YAMLObject):

    def save(self, path: Path) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self, f, default_flow_style=False)


    @classmethod
    def load(cls, path: Path) -> Self:

        with open(path, 'r') as f:
            # It's unsafe, but not unsafer than the pickle module
            config = yaml.unsafe_load(f)

        return config

@dataclass
class ConfigPretrain(ConfigSaveLoadMixin):
    run_name: str
    output_dir: Path
    seed: int
    devices: list[torch.device]
    device: torch.device
    max_cpus_per_device: Optional[int]
    use_ddp: bool
    workers_per_gpu: int
    model: ConfigModel
    data: ConfigData
    optim: ConfigOptim
    preprocessing: ConfigPreprocessing
    load_from_file: bool
    load_path_x: str
    load_path_y: str
    save_file: bool
    save_file_only: bool
    save_path_x: str
    save_path_y: str
    number_of_runs: int

    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig):

        assert not os.path.exists(cfg_hydra.output_dir), f'Output directory {cfg_hydra.output_dir} already exists! Please change to a new folder.'

        output_dir = Path(cfg_hydra.output_dir)

        devices = [torch.device(device) for device in cfg_hydra.devices]

        # Initialize device to cpu, DDP will overwrite this
        device = torch.device("cpu")

        return cls(
            run_name=cfg_hydra.run_name,
            output_dir=output_dir,
            devices=devices,
            device=device,
            max_cpus_per_device=cfg_hydra.max_cpus_per_device,
            use_ddp=len(devices) > 1,
            seed=cfg_hydra.seed,
            workers_per_gpu=cfg_hydra.workers_per_gpu,
            model = ConfigModel(
                name = ModelName[cfg_hydra.model.name],
                hyperparams = OmegaConf.to_container(cfg_hydra.model.hyperparams),   
            ),
            data = ConfigData(
                generator=GeneratorName(cfg_hydra.data.generator),
                min_samples_support=cfg_hydra.data.min_samples_support,
                max_samples_support=cfg_hydra.data.max_samples_support,
                n_samples_query=cfg_hydra.data.n_samples_query,
                min_features=cfg_hydra.data.min_features,
                max_features=cfg_hydra.data.max_features,
                max_classes=cfg_hydra.data.max_classes,
                task=Task[cfg_hydra.data.task],
                sample_multinomial_categorical=cfg_hydra.data.sample_multinomial_categorical,
                sample_multinomial_label=cfg_hydra.data.sample_multinomial_label,
                generator_hyperparams=OmegaConf.to_container(cfg_hydra.data.generator_hyperparams),    # type: ignore
            ),
            optim = ConfigOptim.from_hydra(cfg_hydra.optim),
            preprocessing = ConfigPreprocessing(**cfg_hydra.preprocessing),
            load_from_file = cfg_hydra.load_from_file,
            load_path_x = cfg_hydra.load_path_x,
            load_path_y = cfg_hydra.load_path_y,
            save_file = cfg_hydra.save_file,
            save_file_only = cfg_hydra.save_file_only,
            save_path_x = cfg_hydra.save_path_x,
            save_path_y = cfg_hydra.save_path_y,
            number_of_runs = cfg_hydra.number_of_runs,
        )

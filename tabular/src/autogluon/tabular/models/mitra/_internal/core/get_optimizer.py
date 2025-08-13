import torch
from torch.optim import SGD, Adam, AdamW

from ..._internal.config.config_pretrain import ConfigPretrain


def get_optimizer(hyperparams: dict, model: torch.nn.Module) -> torch.optim.Optimizer:

    optimizer: torch.optim.Optimizer

    if hyperparams['optimizer'] == "adam":
        optimizer = Adam(
            model.parameters(), 
            lr=hyperparams['lr'],
            betas=(0.9, 0.999),
            weight_decay=hyperparams['weight_decay']
        )
    elif hyperparams['optimizer'] == "adamw":
        optimizer = AdamW(
            model.parameters(), 
            lr=hyperparams['lr'],
            betas=(0.9, 0.999),
            weight_decay=hyperparams['weight_decay']
        )
    elif hyperparams['optimizer'] == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
    else:
        raise ValueError("Optimizer not recognized")
    
    return optimizer


def get_optimizer_pretrain(cfg: ConfigPretrain, model: torch.nn.Module) -> torch.optim.Optimizer:

    parameters = [(name, param) for name, param in model.named_parameters()]

    parameters_with_weight_decay = []
    parameters_without_weight_decay = []

    for name, param in parameters:
        if name.endswith("bias") or 'norm' in name or 'embedding' in name:
            parameters_without_weight_decay.append(param)
        else:
            parameters_with_weight_decay.append(param)

    optimizer_parameters = [
        {"params": parameters_with_weight_decay, "weight_decay": cfg.optim.weight_decay},
        {"params": parameters_without_weight_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_parameters, 
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        weight_decay=cfg.optim.weight_decay
    )
    
    return optimizer


class GradScaler(torch.amp.GradScaler):

    def __init__(
        self, 
        enabled: bool = True,
        scale_init: float = 2.**16,
        scale_min: float = 1.,
        growth_interval: int = 2000,
        device: str = 'cuda'
    ):
        super().__init__(enabled=enabled, device="cpu", init_scale=scale_init, growth_interval=growth_interval) # type: ignore
        self._enabled = enabled
        self.scale_min = scale_min
        self.device = device

        if not self._enabled:
            # We write scale=1 to log if the scaler is disabled
            self._scale = torch.tensor((1,), dtype=torch.float32, device=self.device)


    def update(self):

        if not self._enabled:
            return

        super().update()

        if self._scale < self.scale_min:
            super().update(self.scale_min)


def move_optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
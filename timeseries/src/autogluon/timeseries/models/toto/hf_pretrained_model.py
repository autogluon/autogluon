import json
import logging
import os
from pathlib import Path

from transformers import PretrainedConfig, PreTrainedModel

from ._internal.backbone import TotoBackbone


class TotoConfig(PretrainedConfig):
    model_type = "toto"

    def __init__(
        self,
        dropout: float = 0.0,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        output_distribution_classes: list[str] | None = None,
        output_distribution_kwargs: dict | None = None,
        patch_size: int = 64,
        scale_factor_exponent: float = 10.0,
        spacewise_every_n_layers: int = 12,
        spacewise_first: bool = False,
        stabilize_with_global: bool = True,
        stride: int = 64,
        transformers_version: str = "4.49.0",
        use_memory_efficient_attention: bool = False,
        **kwargs,
    ):
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_distribution_classes = output_distribution_classes or ["MixtureOfStudentTsOutput"]
        self.output_distribution_kwargs = output_distribution_kwargs or {"k_components": 24}
        self.patch_size = patch_size
        self.scale_factor_exponent = scale_factor_exponent
        self.spacewise_every_n_layers = spacewise_every_n_layers
        self.spacewise_first = spacewise_first
        self.stabilize_with_global = stabilize_with_global
        self.stride = stride
        self.transformers_version = transformers_version
        self.use_memory_efficient_attention = use_memory_efficient_attention

        super().__init__(**kwargs)


class TotoPretrainedModel(PreTrainedModel):
    config_class = TotoConfig
    base_model_prefix = "model"  # optional, used for weight naming conventions

    def __init__(self, config: TotoConfig):
        super().__init__(config)
        self.model = TotoBackbone(
            patch_size=config.patch_size,
            stride=config.stride,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_hidden_dim=getattr(config, "mlp_hidden_dim", 3072),
            dropout=config.dropout,
            spacewise_every_n_layers=config.spacewise_every_n_layers,
            scaler_cls=getattr(config, "scaler_cls", "model.scaler.CausalPatchStdMeanScaler"),
            output_distribution_classes=config.output_distribution_classes,
            spacewise_first=config.spacewise_first,
            output_distribution_kwargs=config.output_distribution_kwargs,
            use_memory_efficient_attention=False,
            stabilize_with_global=config.stabilize_with_global,
            scale_factor_exponent=config.scale_factor_exponent,
            **getattr(config, "extra_kwargs", {}),
        )
        self.post_init()

    @staticmethod
    def _remap_state_dict_keys(state_dict):
        remap = {
            "mlp.0.w12.weight": "mlp.0.weight",
            "mlp.0.w12.bias": "mlp.0.bias",
            "mlp.0.w3.weight": "mlp.2.weight",
            "mlp.0.w3.bias": "mlp.2.bias",
        }

        new_state = {}
        keys_to_remap = []
        for key in list(state_dict.keys()):
            for old, new in remap.items():
                if old in key:
                    new_key = key.replace(old, new)
                    keys_to_remap.append((key, new_key))
                    break

        new_state = state_dict.copy()
        for old_key, new_key in keys_to_remap:
            new_state[new_key] = new_state.pop(old_key)

        return new_state

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        device_map: str = "cpu",
        strict=True,
        **model_kwargs,
    ):
        """
        Custom checkpoint loading. Used to load a local
        safetensors checkpoint with an optional config.json file.
        """
        import safetensors.torch as safetorch

        if os.path.isdir(checkpoint_path):
            safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
        else:
            safetensors_file = checkpoint_path

        if os.path.exists(safetensors_file):
            model_state = safetorch.load_file(safetensors_file, device=device_map)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {safetensors_file}")

        # Load configuration from config.json if it exists.
        config_file = os.path.join(checkpoint_path, "config.json")
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)

        # Merge any extra kwargs into the configuration.
        config.update(model_kwargs)

        remapped_state_dict = cls._remap_state_dict_keys(model_state)

        instance = cls(**config)

        # Filter out unexpected keys
        filtered_remapped_state_dict = {
            k: v
            for k, v in remapped_state_dict.items()
            if k in instance.state_dict() and not k.endswith("rotary_emb.freqs")
        }

        instance.load_state_dict(filtered_remapped_state_dict, strict=strict)
        instance.to(device_map)  # type: ignore

        return instance

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: Path | str | None = None,
        force_download: bool = False,
        proxies: dict | None = None,
        resume_download: bool | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        device_map: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        from huggingface_hub import constants, hf_hub_download

        transformers_logger = logging.getLogger("transformers.modeling_utils")
        original_level = transformers_logger.level

        try:
            # Here we suppress transformers logger's "some weights were not initialized" error since the
            # remapping hook is only called after the initial model loading.
            transformers_logger.setLevel(logging.ERROR)

            if os.path.isdir(model_id):
                print("Loading weights from local directory")
                model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
                model = cls.load_from_checkpoint(model_file, device_map, strict, **model_kwargs)
            else:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                model = cls.load_from_checkpoint(model_file, device_map, strict, **model_kwargs)
        finally:
            transformers_logger.setLevel(original_level)

        return model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

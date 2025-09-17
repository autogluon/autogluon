from typing import Optional

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
        output_distribution_classes: Optional[list[str]] = None,
        output_distribution_kwargs: Optional[dict] = None,
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
        self._register_load_state_dict_pre_hook(self._remap_state_dict_keys_hook)
        self.post_init()

    def _remap_state_dict_keys_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        remap = {
            "mlp.0.w12.weight": "mlp.0.weight",
            "mlp.0.w12.bias": "mlp.0.bias",
            "mlp.0.w3.weight": "mlp.2.weight",
            "mlp.0.w3.bias": "mlp.2.bias",
        }

        keys_to_remap = []
        for key in list(state_dict.keys()):
            for old, new in remap.items():
                if old in key:
                    new_key = key.replace(old, new)
                    keys_to_remap.append((key, new_key))
                    break

        for old_key, new_key in keys_to_remap:
            state_dict[new_key] = state_dict.pop(old_key)

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, torch_dtype=None, device_map=None, **kwargs):
        # Transformers follows a different load path that does not call load_state_dict hooks when
        # loading with explicit device maps. Here, we first load the model with no device maps and
        # move it.
        model = super().from_pretrained(model_name_or_path, config=config, torch_dtype=torch_dtype, **kwargs)
        if device_map is not None:
            model = model.to(device_map)
        return model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

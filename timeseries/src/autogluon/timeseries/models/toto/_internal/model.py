from typing import Optional

from transformers import PretrainedConfig, PreTrainedModel

from .backbone import TotoBackbone


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
        self.post_init()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = self._map_state_dict_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _map_state_dict_keys(state_dict):
        remap = {
            "mlp.0.w12.weight": "mlp.0.weight",
            "mlp.0.w12.bias": "mlp.0.bias",
            "mlp.0.w3.weight": "mlp.2.weight",
            "mlp.0.w3.bias": "mlp.2.bias",
        }

        def r(k):
            for old, new in remap.items():
                k = k.replace(old, new)
            return k

        return {r(k): v for k, v in state_dict.items()}

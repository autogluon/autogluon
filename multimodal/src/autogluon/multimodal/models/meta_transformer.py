import logging
from typing import Dict, Optional

import torch
from timm import create_model
from timm.models.vision_transformer import Block
from torch import nn

from ..constants import (
    CATEGORICAL,
    FEATURES,
    IMAGE,
    IMAGE_VALID_NUM,
    LABEL,
    LOGITS,
    NUMERICAL,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
)
from .custom_transformer import CLSToken
from .ft_transformer import CategoricalFeatureTokenizer, NumEmbeddings
from .utils import (
    assign_layer_ids,
    get_hf_config_and_model,
    get_image_size_mean_std,
    get_pretrained_tokenizer,
    get_text_segment_num,
    get_text_token_max_len,
    init_weights,
    replace_missing_images_with_learnable,
)

logger = logging.getLogger(__name__)


class MetaTransformer(nn.Module):
    def __init__(
        self,
        prefix: str,
        num_classes: int,
        checkpoint_path: str,
        model_version: str,
        has_image: bool,
        has_text: bool,
        num_numerical_columns: int,
        num_categories: Dict,
        numerical_fill_values: Dict,
        image_size: Optional[int] = None,
        image_norm: Optional[str] = None,
        image_chan_num: Optional[int] = 3,
        use_learnable_image: Optional[bool] = False,
        max_text_len: Optional[int] = None,
        text_segment_num: Optional[int] = 1,
    ):
        super().__init__()
        logger.debug(f"initializing {prefix} (MetaTransformer)")
        self.prefix = prefix
        self.checkpoint_name = checkpoint_path

        if model_version == "base":
            dim = 768
            num_heads = 12
            layer_num = 12
        elif model_version == "large":
            dim = 1024
            num_heads = 16
            layer_num = 24
        else:
            raise ValueError(f"Unsupported model version: {model_version}. Options are 'base' and 'large'.")

        self.model = nn.Sequential(
            *[
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                )
                for i in range(layer_num)
            ]
        )

        checkpoint = torch.load(checkpoint_path)  # nosec B614
        self.checkpoint_path = checkpoint_path
        self.model.load_state_dict(checkpoint, strict=True)

        self.head = nn.Linear(dim, num_classes) if num_classes else nn.Identity()

        self.cls_token = CLSToken(token_dim=dim, initialization="uniform")
        self.config = None

        self.tokenizer = None
        self.text_adaptor = None
        self.image_tokenizer = None
        self.image_adaptor = None
        self.categorical_feature_tokenizer = None
        self.categorical_adapter = None
        self.numerical_feature_tokenizer = None
        self.numerical_adapter = None

        # if has_text or has_image:
        #     clip_ckpt = "openai/clip-vit-base-patch32"
        #     _, clip_model = get_hf_config_and_model(checkpoint_name=clip_ckpt, pretrained=True)

        if has_text:
            checkpoint_name = "microsoft/deberta-v3-base"
            _, text_model = get_hf_config_and_model(checkpoint_name=checkpoint_name, pretrained=True)
            self.text_config = text_model.config
            # refer to https://github.com/invictus717/MetaTransformer/blob/master/Data2Seq/Data2Seq.py#L28
            self.tokenizer = get_pretrained_tokenizer(
                tokenizer_name="hf_auto",
                checkpoint_name=checkpoint_name,
            )
            self.text_embed = text_model.embeddings
            self.text_adaptor = nn.Linear(self.text_config.hidden_size, dim)
            self.tokenizer_name = "hf_auto"
            self.max_text_len = get_text_token_max_len(
                provided_max_len=max_text_len,
                config=self.text_config,
                tokenizer=self.tokenizer,
                checkpoint_name=checkpoint_name,
            )
            self.text_segment_num = get_text_segment_num(
                config=self.text_config,
                provided_segment_num=text_segment_num,
                checkpoint_name=checkpoint_name,
            )
        if has_image:
            image_model = create_model("timm/vit_base_patch16_224.mae", pretrained=True)
            self.image_config = image_model.default_cfg
            self.patch_embed = image_model.patch_embed
            self.image_adaptor = nn.Linear(image_model.embed_dim, dim)
            self.image_size, self.image_mean, self.image_std = get_image_size_mean_std(
                model_name=self.prefix,
                config=self.image_config,
                provided_size=image_size,
                provided_norm_type=image_norm,
                support_variable_input_size=False,
            )
            self.use_learnable_image = use_learnable_image
            if self.use_learnable_image:
                self.learnable_image = nn.Parameter(torch.zeros(image_chan_num, self.image_size, self.image_size))
                logger.debug("will use a learnable image to replace missing ones")

        if num_categories:
            self.num_categories = num_categories
            self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(
                num_categories=list(num_categories.values()),
                token_dim=dim,
                bias=True,
                initialization="normal",
            )
            self.categorical_adapter = nn.Linear(dim, dim)

        if num_numerical_columns > 0:
            self.num_numerical_columns = num_numerical_columns
            self.numerical_fill_values = numerical_fill_values
            self.numerical_feature_tokenizer = NumEmbeddings(
                in_features=num_numerical_columns,
                d_embedding=dim,
                embedding_arch=["linear"],
            )
            self.numerical_adapter = nn.Linear(dim, dim)

        self.out_features = dim

        # init weights
        self.head.apply(init_weights)
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def text_token_ids_key(self):
        return f"{self.prefix}_{TEXT_TOKEN_IDS}"

    @property
    def text_valid_length_key(self):
        return f"{self.prefix}_{TEXT_VALID_LENGTH}"

    @property
    def text_segment_ids_key(self):
        return f"{self.prefix}_{TEXT_SEGMENT_IDS}"

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def categorical_key(self):
        return f"{self.prefix}_{CATEGORICAL}"

    @property
    def numerical_key(self):
        return f"{self.prefix}_{NUMERICAL}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []
        if self.image_tokenizer:
            images = batch[self.image_key]
            image_valid_num = batch[self.image_valid_num_key]
            b, n, c, h, w = images.shape
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))  # (b, n)
            if self.use_learnable_image:
                images = replace_missing_images_with_learnable(
                    images=images,
                    image_masks=image_masks,
                    learnable_image=self.learnable_image,
                )
            image_embeddings = self.patch_embed(images.reshape((b * n, c, h, w)))  # (b*n, l, d)
            assert image_embeddings.ndim == 3
            image_embeddings = self.image_adaptor(image_embeddings)
            multimodal_features.append(image_embeddings)
        if self.text_adaptor:  # text tokenizer is used in text processor
            text_token_ids = batch[self.text_token_ids_key]
            text_valid_length = batch[self.text_valid_length_key]
            steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
            text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)
            # text_embeddings = self.text_embeddings(batch[self.text_token_ids_key])  # (b, l, d)
            input_ids = text_token_ids
            inputs_embeds = None
            attention_mask = text_masks
            position_ids = None
            if "token_type_ids" in self.tokenizer.model_input_names:
                token_type_ids = batch[self.text_segment_ids_key]
            else:
                token_type_ids = None

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            text_embeddings = self.text_embed(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            text_embeddings = self.text_adaptor(text_embeddings)
            assert text_embeddings.ndim == 3
            multimodal_features.append(text_embeddings)
        if self.categorical_feature_tokenizer:
            categorical_inputs = []
            for categorical_input in batch[self.categorical_key]:
                categorical_inputs.append(categorical_input)
            categorical_inputs = torch.stack(categorical_inputs, dim=1)

            categorical_features = self.categorical_feature_tokenizer(categorical_inputs)
            categorical_features = self.categorical_adapter(categorical_features)  # (b, l, d)
            assert categorical_features.ndim == 3
            multimodal_features.append(categorical_features)
        if self.numerical_feature_tokenizer:
            numerical_features = self.numerical_feature_tokenizer(batch[self.numerical_key])
            numerical_features = self.numerical_adapter(numerical_features)
            assert numerical_features.ndim == 3
            multimodal_features.append(numerical_features)

        multimodal_features = torch.cat(multimodal_features, dim=1)
        multimodal_features = self.cls_token(multimodal_features)
        features = self.model(multimodal_features)
        pooled_features = features[:, -1, :]  # CLSToken append the cls token to the sequence tail
        logits = self.head(pooled_features)
        ret = {
            LOGITS: logits,
            FEATURES: pooled_features,
        }
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "embeddings",
            "LayerNorm",
            "wte",
            "wpe",
            "shared.weight",
            "encoder.conv.conv",
            "relative_attention_bias",
            "dummy_layer",
        )
        post_encoder_patterns = ("head", "pooler", "ln_f", "final_layer_norm")
        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

import json
import logging
import os
from typing import Dict, List, Optional

import torch
from timm import create_model
from timm.layers.linear import Linear
from torch import nn

from ..constants import COLUMN, COLUMN_FEATURES, FEATURES, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASKS
from .utils import (
    assign_layer_ids,
    get_column_features,
    get_image_size_mean_std,
    get_model_head,
    replace_missing_images_with_learnable,
)

logger = logging.getLogger(__name__)


# Stores the class names of the timm backbones that support variable input size. You can add more backbones to the list.
SUPPORT_VARIABLE_INPUT_SIZE_TIMM_CLASSES = {"convnext", "efficientnet", "mobilenetv3", "regnet", "resnet"}


class TimmAutoModelForImagePrediction(nn.Module):
    """
    Support TIMM image backbones.
    Refer to https://github.com/rwightman/pytorch-image-models
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: Optional[int] = 0,
        mix_choice: Optional[str] = "all_logits",
        pretrained: Optional[bool] = True,
        image_size: Optional[int] = None,
        image_norm: Optional[str] = None,
        image_chan_num: Optional[int] = 3,
        use_learnable_image: Optional[bool] = False,
    ):
        """
        Load a pretrained image backbone from TIMM.

        Parameters
        ----------
        prefix
            The prefix of the TimmAutoModelForImagePrediction model.
        checkpoint_name
            Name of the timm checkpoint, or local parent directory of the saved finetuned timm weights and config.
        num_classes
            The number of classes. 1 for a regression task.
        mix_choice
            Choice used for mixing multiple images. We now support.
            - all_images
                The images are directly averaged and passed to the model.
            - all_logits
                The logits output from individual images are averaged to generate the final output.
        pretrained
            Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        image_size
            The provided width / height of a square image.
        """
        super().__init__()
        # In TIMM, if num_classes==0, then create_model would automatically set self.model.head = nn.Identity()
        logger.debug(f"initializing {prefix} (TimmAutoModelForImagePrediction)")
        logger.debug(f"model checkpoint: {checkpoint_name}")
        if os.path.exists(checkpoint_name):
            checkpoint_path = f"{checkpoint_name}/pytorch_model.bin"
            try:
                with open(f"{checkpoint_name}/config.json") as f:
                    self.config = json.load(f)
                    pretrained_cfg = self.config.get("pretrained_cfg", {})
                    for k, v in pretrained_cfg.items():
                        if k not in self.config:
                            self.config[k] = v
                    self.checkpoint_name = self.config.get("architecture", None)
                    self.model = create_model(self.checkpoint_name, checkpoint_path=checkpoint_path, num_classes=0)
                    # create a head with new num_classes
                    self.head = (
                        Linear(in_features=self.config["num_features"], out_features=num_classes)
                        if num_classes > 0
                        else nn.Identity()
                    )
                    self.num_classes = num_classes if num_classes is not None else 0
            except:
                raise ValueError(f"Timm model path {checkpoint_name} does not exist or model is invalid.")
        else:
            self.checkpoint_name = checkpoint_name
            self.model = create_model(checkpoint_name, pretrained=pretrained, num_classes=num_classes)
            self.head = get_model_head(model=self.model)
            self.config = self.model.default_cfg
            self.num_classes = self.model.num_classes

        self.pretrained = pretrained
        self.out_features = self.model.num_features
        self.global_pool = self.model.global_pool if hasattr(self.model, "global_pool") else None
        self.model.reset_classifier(0)  # remove the internal head

        self.mix_choice = mix_choice
        logger.debug(f"mix_choice: {mix_choice}")

        self.prefix = prefix
        self.image_size, self.image_mean, self.image_std = get_image_size_mean_std(
            model_name=self.prefix,
            config=self.config,
            provided_size=image_size,
            provided_norm_type=image_norm,
            support_variable_input_size=self.support_variable_input_size(),
        )
        self.image_chan_num = image_chan_num
        self.use_learnable_image = use_learnable_image
        if self.use_learnable_image:
            self.learnable_image = nn.Parameter(torch.zeros(image_chan_num, self.image_size, self.image_size))
            logger.debug("will use a learnable image to replace missing ones")

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def input_keys(self):
        return [self.image_key, self.image_valid_num_key]

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def support_variable_input_size(self):
        """Whether the TIMM image support images sizes that are different from the default used in the backbones"""
        if "test_input_size" in self.config and self.config["test_input_size"] != self.config["input_size"]:
            return True
        cls_name = type(self.model).__name__.lower()
        for k in SUPPORT_VARIABLE_INPUT_SIZE_TIMM_CLASSES:
            if cls_name in k:
                return True
        return False

    def forward(
        self,
        images: torch.FloatTensor,
        image_valid_num: torch.Tensor,
        image_column_names: Optional[List[str]] = None,
        image_column_indices: Optional[List[torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        images : torch.FloatTensor
            A tensor in [N, C, H, W] layout to represent the images.
        image_valid_num : torch.Tensor
            A tensor that describes valid number of input images.
        image_column_names : list of str, optional
            A list of strings that indicates names of the image columns.
        image_column_indices : list of torch.Tensor, optional
            A list of tensors that indicates start and stop indices of the image columns.

        Returns
        -------
            A dictionary with logits and features.
        """
        column_features = column_feature_masks = dict()
        if self.mix_choice == "all_images":  # mix inputs
            mixed_images = (
                images.sum(dim=1) / torch.clamp(image_valid_num, min=1e-6)[:, None, None, None]
            )  # mixed shape: (b, 3, h, w)
            features = self.model(mixed_images)
            if self.num_classes > 0:
                logits = self.head(features)
            else:
                logits = features

        elif self.mix_choice == "all_logits":  # mix outputs
            b, n, c, h, w = images.shape
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))  # (b, n)

            if self.use_learnable_image:
                images = replace_missing_images_with_learnable(
                    images=images,
                    image_masks=image_masks,
                    learnable_image=self.learnable_image,
                )
            features = self.model(images.reshape((b * n, c, h, w)))  # (b*n, num_features)
            if self.num_classes > 0:
                logits = self.head(features)
                logits = logits.reshape((b, n, -1))  # (b, n, num_classes)
            # reshape features after head prediction
            features = features.reshape((b, n, -1))  # (b, n, num_features)

            if not self.use_learnable_image:
                features = features * image_masks[:, :, None].type_as(features)  # (b, n, num_features)

            # need to collect column features before summing them
            if image_column_names:
                assert len(image_column_names) == len(image_column_indices), "invalid image column inputs"
                # collect features by image column names
                column_features, column_feature_masks = get_column_features(
                    batch=dict(zip(image_column_names, image_column_indices)),
                    column_name_prefix=self.image_column_prefix,
                    features=features,
                    valid_lengths=image_valid_num,
                )

            if self.use_learnable_image:
                features = features.mean(dim=1)
            else:
                features = features.sum(dim=1) / torch.clamp(image_valid_num, min=1e-6)[:, None]  # (b, num_features)
            if self.num_classes > 0:
                if self.use_learnable_image:
                    logits = logits.mean(dim=1)
                else:
                    logits = logits * image_masks[:, :, None].type_as(logits)  # (b, n, num_classes)
                    logits = logits.sum(dim=1) / torch.clamp(image_valid_num, min=1e-6)[:, None]  # (b, num_classes)
            else:
                logits = features

        else:
            raise ValueError(f"unknown mix_choice: {self.mix_choice}")

        return features, logits, column_features, column_feature_masks

    def get_output_dict(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        column_features: Optional[Dict[str, torch.Tensor]] = None,
        column_feature_masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}

        if column_features is not None and len(column_features) > 0:
            assert column_feature_masks is not None and len(column_features) == len(column_feature_masks)
            ret[COLUMN_FEATURES][FEATURES].update(column_features)
            ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)

        ret[FEATURES] = features
        if self.num_classes > 0:
            ret[LOGITS] = logits

        return {self.prefix: ret}

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Due to different backbone architectures in TIMM, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = ("embed", "cls_token", "stem", "bn1", "conv1")
        post_encoder_patterns = ("head", "norm", "bn2")
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

    def dump_config(
        self,
        config_path: str,
    ):
        """
        Save TIMM image model configs to a local file.

        Parameters
        ----------
        config_path:
            A file to where the config is written to.
        """
        from ..utils import filter_timm_pretrained_cfg

        config = {}
        pretrained_cfg = filter_timm_pretrained_cfg(self.config, remove_source=True, remove_null=True)
        # set some values at root config level
        config["architecture"] = pretrained_cfg.pop("architecture")
        config["num_classes"] = self.num_classes
        config["num_features"] = self.out_features

        global_pool_type = getattr(self, "global_pool", None)
        if isinstance(global_pool_type, str) and global_pool_type:
            config["global_pool"] = global_pool_type

        config["pretrained_cfg"] = pretrained_cfg

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            logger.info(f"Timm config saved to {config_path}.")

    def save(self, save_path: str = "./", tokenizers: Optional[dict] = None):
        weights_path = f"{save_path}/pytorch_model.bin"
        torch.save(self.model.state_dict(), weights_path)  # nosec B614
        logger.info(f"Model {self.prefix} weights saved to {weights_path}.")
        config_path = f"{save_path}/config.json"
        self.dump_config(config_path)

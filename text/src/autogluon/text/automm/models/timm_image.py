import torch
import logging
from torch import nn
from timm import create_model
from .utils import assign_layer_ids, init_weights
from ..constants import (
    IMAGE, IMAGE_VALID_NUM, LABEL,
    LOGITS, FEATURES, AUTOMM
)
from typing import Optional

logger = logging.getLogger(AUTOMM)


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
    ):
        """
        Load a pretrained image backbone from TIMM.

        Parameters
        ----------
        prefix
            The prefix of the TimmAutoModelForImagePrediction model.
        checkpoint_name
            Name of the timm checkpoint.
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
        """
        super().__init__()
        # In TIMM, if num_classes==0, then create_model would automatically set self.model.head = nn.Identity()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.model = create_model(checkpoint_name, pretrained=pretrained, num_classes=0)
        self.out_features = self.model.num_features
        self.head = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(init_weights)

        self.mix_choice = mix_choice
        logger.debug(f"mix_choice: {mix_choice}")

        self.prefix = prefix

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

    def forward(
            self,
            batch: dict,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        images = batch[self.image_key]
        image_valid_num = batch[self.image_valid_num_key]
        if self.mix_choice == "all_images":  # mix inputs
            mixed_images = images.sum(dim=1) / image_valid_num[:, None, None, None]  # mixed shape: (b, 3, h, w)
            features = self.model(mixed_images)
            logits = self.head(features)

        elif self.mix_choice == "all_logits":  # mix outputs
            b, n, c, h, w = images.shape
            features = self.model(images.reshape((b * n, c, h, w)))  # (b*n, num_features)
            logits = self.head(features)
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = (steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))).type_as(logits)  # (b, n)
            features = features.reshape((b, n, -1)) * image_masks[:, :, None]  # (b, n, num_features)
            logits = logits.reshape((b, n, -1)) * image_masks[:, :, None]  # (b, n, num_classes)
            features = features.sum(dim=1)  # (b, num_features)
            logits = logits.sum(dim=1)  # (b, num_classes)

        else:
            raise ValueError(f"unknown mix_choice: {self.mix_choice}")

        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }

    def get_layer_ids(self,):
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

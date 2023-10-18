import logging
import os
import time
import warnings
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn

from ..constants import COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, REAL_WORLD_SEM_SEG
from .utils import freeze_model_layers, lookup_mmdet_config, update_mmdet_config

logger = logging.getLogger(__name__)


class SAMForRealWorldSemSeg(nn.Module):
    """
    Support SAM for binary real-world semantic segmentation.
    Refer to https://github.com/facebookresearch/segment-anything
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        config: DictConfig,
        classes: Optional[list] = None,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
    ):
        """
        Load a pretrained object detector from MMdetection.

        Parameters
        ----------
        prefix
            The prefix of the MMdetAutoModelForObjectDetection model.
        checkpoint_name
            Name of the mmdet checkpoint.
        classes
            All classes in this dataset.
        pretrained
            Whether using the pretrained mmdet models. If pretrained=True, download the pretrained model.
        """
        from ..utils import check_if_packages_installed

        check_if_packages_installed(problem_type=REAL_WORLD_SEM_SEG)

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint = None
        self.checkpoint_name = checkpoint_name
        self.config = config
        self.classes = classes
        self.frozen_layers = frozen_layers
        self.config_file = config

        self.device = None
        self.name_to_id = {}

        # TODO: Config only init (without checkpoint)

        self._get_checkpoint_and_config_file(checkpoint_name=checkpoint_name, config_file=None)
        # self._load_config()

        # self._update_classes(classes)
        self._load_checkpoint(self.checkpoint_file)

        freeze_model_layers(self.model, self.frozen_layers)

        setattr(self.model, "forward", self.bound_forward_method)

        self.model.mask_decoder.num_mask_tokens = 1
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(1, self.model.mask_decoder.transformer_dim)
        self.model.mask_decoder.mask_tokens.weight.data[0] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])

    def _reset_classes(self, classes: list):
        temp_ckpt_file = f"temp_ckpt_{int(time.time()*1000)}.pth"
        self._save_weights(temp_ckpt_file)
        self._update_classes(classes)
        self._load_checkpoint()
        os.remove(temp_ckpt_file)

    def _update_classes(self, classes: Optional[list] = None):
        return

    def _load_checkpoint(self, checkpoint_file):
        from segment_anything import sam_model_registry

        self.model = sam_model_registry[self.checkpoint_name](checkpoint=checkpoint_file)

    def set_data_preprocessor_device(self):
        if not self.device:
            self.device = next(self.model.parameters()).device
        if self.device != self.data_preprocessor.device:
            self.data_preprocessor.to(self.device)

    def save(self, save_path: str = "./", tokenizers: Optional[dict] = None):

        weights_save_path = os.path.join(save_path, "model.pth")
        configs_save_path = os.path.join(save_path, "config.py")

        self._save_weights(save_path=weights_save_path)
        self._save_configs(save_path=configs_save_path)

        return save_path

    def _save_weights(self, save_path=None):
        if not save_path:
            save_path = f"./{self.checkpoint_name}_autogluon.pth"

        torch.save({"state_dict": self.model.state_dict(), "meta": {"CLASSES": self.model.CLASSES}}, save_path)

    def _save_configs(self, save_path=None):
        if not save_path:
            save_path = f"./{self.checkpoint_name}_autogluon.py"

        self.config.dump(save_path)

    def _get_checkpoint_and_config_file(self, checkpoint_name: str = None, config_file: str = None):
        from mim.commands import download as mimdownload

        from ..utils import download, get_pretrain_configs_dir

        logger.debug(f"initializing {checkpoint_name}")

        if not checkpoint_name:
            checkpoint_name = self.checkpoint_name
        # if not config_file:
        #     config_file = self.config_file

        AG_CUSTOM_MODELS = {
            "default": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "config_file": "",
            },
            "vit_h": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "config_file": "",
            },
        }

        if os.path.isfile(checkpoint_name):
            checkpoint_file = checkpoint_name
        elif os.path.isdir(checkpoint_name):
            checkpoint_file = os.path.join(checkpoint_name, "model.pth")
            config_file = os.path.join(checkpoint_name, "config.py")
        else:
            if checkpoint_name in AG_CUSTOM_MODELS:
                # TODO: add sha1_hash
                checkpoint_file = download(
                    url=AG_CUSTOM_MODELS[checkpoint_name]["url"],
                )
                if (
                    "source" in AG_CUSTOM_MODELS[checkpoint_name]
                    and AG_CUSTOM_MODELS[checkpoint_name]["source"] == "MegVii"
                ):
                    checkpoint_file = self.convert_megvii_yolox(checkpoint_file)
        if config_file:
            if not os.path.isfile(config_file):
                raise ValueError(f"Invalid checkpoint_name ({checkpoint_name}) or config_file ({config_file}): ")
        else:
            if checkpoint_name in AG_CUSTOM_MODELS:
                config_file = AG_CUSTOM_MODELS[checkpoint_name]["config_file"]
            else:
                try:
                    # download config and checkpoint files using openmim
                    mimdownload(package="mmdet", configs=[checkpoint_name], dest_root=".")
                    config_file = checkpoint_name + ".py"
                except Exception as e:
                    raise ValueError(f"Invalid checkpoint_name ({checkpoint_name}) or config_file ({config_file}): ")

        self.checkpoint_name = checkpoint_name
        self.checkpoint_file = checkpoint_file
        self.config_file = config_file

    def _load_config(self):
        return

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
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def forward(
        self,
        batch,
        # mode,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.
        mode
            "loss" or "predict". TODO: support "tensor"
            https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/base.py#L58C1

        Returns
        -------
            A dictionary with bounding boxes.
        """
        rets = self.model(batch[self.image_key])
        if self.training:
            return {self.prefix: {LOGITS: rets}}
        else:
            return {self.prefix: {LOGITS: rets, LABEL: batch[self.label_key]}}

    def _parse_losses(self, losses):
        return self.model._parse_losses(losses)

    def bound_forward_method(self, x):
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=None, masks=None)
        if hasattr(self.model.image_encoder, "model"):  # for lora only
            self.features = self.model.image_encoder.model(x)
        else:
            self.features = self.model.image_encoder(x)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        input_size = x.size()[-2:]
        masks = self.model.postprocess_masks(low_res_masks, input_size, input_size)

        return masks

    def get_layer_ids(self):
        return {}

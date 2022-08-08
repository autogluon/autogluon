import logging
from typing import Optional

import torch
from torch import nn
from mmdet.models import build_detector
import mmcv

from ..constants import AUTOMM, COLUMN, COLUMN_FEATURES, FEATURES, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASKS
from .utils import assign_layer_ids, get_column_features, get_model_head
from ..utils import download

logger = logging.getLogger(AUTOMM)


class MMdetAutoModelForObjectDetection(nn.Module):
    """
    Support MMDET object detection models.
    Refer to https://github.com/open-mmlab/mmdetection
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: Optional[int] = None,
        pretrained: Optional[bool] = True,
    ):
        """
        Load a pretrained object detector from MMdetection.

        Parameters
        ----------
        prefix
            The prefix of the MMdetAutoModelForObjectDetection model.
        checkpoint_name
            Name of the mmdet checkpoint.
        num_classes
            The number of classes.
        pretrained
            Whether using the pretrained mmdet models. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.pretrained = pretrained
        # TODO: model name to config
        # config = get_mmdet_model_config(checkpoint_name)
        config_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/yolo/yolov3_mobilenetv2_320_300e_coco.py'
        config_file = download(config_url)
        if isinstance(config_file, str):
            self.config = mmcv.Config.fromfile(config_file)
        self.model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg'))
        self.prefix = prefix

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
        data = batch[self.image_key]
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
            print(results)
            exit()

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

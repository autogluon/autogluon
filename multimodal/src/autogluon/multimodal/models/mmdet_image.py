import logging
import os
import warnings
from typing import Optional

import torch
from torch import nn

try:
    import mmcv
    from mmcv.ops import RoIPool
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
except ImportError:
    mmcv = None

try:
    import mmdet
    from mmdet.core import get_classes
    from mmdet.models import build_detector
except ImportError:
    mmdet = None

from ..constants import AUTOMM, BBOX, COLUMN, COLUMN_FEATURES, FEATURES, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASKS

logger = logging.getLogger(AUTOMM)


class MMDetAutoModelForObjectDetection(nn.Module):
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

        if checkpoint_name == "faster_rcnn_r50_fpn_1x_voc0712":
            # download voc configs in our s3 bucket
            from ..utils import download

            if not os.path.exists("voc_config"):
                os.makedirs("voc_config")
            # TODO: add sha1_hash
            checkpoint = download(
                url="https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth",
            )
            config_file = download(
                url="https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712.py",
            )
            download(
                url="https://automl-mm-bench.s3.amazonaws.com/voc_script/default_runtime.py",
                path="voc_config",
            )
            download(
                url="https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn.py",
                path="voc_config",
            )
            download(
                url="https://automl-mm-bench.s3.amazonaws.com/voc_script/voc0712.py",
                path="voc_config",
            )
        else:
            from mim.commands import download

            # download config and checkpoint files using openmim
            checkpoint = download(package="mmdet", configs=[checkpoint_name], dest_root=".")[0]
            config_file = checkpoint_name + ".py"

        # read config files
        assert mmcv is not None, "Please install mmcv-full by: mim install mmcv-full."
        if isinstance(config_file, str):
            self.config = mmcv.Config.fromfile(config_file)

        # build model and load pretrained weights
        assert mmdet is not None, "Please install MMDetection by: pip install mmdet."
        self.model = build_detector(self.config.model, test_cfg=self.config.get("test_cfg"))

        if checkpoint is not None:
            checkpoint = load_checkpoint(self.model, checkpoint, map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            self.model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            warnings.simplefilter("once")
            warnings.warn("Class names are not saved in the checkpoint's " "meta data, use COCO classes by default.")
            self.model.CLASSES = get_classes("coco")
        self.model.cfg = self.config  # save the config in the model for convenience

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
            A dictionary with bounding boxes.
        """
        data = batch[self.image_key]

        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        data["img"] = [img.data[0] for img in data["img"]]

        device = next(self.model.parameters()).device  # model device
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(m, RoIPool), "CPU inference with RoIPool is not supported currently."

        results = self.model(return_loss=False, rescale=True, **data)

        ret = {BBOX: results}
        return {self.prefix: ret}

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Setting all layers as the same id 0 for now.
        TODO: Need to investigate mmdetection's model definitions

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id

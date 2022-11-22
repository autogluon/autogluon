import logging
import os
import warnings
from typing import Optional

import torch
from torch import nn

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import mmcv
    from mmcv.ops import RoIPool
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
except ImportError as e:
    mmcv = None

try:
    import mmdet
    from mmdet.core import get_classes
    from mmdet.models import build_detector
except ImportError as e:
    mmdet = None

from ..constants import AUTOMM, BBOX, COLUMN, COLUMN_FEATURES, FEATURES, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASKS
from .utils import lookup_mmdet_config, update_mmdet_config

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
        classes: Optional[list] = None,
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
        classes
            All classes in this dataset.
        pretrained
            Whether using the pretrained mmdet models. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.classes = classes

        if self.classes:
            if self.num_classes:
                assert len(self.classes) == self.num_classes
            else:
                self.num_classes = len(self.classes)

        checkpoint, config_file = self._load_checkpoint_and_config()

        # read config files
        assert mmcv is not None, "Please install mmcv-full by: mim install mmcv-full."
        if isinstance(config_file, str):
            self.config = mmcv.Config.fromfile(config_file)

        if self.num_classes:
            update_mmdet_config(key="num_classes", value=self.num_classes, config=self.config)
        else:
            self.num_classes = lookup_mmdet_config(key="num_classes", config=self.config)
            if not self.num_classes:
                raise ValueError("Cannot retrieve num_classes for current model structure.")

        self.id2label = dict(zip(range(self.num_classes), range(self.num_classes)))

        # build model and load pretrained weights
        assert mmdet is not None, "Please install MMDetection by: pip install mmdet."
        self.model = build_detector(self.config.model, test_cfg=self.config.get("test_cfg"))

        if self.pretrained and checkpoint is not None:  # TODO: enable training from scratch
            checkpoint = load_checkpoint(self.model, checkpoint, map_location="cpu")

        if self.classes:
            self.model.CLASSES = self.classes
        else:
            if num_classes == 20:  # TODO: remove hardcode
                warnings.simplefilter("once")
                warnings.warn(
                    f"Using VOC classes because num_classes = {num_classes}. Provide data while init MultiModalPredictor if this is not VOC."
                )
                self.model.CLASSES = get_classes("voc")
            elif num_classes == 80:
                warnings.simplefilter("once")
                warnings.warn(
                    f"Using COCO classes because num_classes = {num_classes}. Provide data while init MultiModalPredictor if this is not COCO."
                )
                self.model.CLASSES = get_classes("coco")
            elif "CLASSES" in checkpoint.get("meta", {}):
                warnings.simplefilter("once")
                warnings.warn(
                    f"Using classes provided in checkpoints: {checkpoint['meta']['CLASSES']}. Provide data while init MultiModalPredictor if this is not expected."
                )
                self.model.CLASSES = checkpoint["meta"]["CLASSES"]
            else:
                raise ValueError("Classes need to be specified.")

        self.model.cfg = self.config  # save the config in the model for convenience

        self.prefix = prefix

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    def _load_checkpoint_and_config(self, checkpoint_name=None):
        if not checkpoint_name:
            checkpoint_name = self.checkpoint_name
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

        return checkpoint, config_file

    def dump_config(self, path):
        self.config.dump(path)

        self.name_to_id = self.get_layer_ids()

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
        # TODO: refactor this to work like forward() in MMDet, and support realtime predict
        logger.warning("MMDetAutoModelForObjectDetection.forward() is deprecated since it does not support multi gpu.")

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

    def forward_test(self, imgs, img_metas, rescale=True):
        return self.model.forward_test(imgs=imgs, img_metas=img_metas, rescale=rescale)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
        return self.model.forward_train(img=img, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

    def _parse_losses(self, losses):
        return self.model._parse_losses(losses)

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Setting all layers as the same id 0 for now.
        TODO: Need to investigate mmdetection's model definitions
        Currently only head to 0 others to 1.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        # for some models, use head lr in "head" of bbox_head
        # now support: yolov3, faster_rcnn, deformable_detr, yolox, vfnet, centernet, cascade_rcnn, detr, htc, atss, ssd
        registered_head_layers_patterns = [
            "bbox_head.fc_cls",
            "bbox_head.fc_reg",
            "bbox_head.convs_pred",
            "bbox_head.cls_branches",
            "bbox_head.multi_level_conv_cls",
            "bbox_head.vfnet_cls",
            "bbox_head.heatmap_head",
            "bbox_head.atss_cls",
            "bbox_head.cls_convs",
        ]
        # for other models, use head lr in whole bbox_head
        default_head_layers_patterns = ["bbox_head"]

        head_registered = False
        for n, _ in self.named_parameters():
            name_to_id[n] = 1
            for pattern in registered_head_layers_patterns:
                if pattern in n:
                    name_to_id[n] = 0
                    head_registered = True

        if not head_registered:
            for n, _ in self.named_parameters():
                name_to_id[n] = 1
                for pattern in default_head_layers_patterns:
                    if pattern in n:
                        name_to_id[n] = 0

        return name_to_id

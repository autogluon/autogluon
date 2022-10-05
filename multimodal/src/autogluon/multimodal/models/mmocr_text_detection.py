import logging
from typing import Optional

try:
    from mmcv.parallel import scatter
except ImportError:
    mmcv = None
try:
    from mmocr.utils.model import revert_sync_batchnorm
except ImportError:
    mmocr = None
from torch import nn

from ..constants import AUTOMM, BBOX, COLUMN, COLUMN_FEATURES, FEATURES, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASKS
from .utils import assign_layer_ids, get_column_features, get_mmocr_config_and_model, get_model_head

logger = logging.getLogger(AUTOMM)


class MMOCRAutoModelForTextDetection(nn.Module):
    """
    Support MMOCR object detection models.
    Refer to https://github.com/open-mmlab/mmocr
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: Optional[int] = None,
        pretrained: Optional[bool] = True,
    ):
        """
        Load a pretrained ocr text detection detector from MMOCR.

        Parameters
        ----------
        prefix
            The prefix of the MMdetAutoModelForTextDetection model.
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

        self.config, self.model = get_mmocr_config_and_model(checkpoint_name)
        self.model = revert_sync_batchnorm(self.model)
        self.model.cfg = self.config
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
        # single image
        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        data["img"] = [img.data[0] for img in data["img"]]

        device = next(self.model.parameters()).device  # model device
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]

        results = self.model(return_loss=False, rescale=True, **data)

        ret = {BBOX: results[0]["boundary_result"]}
        return {self.prefix: ret}

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Setting all layers as the same id 0 for now.
        TODO: Need to investigate mmocr's model definitions

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id

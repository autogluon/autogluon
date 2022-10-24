import logging
from typing import List, Optional

try:
    import mmcv
    from mmcv.parallel import collate, scatter
except ImportError:
    mmcv = None
try:
    from mmocr.utils.model import revert_sync_batchnorm
except ImportError:
    mmocr = None
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmocr.datasets.pipelines.crop import crop_img
from torch import nn

from ..constants import (
    AUTOMM,
    BBOX,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    IMAGE,
    IMAGE_VALID_NUM,
    LABEL,
    LOGITS,
    MASKS,
    SCORE,
    TEXT,
)
from .utils import assign_layer_ids, get_column_features, get_mmocr_config_and_model, get_model_head

logger = logging.getLogger(AUTOMM)


class MMOCRAutoModel(nn.Module):
    """
    Support MMOCR models.
    Refer to https://github.com/open-mmlab/mmocr
    """

    def __init__(self, prefix: str, det_ckpt_name: str, recog_ckpt_name: str, kie_ckpt_name: str):
        """
        Load a pretrained ocr text detection detector from MMOCR.

        Parameters
        ----------
        prefix
            The prefix of the MMdetAutoModel model.
        det_ckpt_name
            Name of the text detection checkpoint.
        recog_ckpt_name
            Name of the text recognition checkpoint.
        kie_ckpt_name
            Name of the key information extraction checkpoint.
        """
        super().__init__()

        self.recog_model = None
        if recog_ckpt_name != "None":
            recog_config, self.recog_model = get_mmocr_config_and_model(recog_ckpt_name)
            self.recog_model = revert_sync_batchnorm(self.recog_model)
            self.recog_model.cfg = recog_config
            self.model = self.recog_model
            self.config = recog_config

        self.det_model = None
        if det_ckpt_name != "None":
            det_config, self.det_model = get_mmocr_config_and_model(det_ckpt_name)
            self.det_model = revert_sync_batchnorm(self.det_model)
            self.det_model.cfg = det_config
            self.model = self.det_model
            self.config = det_config

        # TODO
        self.kie_model = None
        if kie_ckpt_name != "None":
            kie_config, self.kie_model = get_mmocr_config_and_model(kie_ckpt_name)

        if self.det_model != None and self.recog_model != None:
            recog_config.data.test.pipeline[0].type = "LoadImageFromNdarray"
            recog_config.data.test.pipeline = replace_ImageToTensor(recog_config.data.test.pipeline)
            self.recog_test_pipeline = Compose(recog_config.data.test.pipeline)

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
        if isinstance(data["img_metas"], List):
            data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        else:
            data["img_metas"] = data["img_metas"].data

        if isinstance(data["img"], List):
            data["img"] = [img.data[0] for img in data["img"]]
        else:
            data["img"] = data["img"].data

        device = next(self.model.parameters()).device  # model device
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]

        if self.det_model != None and self.recog_model == None:
            det_results = self.det_model(return_loss=False, rescale=True, **data)
            ret = {BBOX: det_results[0]["boundary_result"]}
            return {self.prefix: ret}
        elif self.det_model == None and self.recog_model != None:
            recog_results = self.model(return_loss=False, rescale=True, **data)
            ret = {TEXT: recog_results[0]["text"], SCORE: recog_results[0]["score"]}
            return {self.prefix: ret}
        elif self.det_model != None and self.recog_model != None:
            det_results = self.det_model(return_loss=False, rescale=True, **data)
            arrays = []
            for img_meta in data["img_metas"]:
                arrays.append(mmcv.imread(img_meta[0]["filename"]))

            bboxes_list = [res for res in det_results]
            end2end_res = []
            img_e2e_res = {"result": []}
            for bboxes, arr in zip(bboxes_list, arrays):
                for bbox in bboxes["boundary_result"]:
                    box_res = {}
                    box_res["box"] = [round(x) for x in bbox[:-1]]
                    box_res["box_score"] = float(bbox[-1])
                    box = bbox[:8]
                    if len(bbox) > 9:
                        min_x = min(bbox[0:-1:2])
                        min_y = min(bbox[1:-1:2])
                        max_x = max(bbox[0:-1:2])
                        max_y = max(bbox[1:-1:2])
                        box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
                    box_img = crop_img(arr, box)

                    # only single image now
                    data = dict(
                        img=box_img,
                        ann_info=None,
                        img_info=dict(width=box_img.shape[1], height=box_img.shape[0]),
                        bbox_fields=[],
                    )

                    data = self.recog_test_pipeline(data)
                    data = [data]
                    data = collate(data, samples_per_gpu=1)

                    if isinstance(data["img_metas"], list):
                        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
                    else:
                        data["img_metas"] = data["img_metas"].data

                    if isinstance(data["img"], list):
                        data["img"] = [img.data for img in data["img"]]
                        if isinstance(data["img"][0], list):
                            data["img"] = [img[0] for img in data["img"]]
                    else:
                        data["img"] = data["img"].data
                    if next(self.model.parameters()).is_cuda:
                        # scatter to specified GPU
                        data = scatter(data, [device])[0]

                    recog_results = self.recog_model(return_loss=False, rescale=True, **data)

                    text = recog_results[0]["text"]
                    text_score = recog_results[0]["score"]
                    if isinstance(text_score, list):
                        text_score = sum(text_score) / max(1, len(text))
                    box_res["text"] = text
                    box_res["text_score"] = text_score
                    img_e2e_res["result"].append(box_res)
                end2end_res.append(img_e2e_res)
            final_res = []
            for res in end2end_res:
                simple_res = {}
                simple_res["text"] = [x["text"] for x in res["result"]]
            final_res.append(simple_res)

            # TODO
            if self.kie_model != None:
                return 1
            ret = {TEXT: final_res[0]["text"]}
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

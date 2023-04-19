import logging
import os
from typing import Optional

import torch
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
    LOGIT_SCALE,
    LOGITS,
    MASKS,
    PROMPT,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
)
from .utils import assign_layer_ids, get_column_features, get_hf_config_and_model, init_weights

logger = logging.getLogger(__name__)


class OVDModel(nn.Module):
    """
    Support the OVD models.
    Now support GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: Optional[int] = None,
        pretrained: Optional[bool] = True,
    ):
        """
        Load the pretrained CLIP from huggingface transformers.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint.
        num_classes
            The number of classes. 1 for a regression task.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes

        self.config, self.model = self.get_ovd_config_and_model(
            checkpoint_name=checkpoint_name, pretrained=pretrained
        )  # TODO: implement this

        self.box_threshold = self.config.box_threshold
        self.text_threshold = self.config.text_threshold

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

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    def forward(
        self,
        batch: dict,
        with_logits: bool = True,
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
        from .grounding_dino.util import get_phrases_from_posmap

        def process_caption(caption):
            caption = caption.lower().strip()
            if not caption.endswith("."):
                caption = caption + "."
            return caption

        captions = [process_caption(caption) for caption in batch[PROMPT]]
        images = batch["images"]

        with torch.no_grad():
            outputs = self.model(images, captions=captions)
        logits = (
            outputs["pred_logits"].cpu().sigmoid()
        )  # (bs, nq, 256)  # TODO: will .cpu() affect the data collection of lightning?
        boxes = outputs["pred_boxes"].cpu()[0]  # (bs, nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=2)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # bs, num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # bs, num_filt, 4

        # get phrase
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(captions)
        # build pred
        pred_phrases = []
        batch_size = boxes_filt.shape[0]
        for sample_idx in range(batch_size):
            pred_phrases_per_sample = []
            for logit, box in zip(logits_filt[sample_idx], boxes_filt[sample_idx]):
                pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer)
                if with_logits:
                    pred_phrases_per_sample.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases_per_sample.append(pred_phrase)
            pred_phrases.append(pred_phrases_per_sample)

        ret = {BBOX: boxes_filt}
        ret = {PROMPT: pred_phrases}

        return ret

    def get_ovd_config_and_model(
        self,
        checkpoint_name: str,
        pretrained: bool = True,
    ):
        from ..utils import download, get_pretrain_configs_dir

        ovd_configs_dir = get_pretrain_configs_dir(subfolder="ovd")
        OVD_CUSTOM_MODELS = {
            "GroundingDINO_SwinB": {
                "url": "https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth",
                "config_file": os.path.join(ovd_configs_dir, "grounding_dino", "GroundingDINO_SwinB.cfg.py"),
            },
            "GroundingDINO_SwinT_OGC": {
                "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
                "config_file": os.path.join(ovd_configs_dir, "grounding_dino", "GroundingDINO_SwinT_OGC.py"),
            },
        }

        if checkpoint_name in ["GroundingDINO_SwinB" or "GroundingDINO_SwinT_OGC"]:
            from .grounding_dino.models import build_model
            from .grounding_dino.util import clean_state_dict, SLConfig

            model_config_path = OVD_CUSTOM_MODELS[checkpoint_name]["config_file"]

            if checkpoint_name in OVD_CUSTOM_MODELS:
                # TODO: add sha1_hash
                model_checkpoint_path = download(
                    url=OVD_CUSTOM_MODELS[checkpoint_name]["url"],
                )
            else:
                raise ValueError(
                    f"Checkpoint name not available {checkpoint_name}. All supported OVD models are {OVD_CUSTOM_MODELS.keys()}"
                )

            config = SLConfig.fromfile(model_config_path)
            config.device = "cpu"  # lighting will handle the device part
            model = build_model(config)
            if pretrained:
                checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
                load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
                logger.debug(f"load_res: \n{load_res}")
        else:
            raise ValueError(f"checkpoint name {checkpoint_name} not supported.")

        return config, model

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefixes = ["model.text_model", "model.vision_model", "model"]
        # later model prefixes can't starts with the early ones
        for i, model_pre in enumerate(model_prefixes):
            for model_pre2 in model_prefixes[i + 1 :]:
                if model_pre2.startswith(model_pre):
                    raise ValueError(
                        f"{model_pre} is a substring of {model_pre2}. Need to swap them in {model_prefixes}."
                    )

        pre_encoder_patterns = ("embeddings", "pre")
        post_encoder_patterns = ("head", "final", "post", "logit", "project")
        names = [n for n, _ in self.named_parameters()]

        name_to_id = {}
        for per_prefix in model_prefixes:
            per_model_name_to_id, names = assign_layer_ids(
                names=names,
                pre_encoder_patterns=pre_encoder_patterns,
                post_encoder_patterns=post_encoder_patterns,
                model_pre=per_prefix,
            )
            name_to_id.update(per_model_name_to_id)

        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

import logging
import os
import time
import warnings
from typing import Optional

import torch
from torch import nn

from ..constants import BBOX, BBOX_FORMATS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, XYXY
from .utils import freeze_model_layers, lookup_mmdet_config, update_mmdet_config

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import mmcv
    import mmdet
    import mmengine
    from mmdet.registry import MODELS
    from mmengine.runner import load_checkpoint
except ImportError as e:
    mmcv = None
    mmdet = None
    mmengine = None


logger = logging.getLogger(__name__)


class MMDetAutoModelForObjectDetection(nn.Module):
    """
    Support MMDET object detection models.
    Refer to https://github.com/open-mmlab/mmdetection
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        config_file: Optional[str] = None,
        classes: Optional[list] = None,
        pretrained: Optional[bool] = True,
        output_bbox_format: Optional[str] = XYXY,
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

        check_if_packages_installed(package_names=["mmcv", "mmengine", "mmdet"])

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint = None
        self.checkpoint_name = checkpoint_name
        self.config_file = config_file
        self.classes = classes
        self.frozen_layers = frozen_layers

        self.device = None

        if output_bbox_format.lower() in BBOX_FORMATS:
            self.output_bbox_format = output_bbox_format.lower()
        else:
            raise ValueError(
                f"Not supported bounding box output format for object detection: {output_bbox_format}. All supported bounding box output formats are: {BBOX_FORMATS}."
            )

        # TODO: Config only init (without checkpoint)

        self._get_checkpoint_and_config_file(checkpoint_name=checkpoint_name, config_file=config_file)
        self._load_config()

        self._update_classes(classes)
        self._load_checkpoint(self.checkpoint_file)

        freeze_model_layers(self.model, self.frozen_layers)

    def _reset_classes(self, classes: list):
        temp_ckpt_file = f"temp_ckpt_{int(time.time()*1000)}.pth"
        self._save_weights(temp_ckpt_file)
        self._update_classes(classes)
        self._load_checkpoint()
        os.remove(temp_ckpt_file)

    def _update_classes(self, classes: Optional[list] = None):
        if classes:
            self.num_classes = len(classes)
            self.classes = classes
            update_mmdet_config(key="num_classes", value=self.num_classes, config=self.config)
        else:
            self.num_classes = lookup_mmdet_config(key="num_classes", config=self.config)
            if not self.num_classes:
                self.num_classes = 1
                warnings.warn(
                    f"num_classes is not provided and is set to default value {self.num_classes} and this may cause error. Please provide sample_data_path in predictor's initialization."
                )
            self.classes = None
        self.id2label = dict(zip(range(self.num_classes), range(self.num_classes)))

    def _load_checkpoint(self, checkpoint_file):
        # build model and load pretrained weights
        from mmdet.utils import register_all_modules

        register_all_modules()  # https://github.com/open-mmlab/mmdetection/issues/9719

        self.model = MODELS.build(self.config.model)
        # yolox use self.config.model.data_preprocessor, yolov3 use self.config.data_preprocessor
        self.data_preprocessor = MODELS.build(
            self.config.data_preprocessor
            if "data_preprocessor" in self.config
            else self.config.model.data_preprocessor
        )

        if self.pretrained and checkpoint_file is not None:  # TODO: enable training from scratch
            self.checkpoint = load_checkpoint(self.model, checkpoint_file, map_location="cpu")

        # save the config and classes in the model for convenience
        self.model.cfg = self.config
        if self.classes:
            self.model.CLASSES = self.classes
        else:
            if self.checkpoint and "CLASSES" in self.checkpoint.get("meta", {}):
                warnings.simplefilter("once")
                warnings.warn(
                    f"Using classes provided in checkpoints: {self.checkpoint['meta']['CLASSES']}. Provide data while init MultiModalPredictor if this is not expected."
                )
                self.model.CLASSES = self.checkpoint["meta"]["CLASSES"]
            else:
                warnings.warn(
                    f"CLASSES is not provided and this may cause error. Please provide sample_data_path in predictor's initialization."
                )

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id <= 0]

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
        if not config_file:
            config_file = self.config_file

        mmdet_configs_dir = get_pretrain_configs_dir(subfolder="detection")

        AG_CUSTOM_MODELS = {
            "faster_rcnn_r50_fpn_1x_voc0712": {
                "url": "https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth",
                "config_file": os.path.join(mmdet_configs_dir, "voc", "faster_rcnn_r50_fpn_1x_voc0712.py"),
            },
            "yolox_nano": {
                "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
                "config_file": os.path.join(mmdet_configs_dir, "yolox", "yolox_nano_8xb8-300e_coco.py"),
                "source": "MegVii",
            },
            "yolox_tiny": {
                "url": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
                "config_file": os.path.join(mmdet_configs_dir, "yolox", "yolox_tiny_8xb8-300e_coco.py"),
            },
            "yolox_s": {
                "url": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
                "config_file": os.path.join(mmdet_configs_dir, "yolox", "yolox_s_8xb8-300e_coco.py"),
            },
            "yolox_m": {
                "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",  # Megvii weight, need more verifications
                "config_file": os.path.join(mmdet_configs_dir, "yolox", "yolox_m_8xb8-300e_coco.py"),
                "source": "MegVii",
            },
            "yolox_l": {
                "url": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
                "config_file": os.path.join(mmdet_configs_dir, "yolox", "yolox_l_8xb8-300e_coco.py"),
            },
            "yolox_l_objects365": {  # TODO: update with better pretrained weights
                "url": "https://automl-mm-bench.s3.amazonaws.com/object_detection/checkpoints/yolox/yolox_l_objects365_temp.pth",
                "config_file": os.path.join(mmdet_configs_dir, "yolox", "yolox_l_8xb8-300e_coco.py"),
            },
            "yolox_x": {
                "url": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
                "config_file": os.path.join(mmdet_configs_dir, "yolox", "yolox_x_8xb8-300e_coco.py"),
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
            else:
                # download config and checkpoint files using openmim
                checkpoint_file = mimdownload(package="mmdet", configs=[checkpoint_name], dest_root=".")[0]

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
        # read config files
        if isinstance(self.config_file, str):
            self.config = mmengine.Config.fromfile(self.config_file)
        else:
            if not isinstance(self.config_file, dict):
                raise ValueError(
                    f"The variable config_file has type {type(self.config_file)}."
                    f"Detection Model's config_file should either be a str of file path, or a dict as config."
                )

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
        mode,
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

        self.set_data_preprocessor_device()
        data = self.data_preprocessor(batch)
        rets = self.model(
            inputs=data["inputs"],
            data_samples=data["data_samples"],
            mode=mode,
        )

        if mode == "loss":
            return rets
        elif mode == "predict":
            # for detailed data structure, see https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/det_data_sample.py
            return [{BBOX: ret.pred_instances, LABEL: ret.gt_instances} for ret in rets]
        else:
            raise ValueError(f"{mode} mode is not supported.")

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
            "bbox_head.multi_level_conv_reg",
            "bbox_head.multi_level_conv_obj",
            "bbox_head.vfnet_cls",
            "bbox_head.heatmap_head",
            "bbox_head.atss_cls",
            "bbox_head.cls_convs",
        ]
        # for other models, use head lr in whole bbox_head
        default_head_layers_patterns = ["bbox_head"]

        head_registered = False
        is_yolox = False
        for n, _ in self.named_parameters():
            name_to_id[n] = 1
            for pattern in registered_head_layers_patterns:
                if pattern in n:
                    name_to_id[n] = 0
                    head_registered = True
                if "bbox_head.multi_level_conv_cls" in n:
                    is_yolox = True

        if not head_registered:
            for n, _ in self.named_parameters():
                name_to_id[n] = 1
                for pattern in default_head_layers_patterns:
                    if pattern in n:
                        name_to_id[n] = 0

        if is_yolox and "use_layer_id" in self.config:
            name_to_id = self.get_yolox_layer_ids()

        return name_to_id

    def get_yolox_layer_ids(self):
        # logic not straight forward, need to print out the model to understand
        name_to_value = {}
        for name, _ in self.named_parameters():
            n = name
            n = n.replace("backbone", "0")
            n = n.replace("neck", "1")
            n = n.replace("bbox_head", "2")

            # backbone
            n = n.replace("stem", "0")

            # neck
            n = n.replace("reduce_layers", "0")
            n = n.replace("top_down_blocks", "1")
            n = n.replace("downsamples", "2")
            n = n.replace("bottom_up_blocks", "3")
            n = n.replace("out_convs", "4")

            n = n.replace("main_conv", "0")
            n = n.replace("short_conv", "1")
            n = n.replace("final_conv", "2")
            n = n.replace("blocks", "3")

            # bbox_head
            n = n.replace("multi_level_cls_convs", "0")
            n = n.replace("multi_level_reg_convs", "0")
            n = n.replace("multi_level_conv_cls", "1")
            n = n.replace("multi_level_conv_reg", "1")
            n = n.replace("multi_level_conv_obj", "1")

            value = int("".join(c for c in n if c.isdigit()).ljust(8, "0"))
            name_to_value[name] = value

        values = list(set(name_to_value.values()))
        values.sort(reverse=True)
        value_to_id = dict(zip(values, range(len(values))))

        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = value_to_id[name_to_value[n]]
        return name_to_id

    def convert_megvii_yolox(self, source_path):
        """
        Convert YOLOX in megvii naming to mmdetection naming.
        Using code script from: https://github.com/haiyang-tju/dl_tools/blob/master/megvii_nano_2_mmdet.py
        """
        sd = source_path

        model_dict = torch.load(sd, map_location=torch.device("cpu"))
        if "state_dict" in model_dict:
            model_dict = model_dict["state_dict"]
        if "model" in model_dict:
            model_dict = model_dict["model"]

        new_dict = dict()
        for k, v in model_dict.items():
            new_k = k

            if "backbone.backbone." in k:
                new_k = k.replace("backbone.backbone.", "backbone.")
            if "backbone.dark2." in new_k:
                new_k = new_k.replace("backbone.dark2.", "backbone.stage1.")
            if "backbone.dark3." in new_k:
                new_k = new_k.replace("backbone.dark3.", "backbone.stage2.")
            if "backbone.dark4." in new_k:
                new_k = new_k.replace("backbone.dark4.", "backbone.stage3.")
            if "backbone.dark5." in new_k:
                new_k = new_k.replace("backbone.dark5.", "backbone.stage4.")
            if "dconv." in new_k:
                new_k = new_k.replace("dconv.", "depthwise_conv.")
            if "pconv." in new_k:
                new_k = new_k.replace("pconv.", "pointwise_conv.")
            if "backbone.stage1.1.conv1." in new_k:
                new_k = new_k.replace("backbone.stage1.1.conv1.", "backbone.stage1.1.main_conv.")
            if "backbone.stage1.1.conv2." in new_k:
                new_k = new_k.replace("backbone.stage1.1.conv2.", "backbone.stage1.1.short_conv.")
            if "backbone.stage1.1.conv3." in new_k:
                new_k = new_k.replace("backbone.stage1.1.conv3.", "backbone.stage1.1.final_conv.")
            if ".m." in new_k:
                new_k = new_k.replace(".m.", ".blocks.")
            if "backbone.stage2.1.conv1." in new_k:
                new_k = new_k.replace("backbone.stage2.1.conv1.", "backbone.stage2.1.main_conv.")
            if "backbone.stage2.1.conv2." in new_k:
                new_k = new_k.replace("backbone.stage2.1.conv2.", "backbone.stage2.1.short_conv.")
            if "backbone.stage2.1.conv3." in new_k:
                new_k = new_k.replace("backbone.stage2.1.conv3.", "backbone.stage2.1.final_conv.")
            if "backbone.stage3.1.conv1." in new_k:
                new_k = new_k.replace("backbone.stage3.1.conv1.", "backbone.stage3.1.main_conv.")
            if "backbone.stage3.1.conv2." in new_k:
                new_k = new_k.replace("backbone.stage3.1.conv2.", "backbone.stage3.1.short_conv.")
            if "backbone.stage3.1.conv3." in new_k:
                new_k = new_k.replace("backbone.stage3.1.conv3.", "backbone.stage3.1.final_conv.")
            if "backbone.stage4.2.conv1." in new_k:
                new_k = new_k.replace("backbone.stage4.2.conv1.", "backbone.stage4.2.main_conv.")
            if "backbone.stage4.2.conv2." in new_k:
                new_k = new_k.replace("backbone.stage4.2.conv2.", "backbone.stage4.2.short_conv.")
            if "backbone.stage4.2.conv3." in new_k:
                new_k = new_k.replace("backbone.stage4.2.conv3.", "backbone.stage4.2.final_conv.")
            if "backbone.lateral_conv0." in new_k:
                new_k = new_k.replace("backbone.lateral_conv0.", "neck.reduce_layers.0.")
            if "backbone.reduce_conv1." in new_k:
                new_k = new_k.replace("backbone.reduce_conv1.", "neck.reduce_layers.1.")
            if "backbone.C3_p4." in new_k:
                new_k = new_k.replace("backbone.C3_p4.", "neck.top_down_blocks.0.")
            if "neck.top_down_blocks.0.conv1." in new_k:
                new_k = new_k.replace("neck.top_down_blocks.0.conv1.", "neck.top_down_blocks.0.main_conv.")
            if "neck.top_down_blocks.0.conv2." in new_k:
                new_k = new_k.replace("neck.top_down_blocks.0.conv2.", "neck.top_down_blocks.0.short_conv.")
            if "neck.top_down_blocks.0.conv3." in new_k:
                new_k = new_k.replace("neck.top_down_blocks.0.conv3.", "neck.top_down_blocks.0.final_conv.")
            if "backbone.C3_p3." in new_k:
                new_k = new_k.replace("backbone.C3_p3.", "neck.top_down_blocks.1.")
            if "neck.top_down_blocks.1.conv1." in new_k:
                new_k = new_k.replace("neck.top_down_blocks.1.conv1.", "neck.top_down_blocks.1.main_conv.")
            if "neck.top_down_blocks.1.conv2." in new_k:
                new_k = new_k.replace("neck.top_down_blocks.1.conv2.", "neck.top_down_blocks.1.short_conv.")
            if "neck.top_down_blocks.1.conv3." in new_k:
                new_k = new_k.replace("neck.top_down_blocks.1.conv3.", "neck.top_down_blocks.1.final_conv.")

            if "backbone.bu_conv2." in new_k:
                new_k = new_k.replace("backbone.bu_conv2.", "neck.downsamples.0.")
            if "backbone.bu_conv1." in new_k:
                new_k = new_k.replace("backbone.bu_conv1.", "neck.downsamples.1.")

            if "backbone.C3_n3." in new_k:
                new_k = new_k.replace("backbone.C3_n3.", "neck.bottom_up_blocks.0.")
            if "neck.bottom_up_blocks.0.conv1." in new_k:
                new_k = new_k.replace("neck.bottom_up_blocks.0.conv1.", "neck.bottom_up_blocks.0.main_conv.")
            if "neck.bottom_up_blocks.0.conv2." in new_k:
                new_k = new_k.replace("neck.bottom_up_blocks.0.conv2.", "neck.bottom_up_blocks.0.short_conv.")
            if "neck.bottom_up_blocks.0.conv3." in new_k:
                new_k = new_k.replace("neck.bottom_up_blocks.0.conv3.", "neck.bottom_up_blocks.0.final_conv.")
            if "backbone.C3_n4." in new_k:
                new_k = new_k.replace("backbone.C3_n4.", "neck.bottom_up_blocks.1.")
            if "neck.bottom_up_blocks.1.conv1." in new_k:
                new_k = new_k.replace("neck.bottom_up_blocks.1.conv1.", "neck.bottom_up_blocks.1.main_conv.")
            if "neck.bottom_up_blocks.1.conv2." in new_k:
                new_k = new_k.replace("neck.bottom_up_blocks.1.conv2.", "neck.bottom_up_blocks.1.short_conv.")
            if "neck.bottom_up_blocks.1.conv3." in new_k:
                new_k = new_k.replace("neck.bottom_up_blocks.1.conv3.", "neck.bottom_up_blocks.1.final_conv.")

            if "head.stems." in new_k:
                new_k = new_k.replace("head.stems.", "neck.out_convs.")
            if "head.cls_convs." in new_k:
                new_k = new_k.replace("head.cls_convs.", "bbox_head.multi_level_cls_convs.")
            if "head.reg_convs." in new_k:
                new_k = new_k.replace("head.reg_convs.", "bbox_head.multi_level_reg_convs.")
            if "head.cls_preds." in new_k:
                new_k = new_k.replace("head.cls_preds.", "bbox_head.multi_level_conv_cls.")
            if "head.reg_preds." in new_k:
                new_k = new_k.replace("head.reg_preds.", "bbox_head.multi_level_conv_reg.")
            if "head.obj_preds." in new_k:
                new_k = new_k.replace("head.obj_preds.", "bbox_head.multi_level_conv_obj.")

            if "bbox_head.multi_level_conv_cls." in new_k:
                if self.classes:
                    new_dict[new_k] = v[: len(self.classes), ...]  # there take the num_classes
                else:
                    new_dict[new_k] = v
            else:
                new_dict[new_k] = v

        data = {"state_dict": new_dict}

        target_directory = os.path.splitext(sd)[0] + f"_cvt.pth"
        torch.save(data, target_directory)

        return target_directory

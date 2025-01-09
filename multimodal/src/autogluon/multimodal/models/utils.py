import functools
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch._dynamo
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer, CLIPTokenizer, ElectraTokenizer
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import (
    ALL_MODALITIES,
    CATEGORICAL,
    CATEGORICAL_MLP,
    CLASS_LOGITS,
    CLIP,
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    DOCUMENT,
    DOCUMENT_TRANSFORMER,
    FT_TRANSFORMER,
    FUSION_MLP,
    FUSION_NER,
    FUSION_TRANSFORMER,
    HF_TEXT,
    IMAGE,
    LOGITS,
    META_TRANSFORMER,
    MMDET_IMAGE,
    MMOCR_TEXT_DET,
    MMOCR_TEXT_RECOG,
    NER_TEXT,
    NUMERICAL,
    NUMERICAL_MLP,
    OCR,
    PEFT_ADDITIVE_STRATEGIES,
    REGRESSION,
    SAM,
    SEMANTIC_MASK,
    SEMANTIC_SEGMENTATION,
    SEMANTIC_SEGMENTATION_IMG,
    T_FEW,
    TEXT,
    TEXT_NER,
    TIMM_IMAGE,
)
from .adaptation_layers import ConvLoRALinear, IA3Linear, IA3LoRALinear, LoRALinear

logger = logging.getLogger(__name__)


ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    "hf_auto": AutoTokenizer,
}


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias.to(x) - self.dummy_bias.to(x)


def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def assign_encoder_layer_ids(
    encoder_names: List[List[str]],
):
    """
    Assign ids to encoder layers. The encoder may contain several blocks e.g., block1 and block2.
    This function iterates through all the layers of each block from the input end towards the output end.
    It increases 1 on the layer id when the detected digit in a layer name changes.

    Parameters
    ----------
    encoder_names
        Encoder layer names.

    Returns
    -------
    name_to_id
        The encoder layer-to-id mapping.
    encoder_layer_num
        The encoder layer number.
    """
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split(".")

            for split in n_splits:
                # the first digit encountered is used to infer layer id
                if split.isdigit():
                    inferred_id = int(split)
                    # increase at most 1 one time
                    if inferred_id != last_inferred_id:
                        cur_id += 1  # layer.0 -> layer_id 1
                        last_inferred_id = inferred_id

                    name_to_id[n] = cur_id
                    detect_id = True
                    break

            if detect_id is False:
                raise ValueError(f"parameter name: {n} not has no id inside")

    if len(name_to_id) > 0:
        encoder_layer_num = max(name_to_id.values())
    else:
        encoder_layer_num = 0
    return name_to_id, encoder_layer_num


def assign_non_encoder_layer_ids(
    non_encoder_names: List[str],
    layer_id: int,
):
    """
    Assign the provided id to non-encoder layers.

    Parameters
    ----------
    non_encoder_names
        Names layers not belonging to an encoder.
    layer_id
        provided id.

    Returns
    -------
    A dictionary mapping the layer names (keys) to their ids (values).
    """
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: List[str], post_encoder_patterns: Tuple[str, ...]):
    """
    Group layer names into two types: encoder and non-encoder.
    A layer belongs to encoder if its name contains at least one digit.
    It uses this rule since a model's encoder in Pytorch's implementation
    is generally wrapped by nn.Sequential() or nn.ModuleList(),
    which produce digits in layer names.

    Parameters
    ----------
    names
        Model layer names.
    Returns
    -------
    encoder_names
        A list of encoder layer names.
    non_encoder_names
        A list of non-encoder layer names.
    """
    encoder_names = []
    non_encoder_names = []
    for n in names:
        is_encoder = False
        if any(p in n for p in post_encoder_patterns):
            non_encoder_names.append(n)
            continue
        for i in n.split("."):
            if i.isdigit():
                encoder_names.append(n)
                is_encoder = True
                break
        if not is_encoder:
            non_encoder_names.append(n)

    return encoder_names, non_encoder_names


def group_param_names(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_prefix: Optional[str] = None,
):
    """
    Group layer names into three types: pre-encoder, encoder, and post-encoder.
    If "model_prefix" is provided, the selected layer names must start with it.
    In this case, the left names will be returned for the next-time processing.
    This function first extracts the first-level children modules' names and
    classify them into encoder and non-encoder layers. Note that an encoder may
    consist of several manually named children modules, e.g., block1 and block2.
    The non-encoder layers are further subdivided into pre-encoder and post-encoder.

    Parameters
    ----------
    names
        Model layer names
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_prefix
        A prefix to filter layer names. Only layer names starting with it will be selected.
    Returns
    -------
    left_names
        The layer names left for the next-time processing.
    encoder_names_grouped
        Encoder layer names.
    pre_encoder_names
        Names of layers before the encoder.
    post_encoder_names
        Names of layers after the encoder.
    """
    # two set of patterns can't have intersections
    assert all(pre_p not in post_encoder_patterns for pre_p in pre_encoder_patterns)

    left_names = []
    # in case model_prefix is provided, e.g., the clip model with image and text branches
    selected_names = []
    for n in names:
        if model_prefix is not None and not n.startswith(model_prefix):
            left_names.append(n)
        else:
            selected_names.append(n)

    # split blocks at the first level
    children_prefix = []
    for n in selected_names:
        if model_prefix is not None:
            child_name = n[len(model_prefix) + 1 :].split(".")[0]
            child_prefix = f"{model_prefix}.{child_name}"
        else:
            child_prefix = n.split(".")[0]
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)

    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group, post_encoder_patterns)
        encoder_names_grouped.append(per_encoder_names)
        non_encoder_names.extend(per_non_encoder_names)

    pre_encoder_names = []
    post_encoder_names = []
    for n in non_encoder_names:
        if any(p in n for p in pre_encoder_patterns):
            pre_encoder_names.append(n)
        elif any(p in n for p in post_encoder_patterns):
            post_encoder_names.append(n)
        else:
            raise ValueError(f"parameter name: {n} belong to neither pre or post encoder names")

    # only process left names in next iteration
    return left_names, encoder_names_grouped, pre_encoder_names, post_encoder_names


def reverse_layer_ids(
    encoder_name_to_id: dict,
    pre_enocder_name_to_id: dict,
    post_enocder_name_to_id: dict,
):
    """
    The layer ids need to increase when going from the output end to the input end.
    We need to reverse the ids which were originally assigned in a decreasing order.

    Parameters
    ----------
    encoder_name_to_id
        The layer-to-id mapping of encoder layers.
    pre_enocder_name_to_id
        The layer-to-id mapping of pre-encoder layers.
    post_enocder_name_to_id
        The layer-to-id mapping of post-encoder layers.

    Returns
    -------
    The layer-to-id mapping of all layers with layer ids reversed.
    """
    name_to_id = {**pre_enocder_name_to_id, **encoder_name_to_id, **post_enocder_name_to_id}
    if len(name_to_id) > 0:
        layer_num = max(name_to_id.values())
        # if no post encoder layers, the minimum layer id should be 1
        if len(post_enocder_name_to_id) == 0:
            layer_num += 1
    for n, layer_id in name_to_id.items():
        name_to_id[n] = layer_num - layer_id

    return name_to_id


def assign_layer_ids(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_pre: Optional[str] = None,
):
    """
    Assign ids to all layers. It splits a model into three parts: pre-encoder, encoder, and post-encoder.
    Encoder is generally a stack of multiple similar layers, such as transformer layers. Since encoder is
    generally wrapped by nn.Sequential() or nn.ModuleList(), its inside layer names contain digits.
    It sets 0 as the ids of all post-encoder layers and a maximum id (layer_num) for the all the pre-encoder
    layers. The encoder layers have decreasing ids from the input to the output ends.

    Parameters
    ----------
    names
        model layer names.
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_pre
        The layer names' prefix. Only the layer names with this prefix will be assigned ids. The left
        layer names will be returned.

    Returns
    -------
    name_to_id
        A dictionary mapping the layer names (keys) to their ids (values).
    left_names
        The layer names not starting with the "model_pre".
    """
    try:
        left_names, encoder_names, pre_encoder_names, post_encoder_names = group_param_names(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_prefix=model_pre,
        )
        # add a constraint
        if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
            raise ValueError(f"encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}")

        encoder_name_to_id, encoder_layer_num = assign_encoder_layer_ids(
            encoder_names=encoder_names,
        )

        pre_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=pre_encoder_names, layer_id=0)

        post_encoder_name_to_id = assign_non_encoder_layer_ids(
            non_encoder_names=post_encoder_names, layer_id=encoder_layer_num + 1
        )

        name_to_id = reverse_layer_ids(
            encoder_name_to_id=encoder_name_to_id,
            pre_enocder_name_to_id=pre_encoder_name_to_id,
            post_enocder_name_to_id=post_encoder_name_to_id,
        )
    except Exception as e:
        logger.debug(
            f"When calling assign_layer_ids(), it catches exception: {e}. All the layers will use the same layer_id."
        )
        name_to_id = dict()
        left_names = names

    return name_to_id, left_names


def get_column_features(
    batch: Dict[str, torch.Tensor],
    column_name_prefix: str,
    features: torch.Tensor,
    valid_lengths: torch.Tensor,
    cls_feature: Optional[torch.Tensor] = None,
):
    """
    Index the features of one column defined by `column_name_prefix`.
    This function can be used to index both image and text features.
    The features have shape (b, n, d), where n can be the image number or
    text token number. One column corresponds to a subset of
    the n images or text tokens. One column name can only appear once in the return.

    Parameters
    ----------
    batch
        The batch input containing the feature column information, i.e., indexes.
    column_name_prefix
        The column name prefix of one modality (image or text).
    features
        The features of columns whose names starts with column_name_prefix.
    valid_lengths
        The valid image number or text token number of each sample in a batch.
    cls_feature
        The cls feature containing information from all feature columns.

    Returns
    -------
    The column features with masks. If the column has no valid features, its
    mask is 0.
    """
    column_features = {}
    feature_masks = {}

    cut_idx = len(column_name_prefix) + 1
    if cls_feature is not None:
        all_column_names = []
        # create a zero mask to do logical_or with each column's mask
        joint_mask = torch.zeros(features.shape[0]).to(features)  # (b,)
    for key in batch:
        if key.startswith(column_name_prefix):
            per_col_features = []
            per_col_masks = torch.zeros(features.shape[0]).to(features)  # (b,)
            assert batch[key].ndim == 2 and batch[key].shape[1] == 2
            for i, per_sample_col_idx in enumerate(batch[key]):
                start_idx = per_sample_col_idx[0]
                end_idx = per_sample_col_idx[1]
                if start_idx < end_idx:
                    assert end_idx <= valid_lengths[i]
                    per_col_features.append(features[i, start_idx:end_idx].mean(dim=0))
                    per_col_masks[i] = 1
                else:  # the column has no valid image/text.
                    per_col_features.append(torch.zeros_like(features[0, 0]))
                    per_col_masks[i] = 0
            column_name = key[cut_idx:]
            column_features[column_name] = torch.stack(per_col_features, dim=0)  # (b, num_features)
            feature_masks[column_name] = per_col_masks  # (b,)
            if cls_feature is not None:
                all_column_names.append(column_name)
                joint_mask = torch.logical_or(joint_mask, per_col_masks)

    # all the columns of one model's input share the model's cls feature
    if (
        cls_feature is not None and len(all_column_names) > 0
    ):  # some models', e.g, timm_image, output doesn't have the cls feature.
        # remove the individual column features since these column features not independent
        for column_name in all_column_names:
            column_features.pop(column_name)
            feature_masks.pop(column_name)

        joint_column_name = "_".join(all_column_names)
        column_features[joint_column_name] = cls_feature
        feature_masks[joint_column_name] = joint_mask.to(features)

    # print(f"column_features: {column_features}")
    return column_features, feature_masks


def create_adaptation(peft: str, layer: nn.Module, lora_r: int, lora_alpha: int, **kwargs):
    """
    Creates a model adaptation module (IA3, LoRA, IA3_LoRA) given a linear layer.

    Parameters
    ----------
    peft
        Name of the adaptation module.
    layer
       The layer the adaptation module should be applied to.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules

    Returns
    -------
    Model with injected LoRA modules.
    """
    if "ia3_lora" in peft:
        return IA3LoRALinear(
            layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False
        )
    elif "conv_lora" in peft:
        return ConvLoRALinear(
            layer.in_features,
            layer.out_features,
            r=lora_r,
            lora_alpha=lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=kwargs["conv_lora_expert_num"],
        )
    elif "ia3" in peft:
        return IA3Linear(layer.in_features, layer.out_features, merge_weights=False)
    elif "lora" in peft:
        return LoRALinear(layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
    elif peft is not None:
        raise NotImplementedError(
            f"The efficient finetuning strategy '{peft}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_ADDITIVE_STRATEGIES)}."
        )


def inject_adaptation_to_linear_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    Injects trainable adatio Low-Rank decomposition matrices (LoRA) into linear
    layers of a PyTorch model. Used for efficient fine-tuning of large
    pre-trained models.

    Parameters
    ----------
    model
        A PyTorch model.
    peft
        Efficient finetuning method that should be applied.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules
    extra_trainable_params
        Not to apply loRA to modules filtered by name, and these modules are not frozen during training (e.g. "mask_decoder").
        If None, all the modules except for those applied loRA are frozen.
    Returns
    -------
    Model with injected LoRA modules.
    """
    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(
            re.search(filter_layer, m_name) for filter_layer in model.frozen_layers
        ):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    adaptation_layer = create_adaptation(peft, layer, lora_r, lora_alpha, **kwargs)
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model  # return model to enable method chaining


def get_model_head(model: nn.Module):
    """
    Return the model's head. Different models may have different head names.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    The model's head.
    """
    if hasattr(model, "head"):
        head = model.head  # move the head outside
    elif hasattr(model, "last_linear"):
        head = model.last_linear
    elif hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier
    else:
        raise ValueError(f"Model {type(model)} doesn't have head. Need to check its implementation.")

    return head.fc if hasattr(head, "fc") else head


def get_hf_config_and_model(
    checkpoint_name: str, pretrained: Optional[bool] = True, low_cpu_mem_usage: Optional[bool] = False
):
    """
    Get a Huggingface config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name or a local path that saves a custom checkpoint.
    pretrained
         Whether using the pretrained weights. If pretrained=True, download the pretrained model.
    low_cpu_mem_usage
        Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.

    Returns
    -------
    A Huggingface config and model.
    """
    config = AutoConfig.from_pretrained(checkpoint_name)

    if pretrained:
        model = AutoModel.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage)
    else:
        model = AutoModel.from_config(config)

    return config, model


def apply_sigmoid(output: Dict):
    """
    Apply the sigmoid to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with logits transformed by sigmoid.
    """
    for k, v in output.items():
        output[k][LOGITS] = torch.sigmoid(v[LOGITS].float())
    return output


def apply_multi_class_semantic_seg_postprocess(output: Dict):
    """
    Apply the semantic postprocessing to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with post-proceesed semantic masks.
    """

    def semantic_inference(mask_cls, mask_pred):
        """
        Post-processing mask prediction for multi-class semantic segmentation inference based on https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py

        Args:
            mask_cls (`torch.Tensor`):
                Class logits. A tensor of shape `(num_queries, num_classes + 1)` (include the "no object" category).

            mask_pred (`torch.Tensor`):
                Mask logits. A tensor of shape `(num_queries, height, width)`.

        Returns:
            semseg (`torch.Tensor`): The processed mask prediction. A tensor of shape `(num_classes, height, width)`.

        References:
        [1] https://arxiv.org/abs/2107.06278
        [2] https://arxiv.org/abs/2112.01527
        """
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    for k, v in output.items():
        pred_classes = output[k][CLASS_LOGITS]
        pred_masks = output[k][LOGITS]
        semantic_masks = []
        for mask_cls_result, mask_pred_result in zip(
            pred_classes, pred_masks
        ):  # bs, num_q, num_class and bs, num_q, h, w
            per_sample_semantic_masks = semantic_inference(mask_cls_result, mask_pred_result)  # num_class, h, w
            semantic_masks.append(per_sample_semantic_masks)
        semantic_masks = torch.stack(semantic_masks, dim=0)
        output[k][SEMANTIC_MASK] = semantic_masks
    return output


def get_model_postprocess_fn(problem_type: str, loss_func: _Loss):
    """
    Get the postprocessing function for the model outputs.

    Parameters
    ----------
    problem_type
        The problem type, e.g., classification or regression.
    loss_func
        The loss function used in training.

    Returns
    -------
    The postprocessing function.
    """
    postprocess_func = None
    if problem_type == REGRESSION:
        if isinstance(loss_func, nn.BCEWithLogitsLoss):
            postprocess_func = apply_sigmoid
    elif problem_type == SEMANTIC_SEGMENTATION:
        if isinstance(loss_func, Mask2FormerLoss):
            postprocess_func = apply_multi_class_semantic_seg_postprocess
        else:
            postprocess_func = apply_sigmoid

    return postprocess_func


def get_mmocr_config_and_model(checkpoint_name: str):
    """
    Get an MMOCR config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name.

    Returns
    -------
    An MMOCR config and model.
    """
    from ..utils import check_if_packages_installed

    check_if_packages_installed(package_names=["mmcv"])
    try:
        import mmocr
        from mmocr.models import build_detector
    except ImportError:
        mmocr = None
    from mim.commands.download import download

    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")

    # read config files
    check_if_packages_installed(problem_type=OCR)
    config_file = checkpoint_name + ".py"
    if isinstance(config_file, str):
        config = mmcv.Config.fromfile(config_file)

    # build model and load pretrained weights
    checkpoint = checkpoints[0]
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    return config, model


def lookup_mmdet_config(key, config):
    if key in config:
        return config[key]
    for subconfig in config.values():
        if isinstance(subconfig, dict):
            result = lookup_mmdet_config(key, subconfig)
            if result is not None:
                return result
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    result = lookup_mmdet_config(key, subsubconfig)
                    if result is not None:
                        return result
    return None


def update_mmdet_config(key, value, config):
    for k, subconfig in config.items():
        if key == k:
            config[k] = value
        elif isinstance(subconfig, dict):
            update_mmdet_config(key, value, subconfig)
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    update_mmdet_config(key, value, subsubconfig)


def run_model(model: nn.Module, batch: dict, trt_model: Optional[nn.Module] = None):
    from ..utils.onnx import OnnxModule
    from .document_transformer import DocumentTransformer
    from .fusion.fusion_mlp import MultimodalFusionMLP
    from .hf_text import HFAutoModelForTextPrediction
    from .t_few import TFewModel
    from .timm_image import TimmAutoModelForImagePrediction

    supported_models = (
        TimmAutoModelForImagePrediction,
        HFAutoModelForTextPrediction,
        MultimodalFusionMLP,
        TFewModel,
        OnnxModule,
    )
    pure_model = model
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        pure_model = model._orig_mod
    if isinstance(model, nn.DataParallel):
        pure_model = model.module
    if isinstance(pure_model, OnnxModule):
        for k in batch:
            # HACK input data types in ONNX
            if batch[k].dtype == torch.int32:
                batch[k] = batch[k].to(torch.int64)
    # DocumentTransformer inherited from HFAutoModelForTextPrediction
    if (not isinstance(pure_model, DocumentTransformer)) and isinstance(pure_model, supported_models):
        input_vec = [batch[k] for k in pure_model.input_keys]
        column_names, column_values = [], []
        for k in batch.keys():
            has_image_column_prefix = isinstance(pure_model, TimmAutoModelForImagePrediction) and k.startswith(
                pure_model.image_column_prefix
            )
            has_text_column_prefix = isinstance(pure_model, HFAutoModelForTextPrediction) and k.startswith(
                pure_model.text_column_prefix
            )
            if has_image_column_prefix or has_text_column_prefix:
                column_names.append(k)
                column_values.append(batch[k])
        if column_names != [] and column_values != []:
            input_vec.append(column_names)
            input_vec.append(column_values)
        if isinstance(pure_model, OnnxModule):
            # OnnxModule doesn't support multi-gpu yet
            output_vec = pure_model(*tuple(input_vec))
        else:
            output_vec = model(*tuple(input_vec))

        output = pure_model.get_output_dict(*output_vec)
    else:
        output = model(batch)
    return output


def freeze_model_layers(model, frozen_layers):
    """
    Freeze model layers with pattern in frozen_layers.

    Parameters
    ----------
    model
        The pytorch model.
    frozen_layers
        A list of substrings of frozen layers' names.

        e.g. if frozen_layers = ["backbone", "neck"],
            all layers including "backbone" or "neck" in the name will be frozen.
    """

    if not frozen_layers:
        return

    is_frozen_layer = lambda n: any(bb in n for bb in frozen_layers)

    for n, p in model.named_parameters():
        if is_frozen_layer(n):
            p.requires_grad = False


def get_pretrained_tokenizer(
    tokenizer_name: str,
    checkpoint_name: str,
    use_fast: Optional[bool] = True,
    add_prefix_space: Optional[bool] = None,
):
    """
    Load the tokenizer for a pre-trained huggingface checkpoint.

    Parameters
    ----------
    tokenizer_name
        The tokenizer type, e.g., "bert", "clip", "electra", and "hf_auto".
    checkpoint_name
        Name of a pre-trained checkpoint.
    use_fast
        Use a fast Rust-based tokenizer if it is supported for a given model.
        If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
        See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast

    Returns
    -------
    A tokenizer instance.
    """
    try:
        tokenizer_class = ALL_TOKENIZERS[tokenizer_name]
        if add_prefix_space is None:
            return tokenizer_class.from_pretrained(checkpoint_name, use_fast=use_fast)
        else:
            return tokenizer_class.from_pretrained(
                checkpoint_name, use_fast=use_fast, add_prefix_space=add_prefix_space
            )
    except TypeError as e:
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_name)
            warnings.warn(
                f"Current checkpoint {checkpoint_name} does not support AutoTokenizer. "
                "Switch to BertTokenizer instead.",
                UserWarning,
            )
            return tokenizer
        except:
            raise e


def extract_value_from_config(
    config: Dict,
    keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some hyper-parameter's value.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += extract_value_from_config(v, keys)
        else:
            pass

    return result


def extract_image_hparams_from_config(model_name: str, config):
    """
    Extract some default hyper-parameters, e.g., image size, mean, and std,
    from a pre-trained (timm or huggingface) checkpoint.

    Parameters
    ----------
    model_name
        Name of model.
    config
        Config of a pre-trained checkpoint.

    Returns
    -------
    image_size
        Image width/height.
    mean
        Image normalization mean.
    std
        Image normalizaiton std.
    """
    if model_name.lower().startswith((TIMM_IMAGE, META_TRANSFORMER)):
        image_size = config["input_size"][-1]
        image_mean = config["mean"]
        image_std = config["std"]
    elif model_name.lower().startswith((CLIP, DOCUMENT_TRANSFORMER)):
        extracted = extract_value_from_config(
            config=config.to_diff_dict(),
            keys=("image_size",),
        )
        if len(extracted) == 0:
            image_size = None
        elif len(extracted) >= 1:
            image_size = extracted[0]
            if isinstance(image_size, tuple):
                image_size = image_size[-1]
        else:
            raise ValueError(f" more than one image_size values are detected: {extracted}")
        image_mean = None
        image_std = None
    else:
        raise ValueError(f"Unknown image processor prefix: {model_name}")
    return image_size, image_mean, image_std


def image_mean_std(norm_type: str):
    """
    Get image normalization mean and std by its name.

    Parameters
    ----------
    norm_type
        Name of image normalization.

    Returns
    -------
    Normalization mean and std.
    """
    if norm_type == "inception":
        return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif norm_type == "imagenet":
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    elif norm_type == "clip":
        return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    else:
        raise ValueError(f"unknown image normalization: {norm_type}")


def get_image_size_mean_std(
    model_name: str,
    config,
    provided_size: int,
    provided_norm_type: str,
    support_variable_input_size: Optional[bool] = False,
):
    image_size, image_mean, image_std = extract_image_hparams_from_config(
        model_name=model_name,
        config=config,
    )
    if support_variable_input_size and provided_size is not None:
        # We have detected that the model supports using an image size that is
        # different from the pretrained model, e.g., ConvNets with global pooling
        if provided_size < image_size:
            logger.warning(
                f"The provided image size={provided_size} is smaller than the default size "
                f"of the pretrained backbone, which is {image_size}. "
                f"Detailed configuration of the backbone is in {config}. "
                f"You may like to double check your configuration."
            )
        image_size = provided_size
    elif provided_size is not None and provided_size != image_size:
        logger.warning(
            f"The model does not support using an image size that is different from the default size. "
            f"Provided image size={provided_size}. Default size={image_size}. "
            f"Detailed model configuration={config}. We have ignored the provided image size."
        )

    if image_size is None:
        if provided_size is not None:
            image_size = provided_size
            logger.debug(f"using provided image size: {image_size}.")
        else:
            raise ValueError("image size is missing.")
    else:
        logger.debug(f"using detected image size: {image_size}")

    if image_mean is None or image_std is None:
        if provided_norm_type is not None:
            image_mean, image_std = image_mean_std(provided_norm_type)
            logger.debug(f"using provided normalization: {provided_norm_type}.")
        else:
            raise ValueError("image normalization mean and std are missing.")
    else:
        logger.debug(f"using detected image normalization: {image_mean} and {image_std}.")

    return image_size, image_mean, image_std


def get_text_segment_num(config, provided_segment_num: int, checkpoint_name: str):
    extracted = extract_value_from_config(config=config.to_diff_dict(), keys=("type_vocab_size",))
    if len(extracted) == 0:
        default_segment_num = 1
    elif len(extracted) == 1:
        default_segment_num = extracted[0]
    else:
        raise ValueError(f" more than one type_vocab_size values are detected: {extracted}")

    if default_segment_num <= 0:
        default_segment_num = 1

    if provided_segment_num < default_segment_num:
        warnings.warn(
            f"provided text_segment_num: {provided_segment_num} "
            f"is smaller than {checkpoint_name}'s default: {default_segment_num}"
        )
    text_segment_num = min(provided_segment_num, default_segment_num)
    assert text_segment_num >= 1
    logger.debug(f"text segment num: {text_segment_num}")

    return text_segment_num


def get_text_token_max_len(provided_max_len, config, tokenizer, checkpoint_name):
    """
    Compute the allowable max length of token sequences.

    Parameters
    ----------
    provided_max_len
        The provided max length.
    config
        Model config.
    tokenizer
        Text tokenizer.
    checkpoint_name
        Name of checkpoint.

    Returns
    -------
    Token sequence max length.
    """
    if hasattr(config, "relative_attention") and config.relative_attention:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "position_embedding_type") and "relative" in config.position_embedding_type:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "max_position_embeddings"):
        default_max_len = config.max_position_embeddings
    else:
        default_max_len = tokenizer.model_max_length

    if provided_max_len is None or provided_max_len <= 0:
        max_len = default_max_len
    else:
        if provided_max_len < default_max_len:
            if default_max_len < 10**6:  # Larger than this value usually means infinite.
                warnings.warn(
                    f"provided max length: {provided_max_len} "
                    f"is smaller than {checkpoint_name}'s default: {default_max_len}"
                )
        max_len = min(provided_max_len, default_max_len)

    logger.debug(f"text max length: {max_len}")

    return max_len


def replace_missing_images_with_learnable(
    images: torch.Tensor,
    image_masks,
    learnable_image: nn.Parameter,
):
    b, n, c, h, w = images.shape
    assert learnable_image.shape == (c, h, w)
    for i in range(b):
        for j in range(n):
            if not image_masks[i][j]:  # False means a missing image
                images[i][j] = learnable_image

    return images


def select_model(
    config: DictConfig,
    df_preprocessor,
    strict: Optional[bool] = True,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the default, including hf_text and timm_image, as the model config template
    but the training data don't have images, this function will filter out timm_image.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.
    strict
        If False, allow retaining one model when partial modalities are available for that model.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_feature_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True
    if len(df_preprocessor.ner_feature_names) > 0:
        data_status[TEXT_NER] = True
    if len(df_preprocessor.document_feature_names) > 0:
        data_status[DOCUMENT] = True
    if len(df_preprocessor.semantic_segmentation_feature_names) > 0:
        data_status[SEMANTIC_SEGMENTATION_IMG] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        strict = getattr(model_config, "requires_all_dtypes", strict)
        if not model_config.data_types:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            if any(model_data_status) and not strict:
                selected_model_names.append(model_name)
                # update data types to be consistent with detected
                model_config.data_types = [d_type for d_type in model_config.data_types if data_status[d_type]]
            else:
                delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    if len(fusion_model_name) > 1:
        raise ValueError(f"More than one fusion models `{fusion_model_name}` are detected, but only one is allowed.")

    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    elif len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
        delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")
    for model_name in selected_model_names:
        logger.debug(f"model dtypes: {getattr(config.model, model_name).data_types}")

    # clean up unused model configs
    model_keys = list(config.model.keys())
    for model_name in model_keys:
        if model_name not in selected_model_names + ["names"]:
            delattr(config.model, model_name)

    return config


def create_model(
    model_name: str,
    model_config: DictConfig,
    num_classes: Optional[int] = 0,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
    is_matching: Optional[bool] = False,
):
    """
    Create a single model.

    Parameters
    ----------
    model_name
        Name of the model.
    model_config
        Config of the model.
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
    is_matching
        Whether the model is used for semantic matching.

    Returns
    -------
    A model.
    """
    if model_name.lower().startswith(CLIP):
        from .clip import CLIPForImageText

        model = CLIPForImageText(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            is_matching=is_matching,
        )
    elif model_name.lower().startswith(TIMM_IMAGE):
        from .timm_image import TimmAutoModelForImagePrediction

        model = TimmAutoModelForImagePrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            mix_choice=model_config.mix_choice,
            pretrained=pretrained,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
        )
    elif model_name.lower().startswith(HF_TEXT):
        from .hf_text import HFAutoModelForTextPrediction

        model = HFAutoModelForTextPrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            use_fast=model_config.use_fast,
        )
    elif model_name.lower().startswith(T_FEW):
        from .t_few import TFewModel

        model = TFewModel(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            length_norm=model_config.length_norm,  # Normalizes length to adjust for length bias in target template
            unlikely_loss=model_config.unlikely_loss,  # Adds loss term that lowers probability of incorrect outputs
            mc_loss=model_config.mc_loss,  # Adds multiple choice cross entropy loss
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    elif model_name.lower().startswith(NUMERICAL_MLP):
        from .numerical_mlp import NumericalMLP

        model = NumericalMLP(
            prefix=model_name,
            in_features=num_numerical_columns,
            hidden_features=model_config.hidden_size,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            token_dim=model_config.token_dim,
            embedding_arch=model_config.embedding_arch,
            num_classes=num_classes,
            numerical_fill_values=numerical_fill_values,
        )
    elif model_name.lower().startswith(CATEGORICAL_MLP):
        from .categorical_mlp import CategoricalMLP

        model = CategoricalMLP(
            prefix=model_name,
            num_categories=num_categories,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            num_classes=num_classes,
        )
    elif model_name.lower().startswith(DOCUMENT_TRANSFORMER):
        from .document_transformer import DocumentTransformer

        model = DocumentTransformer(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(MMDET_IMAGE):
        from .mmdet_image import MMDetAutoModelForObjectDetection

        model = MMDetAutoModelForObjectDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            config_file=model_config.config_file,
            classes=classes,
            pretrained=pretrained,
            output_bbox_format=model_config.output_bbox_format,
            frozen_layers=model_config.frozen_layers,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_DET):
        from .mmocr_text_detection import MMOCRAutoModelForTextDetection

        model = MMOCRAutoModelForTextDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_RECOG):
        from .mmocr_text_recognition import MMOCRAutoModelForTextRecognition

        model = MMOCRAutoModelForTextRecognition(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(NER_TEXT):
        from .ner_text import HFAutoModelForNER

        model = HFAutoModelForNER(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
        )
    elif model_name.lower().startswith(FUSION_MLP):
        from .fusion import MultimodalFusionMLP

        model = functools.partial(
            MultimodalFusionMLP,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            aux_loss_weight=model_config.aux_loss_weight,
        )
    elif model_name.lower().startswith(FUSION_NER):
        from .fusion import MultimodalFusionNER

        model = functools.partial(
            MultimodalFusionNER,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout_prob=model_config.drop_rate,
            normalization=model_config.normalization,
            loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
        )
    elif model_name.lower().startswith(FUSION_TRANSFORMER):
        from .fusion import MultimodalFusionTransformer

        model = functools.partial(
            MultimodalFusionTransformer,
            prefix=model_name,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            ffn_hidden_size=model_config.ffn_hidden_size,
            attention_dropout=model_config.attention_dropout,
            residual_dropout=model_config.residual_dropout,
            ffn_dropout=model_config.ffn_dropout,
            attention_normalization=model_config.normalization,
            ffn_normalization=model_config.normalization,
            head_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            head_activation=model_config.head_activation,
            adapt_in_features=model_config.adapt_in_features,
            aux_loss_weight=model_config.aux_loss_weight,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
        )
    elif model_name.lower().startswith(FT_TRANSFORMER):
        from .ft_transformer import FT_Transformer

        model = FT_Transformer(
            prefix=model_name,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            embedding_arch=model_config.embedding_arch,
            token_dim=model_config.token_dim,
            hidden_size=model_config.hidden_size,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            attention_dropout=model_config.attention_dropout,
            attention_normalization=model_config.normalization,
            ffn_hidden_size=model_config.ffn_hidden_size,
            ffn_dropout=model_config.ffn_dropout,
            ffn_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            residual_dropout=model_config.residual_dropout,
            head_normalization=model_config.normalization,
            head_activation=model_config.head_activation,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
            pooling_mode=model_config.pooling_mode,
            checkpoint_name=model_config.checkpoint_name,
            pretrained=pretrained,
        )
    elif model_name.lower().startswith(SAM):
        from .sam import SAMForSemanticSegmentation

        model = SAMForSemanticSegmentation(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            frozen_layers=model_config.frozen_layers,
            num_mask_tokens=model_config.num_mask_tokens,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(META_TRANSFORMER):
        from .meta_transformer import MetaTransformer

        model = MetaTransformer(
            prefix=model_name,
            checkpoint_path=model_config.checkpoint_path,
            num_classes=num_classes,
            model_version=model_config.model_version,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return model


def create_fusion_model(
    config: DictConfig,
    num_classes: Optional[int] = None,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
):
    """
    Create models. It supports the auto models of huggingface text and timm image.
    Multimodal models, e.g., CLIP, should be added case-by-case since their configs and usages
    may be different. It uses MLP for the numerical features, categorical features, and late-fusion.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A Pytorch model.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    logger.debug(f"output_shape: {num_classes}")
    names = sorted(names)
    config.model.names = names
    single_models = []
    fusion_model = None

    for model_name in names:
        model_config = getattr(config.model, model_name)
        model = create_model(
            model_name=model_name,
            model_config=model_config,
            num_classes=num_classes,
            classes=classes,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            pretrained=pretrained,
        )

        if isinstance(model, functools.partial):  # fusion model
            if fusion_model is None:
                fusion_model = model
            else:
                raise ValueError(
                    f"More than one fusion models are detected in {names}. Only one fusion model is allowed."
                )
        else:  # single model
            if config.optim.peft is not None:
                model = apply_peft_adaptation(model, config)
            single_models.append(model)

    if len(single_models) > 1:
        # must have one fusion model if there are multiple independent models
        model = fusion_model(models=single_models)
    elif len(single_models) == 1:
        model = single_models[0]
    else:
        raise ValueError(f"No available models for {names}")

    # build augmenter for multimodal data augmentation
    if config.optim.lemda.turn_on:
        from .fusion import MultimodalFusionMLP

        assert isinstance(model, MultimodalFusionMLP)
        from .augmenter import Augmenter

        augmenter = Augmenter(
            arch_type=config.optim.lemda.arch_type,
            input_dim=model.augmenter_in_features,
            z_dim=config.optim.lemda.z_dim,
            num_layers=config.optim.lemda.num_layers,
            adv_weight=config.optim.lemda.adv_weight,
        )
        model.augmenter = augmenter

    return model


def apply_peft_adaptation(model: nn.Module, config: DictConfig) -> nn.Module:
    """
    Apply an adaptation to the model for efficient fine-tuning.

    Parameters
    ----------
    model
        A PyTorch model.
    config:
        A DictConfig object. The optimization config should be accessible by "config.optimization".
    """
    if config.optim.peft in PEFT_ADDITIVE_STRATEGIES:
        model = inject_adaptation_to_linear_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        model.name_to_id = model.get_layer_ids()  # Need to update name to id dictionary.

    return model


def modify_duplicate_model_names(
    learner,
    postfix: str,
    blacklist: List[str],
):
    """
    Modify a learner's model names if they exist in a blacklist.

    Parameters
    ----------
    learner
        A BaseLearner object.
    postfix
        The postfix used to change the duplicate names.
    blacklist
        A list of names. The provided learner can't use model names in the list.

    Returns
    -------
    The learner guaranteed has no duplicate model names with the blacklist names.
    """
    model_names = []
    for n in learner._config.model.names:
        if n in blacklist:
            new_name = f"{n}_{postfix}"
            assert new_name not in blacklist
            assert new_name not in learner._config.model.names
            # modify model prefix
            if n == learner._model.prefix:
                learner._model.prefix = new_name
            else:
                assert isinstance(learner._model.model, nn.ModuleList)
                for per_model in learner._model.model:
                    if n == per_model.prefix:
                        per_model.prefix = new_name
                        break
            # modify data processor prefix
            for per_modality_processors in learner._data_processors.values():
                for per_processor in per_modality_processors:
                    if n == per_processor.prefix:
                        per_processor.prefix = new_name
            # modify model config keys
            setattr(learner._config.model, new_name, getattr(learner._config.model, n))
            delattr(learner._config.model, n)

            model_names.append(new_name)
        else:
            model_names.append(n)

    learner._config.model.names = model_names

    return learner


def list_timm_models(pretrained=True):
    return timm.list_models(pretrained=pretrained)


def is_lazy_weight_tensor(p: torch.Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate."
        )
        return True
    return False

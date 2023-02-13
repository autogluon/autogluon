import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import AutoConfig, AutoModel

from ..constants import AUTOMM, LOGITS, REGRESSION
from .adaptation_layers import IA3Linear, IA3LoRALinear, LoRALinear

logger = logging.getLogger(__name__)


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


def split_encoder_non_encoder(names: List[str]):
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
        child_name = n[len(model_prefix) + 1 :].split(".")[0]
        child_prefix = f"{model_prefix}.{child_name}"
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)

    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group)
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


def create_adaptation(efficient_finetune: str, layer: nn.Module, lora_r: int, lora_alpha: int):
    """
    Creates a model adaptation module (IA3, LoRA, IA3_LoRA) given a linear layer.

    Parameters
    ----------
    efficient_finetune
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
    if "ia3_lora" in efficient_finetune:
        return IA3LoRALinear(
            layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False
        )
    elif "ia3" in efficient_finetune:
        return IA3Linear(layer.in_features, layer.out_features, merge_weights=False)
    elif "lora" in efficient_finetune:
        return LoRALinear(layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
    elif efficient_finetune is not None and efficient_finetune != "None":
        raise NotImplementedError(
            f"The efficient finetuning strategy '{efficient_finetune}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_STRATEGIES)}."
        )


def inject_adaptation_to_linear_layer(
    model: nn.Module,
    efficient_finetune: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
) -> nn.Module:
    """
    Injects trainable adatio Low-Rank decomposition matrices (LoRA) into linear
    layers of a PyTorch model. Used for efficient fine-tuning of large
    pre-trained models.

    Parameters
    ----------
    model
        A PyTorch model.
    efficient_finetune
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

    Returns
    -------
    Model with injected LoRA modules.
    """
    for m_name, module in dict(model.named_modules()).items():
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    adaptation_layer = create_adaptation(efficient_finetune, layer, lora_r, lora_alpha)
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
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import mmcv
        from mmcv.runner import load_checkpoint
    except ImportError as e:
        warnings.warn(f"Encountered error while import mmcv: {e}")
        mmcv = None
    try:
        import mmocr
        from mmocr.models import build_detector
    except ImportError:
        mmocr = None
    from mim.commands.download import download

    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")

    # read config files
    assert mmcv is not None, "Please install mmcv-full by: mim install mmcv-full."
    config_file = checkpoint_name + ".py"
    if isinstance(config_file, str):
        config = mmcv.Config.fromfile(config_file)

    # build model and load pretrained weights
    assert mmocr is not None, "Please install MMOCR by: pip install mmocr."

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


def run_model(model: nn.Module, batch: dict):
    from .document_transformer import DocumentTransformer
    from .huggingface_text import HFAutoModelForTextPrediction
    from .timm_image import TimmAutoModelForImagePrediction

    supported_models = (TimmAutoModelForImagePrediction, HFAutoModelForTextPrediction)
    pure_model = model.module if isinstance(model, nn.DataParallel) else model
    if (not isinstance(pure_model, DocumentTransformer)) and isinstance(pure_model, supported_models):
        input_vec = [batch[k] for k in pure_model.input_keys]
        column_names, column_values = [], []
        for k in batch.keys():
            if (
                isinstance(pure_model, TimmAutoModelForImagePrediction)
                and k.startswith(pure_model.image_column_prefix)
            ) or (
                isinstance(pure_model, HFAutoModelForTextPrediction) and k.startswith(pure_model.text_column_prefix)
            ):
                column_names.append(k)
                column_values.append(batch[k])
        input_vec.append(column_names)
        input_vec.append(column_values)
        output_vec = model(*tuple(input_vec))
        output = pure_model.get_output_dict(*output_vec)
    else:
        output = model(batch)
    return output

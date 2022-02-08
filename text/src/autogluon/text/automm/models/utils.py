from typing import Optional, Union, Tuple, List, Dict
from torch import nn


def init_weights(module: nn.Module):
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
        layer_keys: Optional[Tuple[str, ...]] = None,
):
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split('.')

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
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: List[str]):
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
        enc_prefix: Optional[str] = None,
        model_prefix: Optional[str] = None,
):

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
        child_name = n[len(model_prefix)+1:].split(".")[0]
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
        enc_pre: Optional[str] = None,
        model_pre: Optional[str] = None,
        layer_keys: Optional[Tuple[str, ...]] = None,
):
    left_names, encoder_names, pre_encoder_names, post_encoder_names = \
        group_param_names(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            enc_prefix=enc_pre,
            model_prefix=model_pre,
        )
    # add a constraint
    if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
        raise ValueError(
            f"encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}"
        )

    encoder_name_to_id, encoder_layer_num = \
        assign_encoder_layer_ids(
            encoder_names=encoder_names,
            layer_keys=layer_keys
        )

    pre_encoder_name_to_id = \
        assign_non_encoder_layer_ids(
            non_encoder_names=pre_encoder_names,
            layer_id=0
        )

    post_encoder_name_to_id = \
        assign_non_encoder_layer_ids(
            non_encoder_names=post_encoder_names,
            layer_id=encoder_layer_num + 1
        )

    name_to_id = reverse_layer_ids(
        encoder_name_to_id=encoder_name_to_id,
        pre_enocder_name_to_id=pre_encoder_name_to_id,
        post_enocder_name_to_id=post_encoder_name_to_id
    )
    return name_to_id, left_names

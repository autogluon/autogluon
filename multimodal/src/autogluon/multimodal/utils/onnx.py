import logging
from typing import Dict, List, Optional, Tuple, Union

from torch import tensor

from ..constants import AUTOMM, FEATURE_EXTRACTION, MULTICLASS

logger = logging.getLogger(__name__)


def get_onnx_input(pipeline: str, config: Optional[Dict] = None):
    """
    Get information for a predictor to export its model in onnx format.

    Parameters
    ----------
    pipeline
        The predictor's pipeline.
    config
        The predictor's config.

    Returns
    -------
    valid_input
        The valid keys for the input batch.
    dynamic_axes
        By default the exported model will have the shapes of all input and output tensors
        set to exactly match those given in ``args``. To specify axes of tensors as
        dynamic (i.e. known only at run-time), set ``dynamic_axes`` to a dict with schema:
            * KEY (str): an input or output name. Each name must also be provided in ``input_names`` or
              ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
              list, each element is an axis index.
        See torch.onnx.export for more explanations.
    default_onnx_path
        The default path of the export onnx model.
    default_batch
        The default batch to help trace and export the model.

    """
    default_onnx_path = None
    if pipeline == FEATURE_EXTRACTION:
        valid_input = [
            "hf_text_text_token_ids",
            "hf_text_text_segment_ids",
            "hf_text_text_valid_length",
        ]
        dynamic_axes = {
            "hf_text_text_token_ids": {
                0: "batch_size",
                1: "sentence_length",
            },
            "hf_text_text_valid_length": {
                0: "batch_size",
            },
            "hf_text_text_segment_ids": {
                0: "batch_size",
                1: "sentence_length",
            },
        }
        if config:
            default_onnx_path = config["model"]["hf_text"]["checkpoint_name"].replace("/", "_") + ".onnx"
        default_batch = {
            "hf_text_text_token_ids": tensor(
                [
                    [101, 1037, 2158, 2003, 2652, 2858, 1012, 102, 0, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 2652, 1037, 2858, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 2652, 1037, 2858, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 6276, 2019, 20949, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 9670, 1012, 102, 0, 0, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 26514, 2330, 1037, 3869, 1012, 102, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 26514, 1037, 20856, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 2652, 1037, 2858, 1012, 102, 0, 0, 0, 0, 0],
                ],
            ),
            "hf_text_text_valid_length": tensor(
                [8, 9, 9, 9, 7, 10, 9, 9],
            ),
            "hf_text_text_segment_ids": tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
        }
    elif pipeline == MULTICLASS:
        valid_input = ["timm_image_image", "timm_image_image_valid_num"]
        dynamic_axes = {
            "timm_image_image": {
                0: "batch_size",
            },
            "timm_image_image_valid_num": {},
        }
        default_onnx_path = "./model.onnx"
        default_batch = None
    else:
        raise ValueError(f"ONNX export is not supported in current pipeline {pipeline}")

    return valid_input, dynamic_axes, default_onnx_path, default_batch

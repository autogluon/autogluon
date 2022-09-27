import logging
from torch import tensor
from typing import Dict, List, Optional, Tuple, Union

from ..constants import AUTOMM, FEATURE_EXTRACTION

logger = logging.getLogger(AUTOMM)


def get_onnx_input(pipeline, config=None):
    onnx_path = None
    if pipeline == FEATURE_EXTRACTION:
        valid_input = [
            "hf_text_text_token_ids",
            "hf_text_text_valid_length",
            "hf_text_text_segment_ids",
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
            onnx_path = config["model"]["hf_text"]["checkpoint_name"].replace("/", "_") + ".onnx"
        default_batch = {
            "hf_text_text_token_ids_column_sentence1": tensor(
                [[1, 7], [1, 8], [1, 8], [1, 8], [1, 6], [1, 9], [1, 8], [1, 8]],
            ),
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
            "hf_text_text_valid_length": tensor([8, 9, 9, 9, 7, 10, 9, 9],),
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
    else:
        raise ValueError(f"ONNX export is not supported in current pipeline {pipeline}")

    return valid_input, dynamic_axes, onnx_path, default_batch

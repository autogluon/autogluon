# flake8: noqa
import base64
import copy
import hashlib
import os
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from PIL import Image

from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.multimodal import MultiModalPredictor


def _cleanup_images():
    files = os.listdir(".")
    for file in files:
        if file.endswith(".png"):
            os.remove(file)


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = MultiModalPredictor.load(model_dir)
    label_column = model._label_column
    column_types = copy.copy(model._column_types)
    column_types.pop(label_column)
    globals()["column_names"] = list(column_types.keys())

    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    image_bytearrays = None
    if input_content_type == "application/x-parquet":
        buf = BytesIO(request_body)
        data = pd.read_parquet(buf)

    elif input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf)

    elif input_content_type == "application/json":
        buf = StringIO(request_body)
        data = pd.read_json(buf)

    elif input_content_type == "application/jsonl":
        buf = StringIO(request_body)
        data = pd.read_json(buf, orient="records", lines=True)

    elif input_content_type == "application/x-npy":
        buf = BytesIO(request_body)
        data = np.load(buf, allow_pickle=True)
        image_bytearrays = []
        for bytes in data:
            im_bytes = base64.b85decode(bytes)
            image_bytearrays.append(im_bytes)

    elif input_content_type == "application/x-image":
        image_bytearrays.append(request_body)

    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    if image_bytearrays is not None:
        data = dict(image=image_bytearrays)  # multimodal image prediction takes in a dict containing image bytearrays
    else:
        # no header
        test_columns = sorted(list(data.columns))
        train_columns = sorted(column_names)
        if test_columns != train_columns:
            num_cols = len(data.columns)

            if num_cols != len(column_names):
                raise Exception(
                    f"Invalid data format. Input data has {num_cols} while the model expects {len(column_names)}"
                )

            else:
                new_row = pd.DataFrame(data.columns).transpose()
                old_rows = pd.DataFrame(data.values)
                data = pd.concat([new_row, old_rows]).reset_index(drop=True)
                data.columns = column_names
        # find image column
        image_column = None
        for column_name, column_type in model._column_types.items():
            if column_type in ("image_path", "image", "image_bytearray"):
                image_column = column_name
                break
        if image_column is not None:
            print(f"Detected image column {image_column}")
            data[image_column] = [base64.b85decode(bytes) for bytes in data[image_column]]

    if model.problem_type == BINARY or model.problem_type == MULTICLASS:
        pred_proba = model.predict_proba(data, as_pandas=True)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
        pred.name = str(pred.name) + "_pred" if pred.name is not None else "pred"
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(data, as_pandas=True)

    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()
    if output_content_type == "application/json":
        output = prediction.to_json()
    elif output_content_type == "application/x-parquet":
        if isinstance(prediction, pd.DataFrame):
            prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
    elif output_content_type == "text/csv":
        output = prediction.to_csv(index=None)
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    _cleanup_images()

    return output, output_content_type

import base64
import copy
import hashlib
import os
import pandas as pd
import numpy as np

from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.multimodal import MultiModalPredictor

from io import BytesIO, StringIO
from PIL import Image


def _save_image_and_update_dataframe_column(bytes):
    im_bytes = base64.b85decode(bytes)
    im_hash = hashlib.md5(im_bytes).hexdigest()
    im = Image.open(BytesIO(im_bytes))
    im_name = f'multimodal_image_{im_hash}.png'
    im.save(im_name)
    print(f'Image saved as {im_name}')

    return im_name


def _cleanup_images():
    files = os.listdir('.')
    for file in files:
        if file.endswith('.png'):
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
    image_paths = None
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
        data = pd.read_json(buf, orient='records', lines=True)

    elif input_content_type == "application/x-npy":
        buf = BytesIO(request_body)
        data = np.load(buf, allow_pickle=True)
        image_paths = []
        for bytes in data:
            im_bytes = base64.b85decode(bytes)
            im_name = hashlib.md5(im_bytes).hexdigest()
            im = Image.open(BytesIO(im_bytes))
            im.save(im_name)
            image_paths.append(im_name)

    elif input_content_type == "application/x-image":
        buf = BytesIO(request_body)
        im = Image.open(buf)
        image_paths = []
        im_name = 'test.png'
        im.save(im_name)
        image_paths.append(im_name)

    else:
        raise ValueError(
            f'{input_content_type} input content type not supported.'
        )

    if image_paths is not None:
        data = dict(image=image_paths)  # multimodal image prediction takes in a dict containing image paths
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
            if column_type in ('image_path', 'image'):
                image_column = column_name
                break
        # save image column bytes to disk and update the column with saved path
        if image_column is not None:
            print(f'Detected image column {image_column}')
            data[image_column] = [_save_image_and_update_dataframe_column(bytes) for bytes in data[image_column]]

    if model.problem_type == BINARY or model.problem_type == MULTICLASS:
        pred_proba = model.predict_proba(data, as_pandas=True)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        pred_proba.columns = [str(c) + '_proba' for c in pred_proba.columns]
        pred.name = str(pred.name) + '_pred' if pred.name is not None else 'pred'
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

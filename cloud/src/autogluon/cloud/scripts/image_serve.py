import base64
import hashlib
import os
import pandas as pd
import numpy as np

from autogluon.core.constants import REGRESSION
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.vision import ImagePredictor
from io import BytesIO
from PIL import Image


def _cleanup_images():
    files = os.listdir('.')
    for file in files:
        if file.endswith('.png'):
            os.remove(file)


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = ImagePredictor.load(model_dir)

    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):

    if input_content_type == "application/x-npy":
        buf = BytesIO(request_body)
        data = np.load(buf, allow_pickle=True)
        image_paths = []
        for bytes in data:
            im_bytes = base64.b85decode(bytes)
            im_hash = hashlib.md5(im_bytes).hexdigest()
            im_name = f'image_{im_hash}.png'
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

    if model._problem_type != REGRESSION:
        pred_proba = model.predict_proba(image_paths, as_pandas=True)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model._problem_type)
        pred_proba.columns = [str(c) + '_proba' for c in pred_proba.columns]
        pred.name = str(pred.name) + '_pred' if pred.name is not None else 'pred'
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(image_paths, as_pandas=True)
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    if output_content_type == "application/json":
        output = prediction.to_json()
    elif output_content_type == "application/x-parquet":
        prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
    elif output_content_type == "text/csv":
        output = prediction.to_csv(index=None)
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    _cleanup_images()

    return output, output_content_type
